//! Shared single-file storage for parked resource bytes.
//!
//! The allocation policy follows Blink's `DiskDataAllocator`: allocations are
//! `(offset, len)` extents in one anonymous temporary file, exact free extents
//! are preferred, otherwise the largest fitting extent is split, and adjacent
//! free extents are coalesced on release. Ported from moli's `moli-disk-pool`.

use std::{
    collections::{BTreeMap, BTreeSet},
    fmt,
    fs::File,
    io,
    sync::Arc,
};

use parking_lot::Mutex;

#[cfg(not(any(unix, windows)))]
type PoolFile = Mutex<File>;
#[cfg(any(unix, windows))]
type PoolFile = File;

#[derive(Debug, Default)]
struct AllocationState {
    may_write: bool,
    file_tail: u64,
    free_chunks_by_offset: BTreeMap<u64, usize>,
    free_chunks_by_size: BTreeSet<(usize, u64)>,
    free_bytes: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Extent {
    offset: u64,
    len: usize,
}

impl AllocationState {
    fn insert_free_chunk(&mut self, extent: Extent) {
        let previous = self.free_chunks_by_offset.insert(extent.offset, extent.len);
        debug_assert!(previous.is_none());
        let inserted = self.free_chunks_by_size.insert((extent.len, extent.offset));
        debug_assert!(inserted);
    }

    fn remove_free_chunk(&mut self, extent: Extent) {
        let removed = self.free_chunks_by_offset.remove(&extent.offset);
        debug_assert_eq!(removed, Some(extent.len));
        let removed = self
            .free_chunks_by_size
            .remove(&(extent.len, extent.offset));
        debug_assert!(removed);
    }
}

fn find_free_chunk(state: &mut AllocationState, size: usize) -> Option<Extent> {
    let exact_fit = state
        .free_chunks_by_size
        .range((size, 0)..=(size, u64::MAX))
        .next()
        .copied();
    let (chunk_size, offset) = if let Some(exact_fit) = exact_fit {
        exact_fit
    } else {
        let &(largest_size, _) = state.free_chunks_by_size.last()?;
        if largest_size <= size {
            return None;
        }
        // Keep the lowest offset when worst-fit chunks share the same size.
        state
            .free_chunks_by_size
            .range((largest_size, 0)..=(largest_size, u64::MAX))
            .next()
            .copied()
            .expect("largest free chunk size should remain indexed")
    };

    let mut chosen = Extent {
        offset,
        len: chunk_size,
    };
    let allocation_len = u64::try_from(size).ok()?;
    let remainder_offset = chosen
        .offset
        .checked_add(allocation_len)
        .expect("free disk extent should not overflow");
    state.remove_free_chunk(chosen);
    state.free_bytes -= size;
    if chosen.len > size {
        state.insert_free_chunk(Extent {
            offset: remainder_offset,
            len: chosen.len - size,
        });
        chosen.len = size;
    }
    Some(chosen)
}

fn release_chunk(state: &mut AllocationState, extent: Extent) {
    let original_len = extent.len;
    let mut merged = extent;

    if let Some((&left_offset, &left_len)) = state
        .free_chunks_by_offset
        .range(..merged.offset)
        .next_back()
    {
        let left_end = left_offset
            .checked_add(u64::try_from(left_len).expect("extent length should fit u64"))
            .expect("free disk extent should not overflow");
        debug_assert!(left_end <= merged.offset);
        if left_end == merged.offset {
            state.remove_free_chunk(Extent {
                offset: left_offset,
                len: left_len,
            });
            merged.offset = left_offset;
            merged.len = merged
                .len
                .checked_add(left_len)
                .expect("merged disk extent should fit usize");
        }
    }

    if let Some((&right_offset, &right_len)) =
        state.free_chunks_by_offset.range(merged.offset..).next()
    {
        let merged_end = merged
            .offset
            .checked_add(u64::try_from(merged.len).expect("extent length should fit u64"))
            .expect("free disk extent should not overflow");
        debug_assert!(merged_end <= right_offset);
        if merged_end == right_offset {
            state.remove_free_chunk(Extent {
                offset: right_offset,
                len: right_len,
            });
            merged.len = merged
                .len
                .checked_add(right_len)
                .expect("merged disk extent should fit usize");
        }
    }

    state.insert_free_chunk(merged);
    state.free_bytes = state
        .free_bytes
        .checked_add(original_len)
        .expect("free disk bytes should fit usize");
}

fn read_exact_at(file: &PoolFile, mut buffer: &mut [u8], mut offset: u64) -> io::Result<()> {
    while !buffer.is_empty() {
        match read_at(file, buffer, offset) {
            Ok(0) => {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "disk pool file ended inside an allocated extent",
                ));
            }
            Ok(read) => {
                offset = offset
                    .checked_add(u64::try_from(read).map_err(|_| {
                        io::Error::new(io::ErrorKind::InvalidInput, "read size is too large")
                    })?)
                    .ok_or_else(|| {
                        io::Error::new(io::ErrorKind::InvalidInput, "read offset overflow")
                    })?;
                buffer = &mut buffer[read..];
            }
            Err(error) if error.kind() == io::ErrorKind::Interrupted => {}
            Err(error) => return Err(error),
        }
    }
    Ok(())
}

fn write_all_at(file: &PoolFile, mut data: &[u8], mut offset: u64) -> io::Result<()> {
    while !data.is_empty() {
        match write_at(file, data, offset) {
            Ok(0) => {
                return Err(io::Error::new(
                    io::ErrorKind::WriteZero,
                    "failed to write an allocated disk extent",
                ));
            }
            Ok(written) => {
                offset = offset
                    .checked_add(u64::try_from(written).map_err(|_| {
                        io::Error::new(io::ErrorKind::InvalidInput, "write size is too large")
                    })?)
                    .ok_or_else(|| {
                        io::Error::new(io::ErrorKind::InvalidInput, "write offset overflow")
                    })?;
                data = &data[written..];
            }
            Err(error) if error.kind() == io::ErrorKind::Interrupted => {}
            Err(error) => return Err(error),
        }
    }
    Ok(())
}

#[cfg(unix)]
fn read_at(file: &PoolFile, buffer: &mut [u8], offset: u64) -> io::Result<usize> {
    use std::os::unix::fs::FileExt;

    file.read_at(buffer, offset)
}

#[cfg(unix)]
fn write_at(file: &PoolFile, data: &[u8], offset: u64) -> io::Result<usize> {
    use std::os::unix::fs::FileExt;

    file.write_at(data, offset)
}

#[cfg(windows)]
fn read_at(file: &PoolFile, buffer: &mut [u8], offset: u64) -> io::Result<usize> {
    use std::os::windows::fs::FileExt;

    file.seek_read(buffer, offset)
}

#[cfg(windows)]
fn write_at(file: &PoolFile, data: &[u8], offset: u64) -> io::Result<usize> {
    use std::os::windows::fs::FileExt;

    file.seek_write(data, offset)
}

#[cfg(not(any(unix, windows)))]
fn read_at(file: &PoolFile, buffer: &mut [u8], offset: u64) -> io::Result<usize> {
    use std::io::{Read, Seek};

    let mut file = file.lock();
    file.seek(io::SeekFrom::Start(offset))?;
    file.read(buffer)
}

#[cfg(not(any(unix, windows)))]
fn write_at(file: &PoolFile, data: &[u8], offset: u64) -> io::Result<usize> {
    use std::io::{Seek, Write as _};

    let mut file = file.lock();
    file.seek(io::SeekFrom::Start(offset))?;
    file.write(data)
}

/// A thread-safe allocator over one anonymous temporary file.
#[derive(Clone)]
pub struct DiskPool {
    inner: Arc<DiskPoolInner>,
}

struct DiskPoolInner {
    file: PoolFile,
    max_capacity: Option<u64>,
    state: Mutex<AllocationState>,
}

/// A point-in-time snapshot of allocator state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DiskPoolDiagnostics {
    pub may_write: bool,
    pub disk_footprint_bytes: u64,
    pub free_bytes: usize,
    pub free_chunk_count: usize,
}

impl DiskPool {
    /// Creates a pool backed by one delete-on-close temporary file.
    ///
    /// `max_capacity` limits the high-water file footprint. Released extents
    /// remain reusable within that limit, matching Blink's allocator.
    pub fn new(max_capacity: Option<u64>) -> io::Result<Self> {
        let file = tempfile::tempfile()?;
        Ok(Self {
            inner: Arc::new(DiskPoolInner {
                file: pool_file(file),
                max_capacity,
                state: Mutex::new(AllocationState {
                    may_write: true,
                    ..AllocationState::default()
                }),
            }),
        })
    }

    /// Whether future reservations may succeed. Deliberately only a hint: a
    /// later disk write can still fail.
    pub fn may_write(&self) -> bool {
        self.inner.state.lock().may_write
    }

    /// Reserves an extent, preferring an exact free extent and then the
    /// largest fitting free extent before growing the file tail.
    pub fn try_reserve_chunk(&self, size: usize) -> Option<ReservedChunk> {
        if size == 0 {
            return None;
        }

        let mut state = self.inner.state.lock();
        if !state.may_write {
            return None;
        }

        let extent = find_free_chunk(&mut state, size).or_else(|| {
            let size = u64::try_from(size).ok()?;
            let new_tail = state.file_tail.checked_add(size)?;
            if self
                .inner
                .max_capacity
                .is_some_and(|capacity| new_tail > capacity)
            {
                return None;
            }
            let extent = Extent {
                offset: state.file_tail,
                len: usize::try_from(size).ok()?,
            };
            state.file_tail = new_tail;
            Some(extent)
        })?;

        Some(ReservedChunk::new(self.clone(), extent))
    }

    /// Writes exactly one reserved extent.
    ///
    /// As in Blink, a disk write failure releases the extent and disables
    /// later writes because failures such as a full disk are rarely transient.
    pub fn write(&self, mut chunk: ReservedChunk, data: &[u8]) -> io::Result<DiskData> {
        if !chunk.belongs_to(self) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "reserved chunk belongs to another disk pool",
            ));
        }

        let extent = chunk.take_extent();
        if extent.len != data.len() {
            self.release(extent);
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "data length does not match reserved disk extent",
            ));
        }

        if let Err(error) = write_all_at(&self.inner.file, data, extent.offset) {
            let mut state = self.inner.state.lock();
            release_chunk(&mut state, extent);
            state.may_write = false;
            return Err(error);
        }

        Ok(DiskData::new(self.clone(), extent))
    }

    /// Reserves and writes one extent.
    ///
    /// `Ok(None)` means the allocator is disabled or has reached its capacity;
    /// an `Err` means an attempted disk write failed.
    #[cfg(test)]
    fn store(&self, data: &[u8]) -> io::Result<Option<DiskData>> {
        let Some(chunk) = self.try_reserve_chunk(data.len()) else {
            return Ok(None);
        };
        self.write(chunk, data).map(Some)
    }

    pub fn diagnostics(&self) -> DiskPoolDiagnostics {
        let state = self.inner.state.lock();
        DiskPoolDiagnostics {
            may_write: state.may_write,
            disk_footprint_bytes: state.file_tail,
            free_bytes: state.free_bytes,
            free_chunk_count: state.free_chunks_by_offset.len(),
        }
    }

    fn shares_state_with(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }

    fn release(&self, extent: Extent) {
        release_chunk(&mut self.inner.state.lock(), extent);
    }

    fn read_exact_file_at(&self, buffer: &mut [u8], offset: u64) -> io::Result<()> {
        read_exact_at(&self.inner.file, buffer, offset)
    }
}

impl fmt::Debug for DiskPool {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("DiskPool")
            .field("diagnostics", &self.diagnostics())
            .finish_non_exhaustive()
    }
}

#[cfg(any(unix, windows))]
fn pool_file(file: File) -> PoolFile {
    file
}

#[cfg(not(any(unix, windows)))]
fn pool_file(file: File) -> PoolFile {
    Mutex::new(file)
}

/// A reserved extent that is automatically returned unless it is written.
pub struct ReservedChunk {
    pool: DiskPool,
    extent: Option<Extent>,
}

impl ReservedChunk {
    fn new(pool: DiskPool, extent: Extent) -> Self {
        Self {
            pool,
            extent: Some(extent),
        }
    }

    fn belongs_to(&self, pool: &DiskPool) -> bool {
        self.pool.shares_state_with(pool)
    }

    fn take_extent(&mut self) -> Extent {
        self.extent
            .take()
            .expect("reserved disk chunk should contain an extent")
    }
}

impl fmt::Debug for ReservedChunk {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ReservedChunk")
            .field("extent", &self.extent)
            .finish_non_exhaustive()
    }
}

impl Drop for ReservedChunk {
    fn drop(&mut self) {
        if let Some(extent) = self.extent.take() {
            self.pool.release(extent);
        }
    }
}

/// Immutable data stored in one pool extent.
///
/// Dropping this value returns its extent to the pool. The pool itself remains
/// alive for as long as any stored data refers to it.
pub struct DiskData {
    pool: DiskPool,
    extent: Extent,
}

impl DiskData {
    fn new(pool: DiskPool, extent: Extent) -> Self {
        Self { pool, extent }
    }

    #[cfg(test)]
    fn offset(&self) -> u64 {
        self.extent.offset
    }

    /// Reads an exact range relative to this extent.
    pub fn read_exact_at(&self, offset: usize, buffer: &mut [u8]) -> io::Result<()> {
        let end = offset.checked_add(buffer.len()).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidInput, "disk data range overflow")
        })?;
        if end > self.extent.len {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "disk data read exceeds extent",
            ));
        }
        let absolute_offset = self
            .extent
            .offset
            .checked_add(u64::try_from(offset).map_err(|_| {
                io::Error::new(io::ErrorKind::InvalidInput, "disk data offset is too large")
            })?)
            .ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidInput, "disk data offset overflow")
            })?;
        self.pool.read_exact_file_at(buffer, absolute_offset)
    }

    pub fn to_vec(&self) -> io::Result<Vec<u8>> {
        let mut data = vec![0; self.extent.len];
        self.read_exact_at(0, &mut data)?;
        Ok(data)
    }
}

impl fmt::Debug for DiskData {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("DiskData")
            .field("offset", &self.extent.offset)
            .field("len", &self.extent.len)
            .finish_non_exhaustive()
    }
}

impl Drop for DiskData {
    fn drop(&mut self) {
        self.pool.release(self.extent);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn store_read_release_reuses_extent() {
        let pool = DiskPool::new(None).unwrap();
        let first = pool.store(b"hello").unwrap().unwrap();
        assert_eq!(first.to_vec().unwrap(), b"hello");
        let offset = first.offset();
        drop(first);
        let second = pool.store(b"world").unwrap().unwrap();
        assert_eq!(second.offset(), offset, "exact free fit should be reused");
        assert_eq!(second.to_vec().unwrap(), b"world");
    }

    #[test]
    fn released_extents_coalesce() {
        let pool = DiskPool::new(None).unwrap();
        let a = pool.store(&[1u8; 8]).unwrap().unwrap();
        let b = pool.store(&[2u8; 8]).unwrap().unwrap();
        let c = pool.store(&[3u8; 8]).unwrap().unwrap();
        drop(b);
        drop(a);
        drop(c);
        let diagnostics = pool.diagnostics();
        assert_eq!(diagnostics.free_bytes, 24);
        assert_eq!(diagnostics.free_chunk_count, 1, "adjacent extents merge");
    }

    #[test]
    fn larger_request_splits_largest_free_chunk() {
        let pool = DiskPool::new(None).unwrap();
        let big = pool.store(&[7u8; 16]).unwrap().unwrap();
        let offset = big.offset();
        drop(big);
        let small = pool.store(&[9u8; 4]).unwrap().unwrap();
        assert_eq!(small.offset(), offset);
        let diagnostics = pool.diagnostics();
        assert_eq!(diagnostics.free_bytes, 12);
    }

    #[test]
    fn capacity_limit_stops_reservations() {
        let pool = DiskPool::new(Some(8)).unwrap();
        let kept = pool.store(&[1u8; 8]).unwrap().unwrap();
        assert!(pool.store(&[2u8; 1]).unwrap().is_none());
        drop(kept);
        assert!(pool.store(&[3u8; 8]).unwrap().is_some());
    }
}
