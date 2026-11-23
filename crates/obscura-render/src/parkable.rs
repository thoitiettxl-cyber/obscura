//! Parkable encoded resource bytes.
//!
//! Completed resource bytes are read through snapshots and may be parked into
//! one `DiskPool` extent so an idle page's image/font cache can give memory
//! back without evicting entries. Reading a parked value unparks it
//! synchronously; the retained disk extent makes a later park a memory-only
//! transition. Ported from moli's `moli-parkable-image`, adapted to obscura's
//! `Arc<[u8]>` resource bytes and driven opportunistically from
//! `RenderResourceCache` instead of a runtime-scheduled sweep.

use std::{
    collections::HashMap,
    fmt, io,
    ops::Deref,
    sync::{
        Arc, Weak,
        atomic::{AtomicU64, Ordering},
    },
    time::{Duration, Instant},
};

use parking_lot::Mutex;

use crate::disk_pool::{DiskPool, ReservedChunk};

const DEFAULT_MIN_SIZE_TO_PARK: usize = 1024;
const DEFAULT_PARKING_DELAY: Duration = Duration::from_secs(30);
const DEFAULT_READER_RELEASE_DELAY: Duration = Duration::from_secs(2);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ParkablePolicy {
    pub min_size_to_park: usize,
    /// How long unused frozen bytes remain resident.
    pub parking_delay: Duration,
    /// Grace period after the last reader releases the bytes.
    pub reader_release_delay: Duration,
}

impl Default for ParkablePolicy {
    fn default() -> Self {
        Self {
            min_size_to_park: DEFAULT_MIN_SIZE_TO_PARK,
            parking_delay: DEFAULT_PARKING_DELAY,
            reader_release_delay: DEFAULT_READER_RELEASE_DELAY,
        }
    }
}

/// Registry + disk pool owner for a page's parkable resource bytes.
#[derive(Clone)]
pub struct ParkableBytesManager {
    inner: Arc<ParkableBytesManagerInner>,
}

struct ParkableBytesManagerInner {
    disk_pool: Mutex<Option<DiskPool>>,
    policy: ParkablePolicy,
    next_id: AtomicU64,
    resident: Mutex<HashMap<u64, Weak<ParkableBytesInner>>>,
}

impl Default for ParkableBytesManager {
    fn default() -> Self {
        Self::new(ParkablePolicy::default())
    }
}

impl ParkableBytesManager {
    pub fn new(policy: ParkablePolicy) -> Self {
        Self {
            inner: Arc::new(ParkableBytesManagerInner {
                disk_pool: Mutex::new(None),
                policy,
                next_id: AtomicU64::new(0),
                resident: Mutex::new(HashMap::new()),
            }),
        }
    }

    /// Wraps bytes in a handle that is never registered: id 0, never parks.
    fn from_frozen_unmanaged(&self, bytes: Arc<[u8]>) -> ParkableBytes {
        ParkableBytes::new(self.clone(), 0, bytes)
    }

    /// Wraps bytes in a parkable handle. Values below the parking threshold
    /// stay unregistered so they can never be swept.
    pub fn from_frozen_bytes(&self, bytes: Arc<[u8]>) -> ParkableBytes {
        let id = self.inner.next_id.fetch_add(1, Ordering::Relaxed).saturating_add(1);
        let image = ParkableBytes::new(self.clone(), id, bytes);
        if image.inner.state.lock().len >= self.inner.policy.min_size_to_park {
            self.inner.resident.lock().insert(id, image.downgrade());
        }
        image
    }

    /// The earliest deadline among schedulable resident images.
    pub fn next_parking_deadline(&self) -> Option<Instant> {
        self.resident_images()
            .into_iter()
            .filter_map(|image| image.parking_deadline())
            .min()
    }

    /// Parks every resident image whose deadline has elapsed. Disk I/O is
    /// synchronous; callers should keep this off latency-critical paths.
    pub fn park_due(&self) {
        let now = Instant::now();
        for image in self.resident_images() {
            if image
                .parking_deadline()
                .is_some_and(|deadline| deadline <= now)
            {
                let _ = image.park();
            }
        }
    }

    fn resident_images(&self) -> Vec<ParkableBytes> {
        self.inner
            .resident
            .lock()
            .values()
            .filter_map(Weak::upgrade)
            .map(|inner| ParkableBytes { inner })
            .collect()
    }

    fn unregister(&self, id: u64) {
        self.inner.resident.lock().remove(&id);
    }

    fn disk_pool(&self) -> Option<DiskPool> {
        let mut pool = self.inner.disk_pool.lock();
        if pool.is_none() {
            *pool = DiskPool::new(None).ok();
        }
        pool.clone()
    }
}

/// Encoded bytes that can move between memory and one disk extent.
#[derive(Clone)]
pub struct ParkableBytes {
    inner: Arc<ParkableBytesInner>,
}

struct ParkableBytesInner {
    // Keep registration last: the storage (and any disk extent it owns) must
    // be dropped before unregistering wakes a capacity-waiting scheduler.
    state: Mutex<FrozenState>,
    registration: Registration,
}

struct Registration {
    id: u64,
    manager: ParkableBytesManager,
}

impl Drop for Registration {
    fn drop(&mut self) {
        self.manager.unregister(self.id);
    }
}

struct FrozenState {
    storage: FrozenStorage,
    park_after: Instant,
    len: usize,
    reader_count: usize,
}

enum FrozenStorage {
    /// Bytes are readable in memory. A disk backup, when present, makes the
    /// next park a memory-only state transition.
    Resident {
        bytes: Arc<[u8]>,
        disk_backup: Option<Arc<crate::disk_pool::DiskData>>,
    },
    /// The first disk write is running without holding the state lock.
    Parking { bytes: Arc<[u8]> },
    /// Resident bytes have been discarded; reads synchronously restore them.
    Parked {
        disk: Arc<crate::disk_pool::DiskData>,
    },
}

/// A read-only snapshot. While a snapshot exists, its image cannot discard
/// the corresponding in-memory bytes.
pub struct ParkableBytesSnapshot {
    // Field order is intentional: the byte reference is released before the
    // lease can make the image eligible for parking.
    data: Arc<[u8]>,
    /// Stable identity of the owning image, for raster-cache deduplication.
    owner_id: u64,
    lease: SnapshotLease,
}

impl ParkableBytesSnapshot {
    pub fn owner_id(&self) -> u64 {
        self.owner_id
    }
}

struct SnapshotLease {
    owner: Weak<ParkableBytesInner>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ParkableDiagnostics {
    pub len: usize,
    pub parked: bool,
    pub resident_bytes: usize,
}

impl ParkableBytes {
    fn new(manager: ParkableBytesManager, id: u64, bytes: Arc<[u8]>) -> Self {
        let park_after = Instant::now()
            .checked_add(manager.inner.policy.parking_delay)
            .expect("parking delay should fit Instant");
        Self {
            inner: Arc::new(ParkableBytesInner {
                state: Mutex::new(FrozenState {
                    len: bytes.len(),
                    storage: FrozenStorage::Resident {
                        bytes,
                        disk_backup: None,
                    },
                    park_after,
                    reader_count: 0,
                }),
                registration: Registration { id, manager },
            }),
        }
    }

    /// Stable identity for deduplication, unchanged across park/unpark cycles.
    /// Unmanaged values all share id 0; callers must not dedup on it.
    pub fn id(&self) -> u64 {
        self.inner.registration.id
    }

    /// Bytes outside any manager's registry (e.g. decoded `data:` URLs): they
    /// are never registered and can never park.
    pub(crate) fn unmanaged(bytes: Arc<[u8]>) -> Self {
        ParkableBytesManager::default().from_frozen_unmanaged(bytes)
    }

    pub fn len(&self) -> usize {
        self.inner.state.lock().len
    }

    /// Returns a zero-copy read-only snapshot of resident bytes. A parked
    /// image is synchronously read back into memory first.
    pub fn snapshot(&self) -> io::Result<ParkableBytesSnapshot> {
        let snapshot = {
            let mut frozen = self.inner.state.lock();

            if let FrozenStorage::Parked { disk } = &frozen.storage {
                let disk = Arc::clone(disk);
                let data: Arc<[u8]> = disk.to_vec()?.into();
                debug_assert_eq!(data.len(), frozen.len);
                frozen.storage = FrozenStorage::Resident {
                    bytes: Arc::clone(&data),
                    disk_backup: Some(disk),
                };
            }
            let bytes = match &frozen.storage {
                FrozenStorage::Resident { bytes, .. }
                | FrozenStorage::Parking { bytes, .. } => Arc::clone(bytes),
                FrozenStorage::Parked { .. } => {
                    unreachable!("parked storage should have been restored")
                }
            };
            frozen.reader_count = frozen
                .reader_count
                .checked_add(1)
                .expect("reader count should fit usize");
            ParkableBytesSnapshot {
                data: bytes,
                owner_id: self.inner.registration.id,
                lease: SnapshotLease {
                    owner: Arc::downgrade(&self.inner),
                },
            }
        };
        Ok(snapshot)
    }

    /// Memory currently retained by the encoded bytes.
    #[cfg(test)]
    fn resident_bytes(&self) -> usize {
        self.diagnostics().resident_bytes
    }

    fn manager(&self) -> &ParkableBytesManager {
        &self.inner.registration.manager
    }

    fn retain_reader(&self) {
        self.inner.state.lock().reader_count += 1;
    }

    fn release_reader(&self) {
        let delay = self.manager().inner.policy.reader_release_delay;
        let mut frozen = self.inner.state.lock();
        frozen.reader_count = frozen
            .reader_count
            .checked_sub(1)
            .expect("reader leases must remain balanced");
        if frozen.reader_count == 0 {
            frozen.park_after = Instant::now()
                .checked_add(delay)
                .expect("reader release delay should fit Instant");
        }
    }

    fn parking_deadline(&self) -> Option<Instant> {
        let frozen = self.inner.state.lock();
        if frozen.len < self.manager().inner.policy.min_size_to_park {
            return None;
        }
        let FrozenStorage::Resident { disk_backup, .. } = &frozen.storage else {
            return None;
        };
        if frozen.reader_count != 0 {
            return None;
        }
        if disk_backup.is_none()
            && !self
                .manager()
                .disk_pool()
                .is_some_and(|pool| pool.may_write())
        {
            return None;
        }
        Some(frozen.park_after)
    }

    /// Attempts to discard resident bytes. A successful first park writes
    /// exactly one disk extent.
    pub(crate) fn park(&self) -> io::Result<()> {
        self.park_impl(false)
    }

    /// Memory-pressure path: bypasses the deadline, never a live reader or
    /// another storage transition.
    #[cfg(test)]
    fn force_park(&self) -> io::Result<()> {
        self.park_impl(true)
    }

    fn park_impl(&self, ignore_deadline: bool) -> io::Result<()> {
        struct Prepared {
            pool: DiskPool,
            chunk: ReservedChunk,
            bytes: Arc<[u8]>,
        }

        let prepared = {
            let mut frozen = self.inner.state.lock();
            let policy = self.manager().inner.policy;
            if frozen.len < policy.min_size_to_park
                || !matches!(frozen.storage, FrozenStorage::Resident { .. })
                || frozen.reader_count != 0
            {
                return Ok(());
            }
            if !ignore_deadline && Instant::now() < frozen.park_after {
                return Ok(());
            }

            let (disk_backup, bytes) = match &frozen.storage {
                FrozenStorage::Resident {
                    bytes, disk_backup,
                } => (disk_backup.clone(), Arc::clone(bytes)),
                _ => unreachable!("non-resident storage returned before parking eligibility"),
            };
            if let Some(disk) = disk_backup {
                frozen.storage = FrozenStorage::Parked { disk };
                return Ok(());
            }

            let Some(pool) = self.manager().disk_pool() else {
                return Ok(());
            };
            let Some(chunk) = pool.try_reserve_chunk(frozen.len) else {
                return Ok(());
            };
            frozen.storage = FrozenStorage::Parking {
                bytes: Arc::clone(&bytes),
            };
            Prepared { pool, chunk, bytes }
        };

        let Prepared { pool, chunk, bytes } = prepared;
        let write_result = pool.write(chunk, &bytes);
        let mut frozen = self.inner.state.lock();
        if !matches!(frozen.storage, FrozenStorage::Parking { .. }) {
            return Err(io::Error::other(
                "parking write completed after its state was replaced",
            ));
        }
        match write_result {
            Ok(disk) if frozen.reader_count == 0 => {
                frozen.storage = FrozenStorage::Parked {
                    disk: Arc::new(disk),
                };
                Ok(())
            }
            Ok(disk) => {
                frozen.storage = FrozenStorage::Resident {
                    bytes,
                    disk_backup: Some(Arc::new(disk)),
                };
                Ok(())
            }
            Err(error) => {
                frozen.storage = FrozenStorage::Resident {
                    bytes,
                    disk_backup: None,
                };
                Err(error)
            }
        }
    }

    fn diagnostics(&self) -> ParkableDiagnostics {
        let frozen = self.inner.state.lock();
        let (parked, resident_bytes) = match &frozen.storage {
            FrozenStorage::Resident { bytes, .. } | FrozenStorage::Parking { bytes } => {
                (false, bytes.len())
            }
            FrozenStorage::Parked { .. } => (true, 0),
        };
        ParkableDiagnostics {
            len: frozen.len,
            parked,
            resident_bytes,
        }
    }

    fn downgrade(&self) -> Weak<ParkableBytesInner> {
        Arc::downgrade(&self.inner)
    }
}

impl fmt::Debug for ParkableBytes {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ParkableBytes")
            .field("id", &self.inner.registration.id)
            .field("diagnostics", &self.diagnostics())
            .finish_non_exhaustive()
    }
}

impl Deref for ParkableBytesSnapshot {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl AsRef<[u8]> for ParkableBytesSnapshot {
    fn as_ref(&self) -> &[u8] {
        self
    }
}

impl Clone for ParkableBytesSnapshot {
    fn clone(&self) -> Self {
        Self {
            data: Arc::clone(&self.data),
            owner_id: self.owner_id,
            lease: self.lease.clone(),
        }
    }
}

impl Clone for SnapshotLease {
    fn clone(&self) -> Self {
        if let Some(inner) = self.owner.upgrade() {
            ParkableBytes { inner }.retain_reader();
        }
        Self {
            owner: self.owner.clone(),
        }
    }
}

impl Drop for SnapshotLease {
    fn drop(&mut self) {
        if let Some(inner) = self.owner.upgrade() {
            ParkableBytes { inner }.release_reader();
        }
    }
}

impl fmt::Debug for ParkableBytesSnapshot {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ParkableBytesSnapshot")
            .field("len", &self.data.len())
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn manager() -> ParkableBytesManager {
        ParkableBytesManager::new(ParkablePolicy {
            min_size_to_park: 4,
            parking_delay: Duration::from_millis(1),
            reader_release_delay: Duration::from_millis(1),
        })
    }

    #[test]
    fn resident_snapshots_share_bytes() {
        let manager = manager();
        let image = manager.from_frozen_bytes(Arc::from(&b"image-bytes"[..]));
        let first = image.snapshot().unwrap();
        let second = image.snapshot().unwrap();
        assert!(Arc::ptr_eq(&first.data, &second.data));
        assert_eq!(&first[..], b"image-bytes");
    }

    #[test]
    fn force_park_discards_memory_and_read_restores() {
        let manager = manager();
        let image = manager.from_frozen_bytes(Arc::from(&b"roundtrip"[..]));
        image.force_park().unwrap();
        assert_eq!(image.resident_bytes(), 0);
        let snapshot = image.snapshot().unwrap();
        assert_eq!(&snapshot[..], b"roundtrip");
        assert_eq!(image.resident_bytes(), 9);
        drop(snapshot);
        // The disk backup makes the next park a memory-only transition.
        image.force_park().unwrap();
        assert_eq!(image.resident_bytes(), 0);
    }

    #[test]
    fn live_reader_defers_parking() {
        let manager = manager();
        let image = manager.from_frozen_bytes(Arc::from(&b"kept-alive"[..]));
        let snapshot = image.snapshot().unwrap();
        image.force_park().unwrap();
        assert_ne!(image.resident_bytes(), 0, "a live reader keeps bytes resident");
        drop(snapshot);
        image.force_park().unwrap();
        assert_eq!(image.resident_bytes(), 0);
    }

    #[test]
    fn small_values_never_park() {
        let manager = manager();
        let image = manager.from_frozen_bytes(Arc::from(&b"x"[..]));
        image.force_park().unwrap();
        assert_ne!(image.resident_bytes(), 0);
    }
}
