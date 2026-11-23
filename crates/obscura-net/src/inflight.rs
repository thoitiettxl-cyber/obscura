use std::collections::HashMap;
use std::future::Future;
use std::hash::Hash;
use std::sync::Mutex;

use tokio::sync::watch;

/// In-flight coalescing (port of camofox `lib/inflight.js`): concurrent
/// calls for the same key share one operation. The first caller is the
/// leader and runs it; callers that arrive while it runs wait for the
/// leader instead of starting a duplicate.
///
/// Followers do not receive the leader's return value — camofox shares
/// the promise itself, but obscura callers re-read shared state (a
/// cache) after `coalesce` returns, which is cheaper than broadcasting
/// a cloned response. If the leader's task dies before finishing, its
/// followers fail open: `wait_for` errors on the dropped sender and
/// they return `false` the same as a completed wait.
pub struct Inflight<K> {
    pending: Mutex<HashMap<K, watch::Sender<Option<()>>>>,
}

impl<K> Default for Inflight<K> {
    fn default() -> Self {
        Self {
            pending: Mutex::new(HashMap::new()),
        }
    }
}

impl<K: Eq + Hash + Clone> Inflight<K> {
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns true when this call ran `op` (leader). Concurrent callers
    /// with the same `key` wait for the leader and return false without
    /// running `op` (followers).
    pub async fn coalesce<F, Fut>(&self, key: &K, op: F) -> bool
    where
        F: FnOnce() -> Fut,
        Fut: Future<Output = ()>,
    {
        enum Acquisition {
            Leader(watch::Sender<Option<()>>),
            Follower(watch::Receiver<Option<()>>),
        }
        let acquisition = {
            let mut pending = self.pending.lock().unwrap();
            match pending.get(key) {
                Some(sender) => Acquisition::Follower(sender.subscribe()),
                None => {
                    let (sender, _receiver) = watch::channel(None);
                    pending.insert(key.clone(), sender.clone());
                    Acquisition::Leader(sender)
                }
            }
        };
        match acquisition {
            Acquisition::Leader(sender) => {
                op().await;
                self.pending.lock().unwrap().remove(key);
                let _ = sender.send(Some(()));
                true
            }
            Acquisition::Follower(mut receiver) => {
                let _ = receiver.wait_for(|outcome| outcome.is_some()).await;
                false
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    use std::time::Duration;

    #[tokio::test]
    async fn concurrent_calls_share_one_operation() {
        let inflight = Arc::new(Inflight::<String>::new());
        let runs = Arc::new(AtomicUsize::new(0));
        let mut tasks = Vec::new();
        for _ in 0..8 {
            let inflight = inflight.clone();
            let runs = runs.clone();
            tasks.push(tokio::spawn(async move {
                inflight
                    .coalesce(&"same".to_string(), || {
                        let runs = runs.clone();
                        async move {
                            tokio::time::sleep(Duration::from_millis(50)).await;
                            runs.fetch_add(1, Ordering::SeqCst);
                        }
                    })
                    .await
            }));
        }
        let mut leaders = 0;
        for task in tasks {
            if task.await.unwrap() {
                leaders += 1;
            }
        }
        assert_eq!(leaders, 1);
        assert_eq!(runs.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn sequential_calls_each_run() {
        let inflight = Inflight::<String>::new();
        let runs = AtomicUsize::new(0);
        for _ in 0..3 {
            assert!(
                inflight
                    .coalesce(&"same".to_string(), || async {
                        runs.fetch_add(1, Ordering::SeqCst);
                    })
                    .await
            );
        }
        assert_eq!(runs.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn distinct_keys_do_not_coalesce() {
        let inflight = Arc::new(Inflight::<String>::new());
        let runs = Arc::new(AtomicUsize::new(0));
        let mut tasks = Vec::new();
        for key in ["a", "b", "c"] {
            let inflight = inflight.clone();
            let runs = runs.clone();
            tasks.push(tokio::spawn(async move {
                inflight
                    .coalesce(&key.to_string(), || {
                        let runs = runs.clone();
                        async move {
                            runs.fetch_add(1, Ordering::SeqCst);
                        }
                    })
                    .await
            }));
        }
        let mut leaders = 0;
        for task in tasks {
            if task.await.unwrap() {
                leaders += 1;
            }
        }
        assert_eq!(leaders, 3);
        assert_eq!(runs.load(Ordering::SeqCst), 3);
    }
}
