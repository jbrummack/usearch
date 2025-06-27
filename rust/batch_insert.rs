use crate::{metric::MetricType, HighLevel, Key, VectorType};
use rayon::prelude::*;
impl<T: VectorType + Sync, const D: usize, M: MetricType + Sync> HighLevel<T, D, M> {
    /// Adds a batch of vectors with multithreading
    /// Faster when inserting large amounts of vectors;
    /// Slower when inserting smaller batches because spinning up the thread pool adds latency.
    /// # Parameters
    /// - `batch`

    ///
    /// # Returns
    /// - `Ok(())` if the batch was inserted successfully
    /// - `Err(cxx::Exception)` if an error occurred during the operation.
    pub fn batch_insert(&self, batch: &[(Key, &[T])]) -> Result<(), cxx::Exception> {
        let len = batch.len();
        self.reserve(len)?;
        batch
            .par_iter()
            .try_for_each(|(key, value)| self.index.add(*key, &value))?;
        Ok(())
    }
}
