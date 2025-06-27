use crate::{ffi::Matches, metric::MetricType, Index, IndexOptions, Key, VectorType};
use std::marker::PhantomData;

/// Higher level abstraction over Index that leverages the rust type system
/// # Parameters
///  - `T`: The scalar type of indexed vectors
///  - `D`: The length of the vector
///  - `M`: The metric used to index the vectors
///
/// # Examples
///
/// Basic usage:
///
/// ```rust
/// use usearch::{HighLevel, metric::Cos};
///
///
/// let index = HighLevel::<f32,4,Cos>::try_default().expect("Failed to create index.");
/// index.reserve(1000).expect("Failed to reserve capacity.");
///
/// // Add vectors to the index
/// let vector1: Vec<f32> = vec![0.0, 1.0, 0.0, 1.0];
/// let vector2: Vec<f32> = vec![1.0, 0.0, 1.0, 0.0];
/// index.add(1, &vector1).expect("Failed to add vector1.");
/// index.add(2, &vector2).expect("Failed to add vector2.");
///
/// // Search for the nearest neighbors to a query vector
/// let query: Vec<f32> = vec![0.5, 0.5, 0.5, 0.5];
/// let results = index.search(&query, 5).expect("Search failed.").result();
/// println!("Cos: {results:?}");
/// ```
/// For more examples, including how to add vectors to the index and perform searches,
/// refer to the individual method documentation.
pub struct HighLevel<T: VectorType, const D: usize, M: MetricType> {
    _t: PhantomData<T>,
    _m: PhantomData<M>,
    index: Index,
}
impl Matches {
    pub fn result(self) -> Vec<ResultElement> {
        let output: Vec<_> = self
            .keys
            .into_iter()
            .zip(self.distances)
            .map(|(key, distance)| ResultElement { distance, key })
            .collect();
        output
    }
}
impl<T: VectorType, const D: usize, M: MetricType> HighLevel<T, D, M> {
    fn make_index(options: &IndexOptions) -> Result<Self, cxx::Exception> {
        let mut index = Index::new(&options)?;
        if let Some(custom_metric) = M::custom_metric::<T>() {
            index.change_metric(custom_metric);
        }
        Ok(Self {
            _t: PhantomData,
            _m: PhantomData,
            index,
        })
    }
    /// Adds a vector to the index under the specified key.
    ///
    /// # Parameters
    /// - `connectivity`: Connectivity (default = 0)
    /// - `expansion_add`: Expansion when adding (default = 0)
    /// - `expansion_search`: Expansion when searching (default = 0)
    /// - `multi`: Allow multiple vectors per key (default = false)
    ///
    /// # Returns
    /// - `Ok(Self)` if the index was created successfully
    /// - `Err(cxx::Exception)` if an error occurred during the operation.
    pub fn new(
        connectivity: usize,
        expansion_add: usize,
        expansion_search: usize,
        multi: bool,
    ) -> Result<Self, cxx::Exception> {
        let options = IndexOptions {
            dimensions: D,
            metric: M::get_kind(),
            quantization: T::quant_type(),
            connectivity,
            expansion_add,
            expansion_search,
            multi,
        };
        Self::make_index(&options)
    }
    /// Reserves memory for a specified number of incoming vectors.
    ///
    /// # Arguments
    ///
    /// * `capacity` - The desired total capacity, including the current size.
    pub fn reserve(&self, n: usize) -> Result<(), cxx::Exception> {
        self.index.reserve(n)
    }
    /// Adds a vector to the index under the specified key.
    ///
    /// # Parameters
    /// - `key`: The key under which the vector should be stored.
    /// - `vector`: A slice representing the vector to be added.
    ///
    /// # Returns
    /// - `Ok(())` if the vector was successfully added to the index.
    /// - `Err(cxx::Exception)` if an error occurred during the operation.
    pub fn add(&self, key: Key, vector: &[T]) -> Result<(), cxx::Exception> {
        self.index.add(key, vector)
    }
    /// Retrieves a vector from the index by its key.
    ///
    /// # Parameters
    /// - `key`: The key of the vector to retrieve.
    /// - `buffer`: A mutable slice where the retrieved vector will be stored. The size of the
    ///   buffer determines the maximum number of elements that can be retrieved.
    ///
    /// # Returns
    /// - `Ok(usize)` indicating the number of elements actually written into the `buffer`.
    /// - `Err(cxx::Exception)` if an error occurred during the operation.
    pub fn get(&self, key: Key, vector: &mut [T]) -> Result<usize, cxx::Exception> {
        self.index.get(key, vector)
    }
    /// Performs a search in the index using the given query vector, returning
    /// up to `count` closest matches.
    ///
    /// # Parameters
    /// - `query`: A slice representing the query vector.
    /// - `count`: The maximum number of matches to return.
    ///
    /// # Returns
    /// - `Ok(ffi::Matches)` containing the matches found.
    /// - `Err(cxx::Exception)` if an error occurred during the search operation.
    pub fn search(&self, query: &[T], count: usize) -> Result<Matches, cxx::Exception> {
        self.index.search(query, count)
    }
    /// Performs a filtered search in the index using a query vector and a custom
    /// filter function, returning up to `count` matches that satisfy the filter.
    ///
    /// # Parameters
    /// - `query`: A slice representing the query vector.
    /// - `count`: The maximum number of matches to return.
    /// - `filter`: A closure that takes a `Key` and returns `true` if the corresponding
    ///   vector should be included in the search results, or `false` otherwise.
    ///
    /// # Returns
    /// - `Ok(ffi::Matches)` containing the matches that satisfy the filter.
    /// - `Err(cxx::Exception)` if an error occurred during the filtered search operation.
    pub fn filtered_search<F>(
        &self,
        query: &[T],
        count: usize,
        filter: F,
    ) -> Result<Matches, cxx::Exception>
    where
        F: Fn(Key) -> bool,
    {
        self.index.filtered_search(query, count, filter)
    }
    ///Create the index with default values
    pub fn try_default() -> Result<Self, cxx::Exception> {
        let mut options = IndexOptions::default();
        options.dimensions = D;
        options.metric = M::get_kind();
        options.quantization = T::quant_type();
        Self::make_index(&options)
    }
    /// Changes the metric used for distance calculations within the index.
    ///
    /// # Returns
    /// - `Ok(HighLevel<T, D, Nm>)` if the metric was successfully changed.
    /// - `Err(cxx::Exception)` if an error occurred during the operation.
    pub fn change_metric<Nm: MetricType>(self) -> HighLevel<T, D, Nm> {
        let mut index = self.index;
        index.change_metric_kind(Nm::get_kind());
        if let Some(custom_metric) = M::custom_metric::<T>() {
            index.change_metric(custom_metric);
        }
        HighLevel {
            _t: PhantomData,
            _m: PhantomData,
            index,
        }
    }
    /// Retrieves the expansion value used during index creation.
    pub fn expansion_add(&self) -> usize {
        self.index.expansion_add()
    }

    /// Retrieves the expansion value used during search.
    pub fn expansion_search(&self) -> usize {
        self.index.expansion_search()
    }

    /// Updates the expansion value used during index creation. Rarely used.
    pub fn change_expansion_add(&self, n: usize) {
        self.index.change_expansion_add(n)
    }

    /// Updates the expansion value used during search operations.
    pub fn change_expansion_search(&self, n: usize) {
        self.index.change_expansion_search(n)
    }

    /// Retrieves the number of dimensions in the vectors indexed.
    pub fn dimensions(&self) -> usize {
        self.index.dimensions()
    }

    /// Retrieves the connectivity parameter that limits connections-per-node in the graph.
    pub fn connectivity(&self) -> usize {
        self.index.connectivity()
    }

    /// Retrieves the current number of vectors in the index.
    pub fn size(&self) -> usize {
        self.index.size()
    }

    /// Retrieves the total capacity of the index, including reserved space.
    pub fn capacity(&self) -> usize {
        self.index.capacity()
    }

    /// Reports expected file size after serialization.
    pub fn serialized_length(&self) -> usize {
        self.index.serialized_length()
    }

    /// Removes the vector associated with the given key from the index.
    ///
    /// # Arguments
    ///
    /// * `key` - The key of the vector to be removed.
    ///
    /// # Returns
    ///
    /// `true` if the vector is successfully removed, `false` otherwise.
    pub fn remove(&self, key: Key) -> Result<usize, cxx::Exception> {
        self.index.remove(key)
    }

    /// Renames the vector under a specific key.
    ///
    /// # Arguments
    ///
    /// * `from` - The key of the vector to be renamed.
    /// * `to` - The new name.
    ///
    /// # Returns
    ///
    /// `true` if the vector is renamed, `false` otherwise.
    pub fn rename(&self, from: Key, to: Key) -> Result<usize, cxx::Exception> {
        self.index.rename(from, to)
    }

    /// Checks if the index contains a vector with a specified key.
    ///
    /// # Arguments
    ///
    /// * `key` - The key to be checked.
    ///
    /// # Returns
    ///
    /// `true` if the index contains the vector with the given key, `false` otherwise.
    pub fn contains(&self, key: Key) -> bool {
        self.index.contains(key)
    }

    /// Count the count of vectors with the same specified key.
    ///
    /// # Arguments
    ///
    /// * `key` - The key to be checked.
    ///
    /// # Returns
    ///
    /// Number of vectors found.
    pub fn count(&self, key: Key) -> usize {
        self.index.count(key)
    }

    /// Saves the index to a specified file.
    ///
    /// # Arguments
    ///
    /// * `path` - The file path where the index will be saved.
    pub fn save(&self, path: &str) -> Result<(), cxx::Exception> {
        self.index.save(path)
    }

    /// Loads the index from a specified file.
    ///
    /// # Arguments
    ///
    /// * `path` - The file path from where the index will be loaded.
    pub fn load(&self, path: &str) -> Result<(), cxx::Exception> {
        self.index.load(path)
    }

    /// Creates a view of the index from a file without loading it into memory.
    ///
    /// # Arguments
    ///
    /// * `path` - The file path from where the view will be created.
    pub fn view(&self, path: &str) -> Result<(), cxx::Exception> {
        self.index.view(path)
    }

    /// Erases all members from the index, closes files, and returns RAM to OS.
    pub fn reset(&self) -> Result<(), cxx::Exception> {
        self.index.reset()
    }

    /// A relatively accurate lower bound on the amount of memory consumed by the system.
    /// In practice, its error will be below 10%.
    pub fn memory_usage(&self) -> usize {
        self.index.memory_usage()
    }

    /// Saves the index to a specified file.
    ///
    /// # Arguments
    ///
    /// * `buffer` - The buffer where the index will be saved.
    pub fn save_to_buffer(&self, buffer: &mut [u8]) -> Result<(), cxx::Exception> {
        self.index.save_to_buffer(buffer)
    }

    /// Loads the index from a specified file.
    ///
    /// # Arguments
    ///
    /// * `buffer` - The buffer from where the index will be loaded.
    pub fn load_from_buffer(&self, buffer: &[u8]) -> Result<(), cxx::Exception> {
        self.index.load_from_buffer(buffer)
    }

    /// Creates a view of the index from a file without loading it into memory.
    ///
    /// # Arguments
    ///
    /// * `buffer` - The buffer from where the view will be created.
    ///
    /// # Safety
    ///
    /// This function is marked as `unsafe` because it stores a pointer to the input buffer.
    /// The caller must ensure that the buffer outlives the index and is not dropped
    /// or modified for the duration of the index's use. Dereferencing a pointer to a
    /// temporary buffer after it has been dropped can lead to undefined behavior,
    /// which violates Rust's memory safety guarantees.
    ///
    /// Example of misuse:
    ///
    /// ```rust,ignore
    /// let index: usearch::Index = usearch::new_index(&usearch::IndexOptions::default()).unwrap();
    ///
    /// let temporary = vec![0u8; 100];
    /// index.view_from_buffer(&temporary);
    /// std::mem::drop(temporary);
    ///
    /// let query = vec![0.0; 256];
    /// let results = index.search(&query, 5).unwrap();
    /// ```
    ///
    /// The above example would result in use-after-free and undefined behavior.
    pub unsafe fn view_from_buffer(&self, buffer: &[u8]) -> Result<(), cxx::Exception> {
        self.index.view_from_buffer(buffer)
    }
}

pub struct Search<T: VectorType, const D: usize> {
    _type_marker: PhantomData<T>,
    index: Index,
}
#[derive(Debug, Clone, Copy)]
pub struct ResultElement {
    pub distance: f32,
    pub key: u64,
}
