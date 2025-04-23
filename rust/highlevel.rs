use std::marker::PhantomData;

use crate::{ffi::Matches, Index, IndexOptions, MetricKind, VectorType};

pub struct Search<T: VectorType, const D: usize> {
    _type_marker: PhantomData<T>,
    index: Index,
}

pub struct ResultElement {
    pub distance: f32,
    pub key: u64,
}

pub struct SearchBuilder<T: VectorType> {
    _type_marker: PhantomData<T>,
    options: IndexOptions,
}
impl<T: VectorType> SearchBuilder<T> {
    pub fn new(dimensions: usize, metric: MetricKind) -> Self {
        let mut options = IndexOptions::default();

        options.quantization = T::quant_type();
        options.dimensions = dimensions;
        options.metric = metric;
        Self {
            options,
            _type_marker: PhantomData,
        }
    }
    pub fn connectivity(self, conn: usize) -> Self {
        let mut options = self.options;
        options.connectivity = conn;
        Self {
            options,
            _type_marker: PhantomData,
        }
    }
    pub fn expansion_add(self, expansion_add: usize) -> Self {
        let mut options = self.options;
        options.expansion_add = expansion_add;
        Self {
            options,
            _type_marker: PhantomData,
        }
    }

    pub fn multi(self, multi: bool) -> Self {
        let mut options = self.options;
        options.multi = multi;
        Self {
            options,
            _type_marker: PhantomData,
        }
    }
    pub fn expansion_search(self, expansion_search: usize) -> Self {
        let mut options = self.options;
        options.expansion_search = expansion_search;
        Self {
            options,
            _type_marker: PhantomData,
        }
    }
}

impl<T: VectorType, const D: usize> Search<T, D> {
    pub fn search(
        &self,
        vector: [T; D],
        count: usize,
    ) -> Result<Vec<ResultElement>, cxx::Exception> {
        let Matches { keys, distances } = self.index.search(&vector, count)?;
        let output: Vec<_> = keys
            .into_iter()
            .zip(distances)
            .map(|(key, distance)| ResultElement { distance, key })
            .collect();
        Ok(output)
    }
    pub fn insert(&self, vector: [T; D], key: u64) -> Result<(), cxx::Exception> {
        self.index.add(key, &vector)
    }
    pub fn new(options: IndexOptions) -> Result<Self, cxx::Exception> {
        let mut options = options;
        options.quantization = T::quant_type();
        options.dimensions = D;
        Ok(Self {
            _type_marker: PhantomData,
            index: Index::new(&options)?,
        })
    }
    pub fn insert_batch(&self, batch: Vec<(u64, Vec<T>)>) -> Result<(), cxx::Exception> {
        todo!()
    }
}
