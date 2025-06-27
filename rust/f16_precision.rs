use half::bf16;
//pub use half::bf16;
/// A struct representing a half-precision floating-point number based on the IEEE 754 standard.
///
/// This struct uses an `i16` to store the half-precision floating-point data, which includes
/// 1 sign bit, 5 exponent bits, and 10 mantissa bits.
use half::f16;

use crate::ffi;
use crate::Distance;
use crate::Index;
use crate::Key;
use crate::MetricFunction;
use crate::ScalarKind;
use crate::VectorType;
/*#[repr(transparent)]
#[allow(non_camel_case_types)]
#[derive(Clone, Copy)]
pub struct f16(i16);*/
impl Bf16HalfUSearchExt for bf16 {}
#[allow(dead_code)]
trait Bf16HalfUSearchExt {
    /// Casts a slice of `i16` integers to a slice of `f16`, allowing operations on half-precision
    /// floating-point data stored in standard 16-bit integer arrays.
    fn from_i16s(slice: &[i16]) -> &[half::bf16] {
        bytemuck::cast_slice(slice)
        //unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const Self, slice.len()) }
    }
    /// Casts a mutable slice of `i16` integers to a mutable slice of `f16`, enabling mutable operations
    /// on half-precision floating-point data.
    fn from_mut_i16s(slice: &mut [i16]) -> &mut [half::bf16] {
        bytemuck::cast_slice_mut(slice)
    }

    /// Converts a slice of `f16` back to a slice of `i16`, useful for storage or manipulation in formats
    /// that require standard integer types.
    fn to_i16s(slice: &[bf16]) -> &[i16] {
        bytemuck::cast_slice(slice)
    }

    /// Converts a mutable slice of `f16` back to a mutable slice of `i16`, enabling further
    /// modifications on the original integer data after operations involving half-precision
    /// floating-point numbers.
    fn to_mut_i16s(slice: &mut [bf16]) -> &mut [i16] {
        bytemuck::cast_slice_mut(slice)
    }
}

impl F16HalfUSearchExt for f16 {}

pub trait F16HalfUSearchExt {
    /// Casts a slice of `i16` integers to a slice of `f16`, allowing operations on half-precision
    /// floating-point data stored in standard 16-bit integer arrays.
    fn from_i16s(slice: &[i16]) -> &[half::f16] {
        bytemuck::cast_slice(slice)
        //unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const Self, slice.len()) }
    }
    /// Casts a mutable slice of `i16` integers to a mutable slice of `f16`, enabling mutable operations
    /// on half-precision floating-point data.
    fn from_mut_i16s(slice: &mut [i16]) -> &mut [half::f16] {
        bytemuck::cast_slice_mut(slice)
    }

    /// Converts a slice of `f16` back to a slice of `i16`, useful for storage or manipulation in formats
    /// that require standard integer types.
    fn to_i16s(slice: &[f16]) -> &[i16] {
        bytemuck::cast_slice(slice)
    }

    /// Converts a mutable slice of `f16` back to a mutable slice of `i16`, enabling further
    /// modifications on the original integer data after operations involving half-precision
    /// floating-point numbers.
    fn to_mut_i16s(slice: &mut [f16]) -> &mut [i16] {
        bytemuck::cast_slice_mut(slice)
    }
}

impl VectorType for bf16 {
    fn search(index: &Index, query: &[Self], count: usize) -> Result<ffi::Matches, cxx::Exception> {
        index.inner.search_f16(bf16::to_i16s(query), count)
    }
    fn get(index: &Index, key: Key, vector: &mut [Self]) -> Result<usize, cxx::Exception> {
        println!("Not implemented for BF16 yet");
        index.inner.get_f16(key, bf16::to_mut_i16s(vector))
    }
    fn add(index: &Index, key: Key, vector: &[Self]) -> Result<(), cxx::Exception> {
        index.inner.add_f16(key, bf16::to_i16s(vector))
    }
    fn filtered_search<F>(
        index: &Index,
        query: &[Self],
        count: usize,
        filter: F,
    ) -> Result<ffi::Matches, cxx::Exception>
    where
        Self: Sized,
        F: Fn(Key) -> bool,
    {
        // Trampoline is the function that knows how to call the Rust closure.
        extern "C" fn trampoline<F: Fn(u64) -> bool>(key: u64, closure_address: usize) -> bool {
            let closure = closure_address as *const F;
            unsafe { (*closure)(key) }
        }

        // Temporarily cast the closure to a raw pointer for passing.
        unsafe {
            let trampoline_fn: usize = std::mem::transmute(trampoline::<F> as *const ());
            let closure_address: usize = &filter as *const F as usize;
            index.inner.filtered_search_f16(
                bf16::to_i16s(query),
                count,
                trampoline_fn,
                closure_address,
            )
        }
    }

    fn change_metric(
        index: &mut Index,
        metric: std::boxed::Box<dyn Fn(*const Self, *const Self) -> Distance + Send + Sync>,
    ) -> Result<(), cxx::Exception> {
        // Store the metric function in the Index.
        type MetricFn = fn(*const bf16, *const bf16) -> Distance;
        index.metric_fn = Some(MetricFunction::BF16Metric(metric));

        // Trampoline is the function that knows how to call the Rust closure.
        // The `first` is a pointer to the first vector, `second` is a pointer to the second vector,
        // and `index_wrapper` is a pointer to the `index` itself, from which we can infer the metric function
        // and the number of dimensions.
        extern "C" fn trampoline(first: usize, second: usize, closure_address: usize) -> Distance {
            let first_ptr = first as *const bf16;
            let second_ptr = second as *const bf16;
            let closure: MetricFn = unsafe { std::mem::transmute(closure_address) };
            closure(first_ptr, second_ptr)
        }

        unsafe {
            let trampoline_fn: usize = std::mem::transmute(trampoline as *const ());
            let closure_address = match index.metric_fn {
                Some(MetricFunction::BF16Metric(ref metric)) => metric as *const _ as usize,
                _ => panic!("Expected BF16Metric"),
            };
            index.inner.change_metric(trampoline_fn, closure_address)
        }

        Ok(())
    }

    fn quant_type() -> ScalarKind {
        ScalarKind::BF16
    }
}
impl VectorType for f16 {
    fn search(index: &Index, query: &[Self], count: usize) -> Result<ffi::Matches, cxx::Exception> {
        index.inner.search_f16(f16::to_i16s(query), count)
    }
    fn get(index: &Index, key: Key, vector: &mut [Self]) -> Result<usize, cxx::Exception> {
        index.inner.get_f16(key, f16::to_mut_i16s(vector))
    }
    fn add(index: &Index, key: Key, vector: &[Self]) -> Result<(), cxx::Exception> {
        index.inner.add_f16(key, f16::to_i16s(vector))
    }
    fn filtered_search<F>(
        index: &Index,
        query: &[Self],
        count: usize,
        filter: F,
    ) -> Result<ffi::Matches, cxx::Exception>
    where
        Self: Sized,
        F: Fn(Key) -> bool,
    {
        // Trampoline is the function that knows how to call the Rust closure.
        extern "C" fn trampoline<F: Fn(u64) -> bool>(key: u64, closure_address: usize) -> bool {
            let closure = closure_address as *const F;
            unsafe { (*closure)(key) }
        }

        // Temporarily cast the closure to a raw pointer for passing.
        unsafe {
            let trampoline_fn: usize = std::mem::transmute(trampoline::<F> as *const ());
            let closure_address: usize = &filter as *const F as usize;
            index.inner.filtered_search_f16(
                f16::to_i16s(query),
                count,
                trampoline_fn,
                closure_address,
            )
        }
    }

    fn change_metric(
        index: &mut Index,
        metric: std::boxed::Box<dyn Fn(*const Self, *const Self) -> Distance + Send + Sync>,
    ) -> Result<(), cxx::Exception> {
        // Store the metric function in the Index.
        type MetricFn = fn(*const f16, *const f16) -> Distance;
        index.metric_fn = Some(MetricFunction::F16Metric(metric));

        // Trampoline is the function that knows how to call the Rust closure.
        // The `first` is a pointer to the first vector, `second` is a pointer to the second vector,
        // and `index_wrapper` is a pointer to the `index` itself, from which we can infer the metric function
        // and the number of dimensions.
        extern "C" fn trampoline(first: usize, second: usize, closure_address: usize) -> Distance {
            let first_ptr = first as *const f16;
            let second_ptr = second as *const f16;
            let closure: MetricFn = unsafe { std::mem::transmute(closure_address) };
            closure(first_ptr, second_ptr)
        }

        unsafe {
            let trampoline_fn: usize = std::mem::transmute(trampoline as *const ());
            let closure_address = match index.metric_fn {
                Some(MetricFunction::F16Metric(ref metric)) => metric as *const _ as usize,
                _ => panic!("Expected F16Metric"),
            };
            index.inner.change_metric(trampoline_fn, closure_address)
        }

        Ok(())
    }

    fn quant_type() -> ScalarKind {
        ScalarKind::F16
    }
}
