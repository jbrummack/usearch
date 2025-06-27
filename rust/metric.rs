
use crate::{Distance, MetricKind, VectorType};

pub trait MetricType {
    fn get_kind() -> MetricKind {
        MetricKind::Unknown
    }
    fn custom_metric<T: VectorType>(
    ) -> Option<Box<dyn Fn(*const T, *const T) -> Distance + Send + Sync>> {
        None
    }
}
macro_rules! define_metric_type {
    ($name:ident) => {
        pub struct $name;

        impl MetricType for $name {
            fn get_kind() -> MetricKind {
                MetricKind::$name
            }
        }
    };
}
define_metric_type!(IP);
define_metric_type!(L2sq);
define_metric_type!(Cos);
define_metric_type!(Pearson);
define_metric_type!(Haversine);
define_metric_type!(Divergence);
define_metric_type!(Hamming);
define_metric_type!(Tanimoto);
define_metric_type!(Sorensen);

//pub struct CustomMetric<T>;}
