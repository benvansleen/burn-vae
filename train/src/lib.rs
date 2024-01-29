pub mod visualization;
#[cfg(not(target_family = "wasm"))]
pub mod metric;

mod train;
pub use train::TrainingConfig;

#[cfg(not(target_family = "wasm"))]
pub use train::train;

mod load;
pub use load::load_model;
