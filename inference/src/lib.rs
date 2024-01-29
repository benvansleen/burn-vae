pub use train::load_model;
pub use train::TrainingConfig as ModelConfig;

pub use dataset::Point;
use once_cell::sync::OnceCell;
pub use vae::Model;

#[cfg(not(target_family = "wasm"))]
use burn::backend::{
    wgpu::{AutoGraphicsApi, WgpuDevice},
    Fusion, Wgpu,
};

#[cfg(not(target_family = "wasm"))]
type Backend = Fusion<Wgpu<AutoGraphicsApi, f32, i32>>;
#[cfg(not(target_family = "wasm"))]
const DEVICE: WgpuDevice = WgpuDevice::BestAvailable;
#[cfg(not(target_family = "wasm"))]
static MODEL: OnceCell<Model<Backend>> = OnceCell::new();

#[cfg(not(target_family = "wasm"))]
pub fn init(dir: &str) {
    MODEL.get_or_init(|| train::load_model::<Backend>(dir, &DEVICE));
}

#[cfg(not(target_family = "wasm"))]
pub fn generate(t: f32, n: usize) -> Vec<Point> {
    MODEL
        .get()
        .expect("Call .init() to load model")
        .generate(t, n, &DEVICE)
}

#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
#[pymodule]
fn _burn_vae(_py: Python, m: &PyModule) -> PyResult<()> {
    #[pyfn(m)]
    fn _init(dir: &str) {
        init(dir);
    }

    #[pyfn(m)]
    fn _generate(t: f32, n: usize) -> Vec<Point> {
        generate(t, n)
    }

    #[pyfn(m)]
    fn encode(x: Vec<Point>) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
        MODEL.get().expect("Call .init() to load model").encode(x)
    }

    Ok(())
}
