pub use train::load_model;
pub use train::TrainingConfig as ModelConfig;
pub use dataset::Point;
use once_cell::sync::OnceCell;
use vae::Model as M;
use burn::{
    tensor::Device,
    record::{BinBytesRecorder, FullPrecisionSettings, Recorder},
};

#[cfg(not(target_family = "wasm"))]
use burn::backend::{
    wgpu::{AutoGraphicsApi, WgpuDevice},
    Fusion, Wgpu,
};

#[cfg(not(target_family = "wasm"))]
type Backend = Fusion<Wgpu<AutoGraphicsApi, f32, i32>>;
#[cfg(target_family = "wasm")]
type Backend = NdArray<f32>;

pub type Model = M<Backend>;

static DEVICE: OnceCell<Device<Backend>> = OnceCell::new();
static MODEL: OnceCell<Model> = OnceCell::new();

#[cfg(target_family = "wasm")]
use burn::backend::{ndarray::NdArrayDevice, NdArray};

pub fn init(dir: &str) {
    #[cfg(not(target_family = "wasm"))]
    let device = WgpuDevice::BestAvailable;
    #[cfg(target_family = "wasm")]
    let device = NdArrayDevice::default();

    let device = DEVICE.get_or_init(|| device);
    MODEL
        .set(train::load_model::<Backend>(dir, device))
        .expect("Failed to initialize model");
}

pub fn load_bytes(config: ModelConfig, weights: Vec<u8>) {
    #[cfg(not(target_family = "wasm"))]
    DEVICE.get_or_init(|| WgpuDevice::BestAvailable);
    #[cfg(target_family = "wasm")]
    DEVICE.get_or_init(NdArrayDevice::default);

    let record = BinBytesRecorder::<FullPrecisionSettings>::default()
        .load(weights)
        .expect("Failed to load weights");
    MODEL.get_or_init(|| config.model.init_with::<Backend>(record));
}

#[cfg(not(target_family = "wasm"))]
pub fn generate(t: f32, n: usize) -> Vec<Point> {
    let device = DEVICE.get().expect("Call .init() to load model");
    MODEL
        .get()
        .expect("Call .init() to load model")
        .generate(t, n, device)
}

#[cfg(target_family = "wasm")]
pub async fn generate(t: f32, n: usize) -> Vec<Point> {
    let device = DEVICE.get().expect("Call .init() to load model");
    MODEL
        .get()
        .expect("Call .init() to load model")
        .generate(t, n, device)
        .await
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
