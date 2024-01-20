use burn::backend::{
    wgpu::{AutoGraphicsApi, WgpuDevice},
    Wgpu,
};
use dataset::Point;
use once_cell::sync::OnceCell;
use pyo3::prelude::*;
use vae::Model;

#[pymodule]
fn _burn_vae(_py: Python, m: &PyModule) -> PyResult<()> {
    type Backend = Wgpu<AutoGraphicsApi, f32, i32>;
    static MODEL: OnceCell<Model<Backend>> = OnceCell::new();
    const DEVICE: WgpuDevice = WgpuDevice::BestAvailable;

    #[pyfn(m)]
    fn init(dir: &str) {
        MODEL.get_or_init(|| {
            train::load_model::<Backend>(dir, &DEVICE)
        });
    }

    #[pyfn(m)]
    fn generate(t: f32, n: usize) -> Vec<Point> {
        MODEL
            .get()
            .expect("Call .init() to load model")
            .generate(t, n, &DEVICE)
    }

    #[pyfn(m)]
    fn encode(x: Vec<Point>) -> (Vec<Point>, Vec<Point>) {
        MODEL
            .get()
            .expect("Call .init() to load model")
            .encode(x)
    }

    Ok(())
}
