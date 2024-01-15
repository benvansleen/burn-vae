pub mod data;
pub mod loss;
pub mod metric;
pub mod mlp;
pub mod model;
pub mod train;
pub mod visualization;

use pyo3::prelude::*;

#[pymodule]
fn _burn_vae(_py: Python, m: &PyModule) -> PyResult<()> {
    use burn::{
        backend::{
            wgpu::{AutoGraphicsApi, WgpuDevice},
            Wgpu,
        },
        config::Config,
        module::Module,
        record::Recorder,
    };
    use model::VAE;
    use once_cell::sync::OnceCell;

    type Backend = Wgpu<AutoGraphicsApi, f32, i32>;
    static MODEL: OnceCell<VAE<Backend>> = OnceCell::new();
    const DEVICE: WgpuDevice = WgpuDevice::BestAvailable;

    #[pyfn(m)]
    fn init(dir: &str) {
        MODEL.get_or_init(|| {
            let config = train::TrainingConfig::load(format!(
                "{dir}/config.json"
            ))
            .expect("Config file not found");
            let record = burn::record::CompactRecorder::new()
                .load(format!("{dir}/model").into())
                .expect("Model not found");

            config
                .model
                .init_with::<Backend>(record)
                .to_device(&DEVICE)
        });
    }

    #[pyfn(m)]
    fn generate(t: f32, n: usize) -> Vec<Vec<f32>> {
        MODEL
            .get()
            .expect("Call .init() to load model")
            .generate(t, n, &DEVICE)
    }

    Ok(())
}
