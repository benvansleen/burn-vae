use burn::{
    config::Config,
    module::Module,
    record::Recorder,
    tensor::{backend::Backend, Device},
};
use vae::Model;

pub fn load_model<B: Backend>(
    dir: &str,
    device: &Device<B>,
) -> Model<B> {
    let config = crate::TrainingConfig::load(format!(
        "{dir}/config.json"
    ))
    .expect("Config file not found");
    let record = burn::record::CompactRecorder::new()
        .load(format!("{dir}/model").into())
        .expect("Model not found");

    config.model.init_with::<B>(record).to_device(device)
}
