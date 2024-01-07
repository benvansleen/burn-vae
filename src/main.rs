use burn::{
    backend::{
        wgpu::{AutoGraphicsApi, WgpuDevice},
        Autodiff, Wgpu,
    },
    config::Config,
    module::Module,
    nn::LinearConfig,
    optim::AdamWConfig,
    record::{CompactRecorder, Recorder},
};
use burn_vae::{
    data::{get_data, INPUT_DIM},
    mlp::MLPBlockConfig,
    model::{DecoderConfig, EncoderConfig, VAEConfig},
    train::{train, TrainingConfig},
    visualization::plot,
};

type Backend = Wgpu<AutoGraphicsApi, f32, i32>;
const ARTIFACTS_DIR: &str = "artifacts/";

fn main() {
    let device = WgpuDevice::BestAvailable;

    const LATENT_DIM: usize = 2;
    train::<Autodiff<Backend>>(
        ARTIFACTS_DIR,
        TrainingConfig::new(
            VAEConfig::new(
                EncoderConfig::new(
                    MLPBlockConfig::new(3, 512, INPUT_DIM, 64)
                        .with_dropout(0.1),
                    LinearConfig::new(64, LATENT_DIM)
                        .with_bias(false),
                    LinearConfig::new(64, LATENT_DIM)
                        .with_bias(false),
                ),
                DecoderConfig::new(
                    MLPBlockConfig::new(4, 512, LATENT_DIM, 64)
                        .with_dropout(0.1),
                    LinearConfig::new(64, INPUT_DIM)
                        .with_bias(false),
                ),
            )
            .with_kl_weight(1e-1)
            .with_latent_dim(LATENT_DIM),
            AdamWConfig::new(),
        )
        .with_num_epochs(1000)
        .with_batch_size(256)
        .with_num_workers(4)
        .with_warmup_steps(500)
        .with_early_stop_patience(15)
        .with_learning_rate(1e1),
        &device,
    );

    let config = TrainingConfig::load(format!(
        "{ARTIFACTS_DIR}/config.json"
    ))
    .expect("Config should exist for model");
    let record = CompactRecorder::new()
        .load(format!("{ARTIFACTS_DIR}/model").into())
        .expect("Model should exist for model");
    let model = config
        .model
        .init_with::<Backend>(record)
        .to_device(&device);

    const N: usize = 5000;
    const MAX_SIZE: usize = 128;
    let mut generated = Vec::new();
    (0..N).step_by(MAX_SIZE).for_each(|_| {
        generated.extend(model.generate(MAX_SIZE, &device))
    });
    let data = get_data(N);
    let (pts, colors) = (data, [100.; N]);
    plot((&pts, &generated), (&colors, &[0.; N]));
}
