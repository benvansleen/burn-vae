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
    visualization::{plot, Trace},
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
                    MLPBlockConfig::new(3, 128, INPUT_DIM, 32)
                        .with_dropout(0.1),
                    LinearConfig::new(32, LATENT_DIM)
                        .with_bias(true),
                    LinearConfig::new(32, LATENT_DIM)
                        .with_bias(true),
                ),
                DecoderConfig::new(
                    MLPBlockConfig::new(6, 128, LATENT_DIM, 64)
                        .with_dropout(0.1),
                    LinearConfig::new(64, INPUT_DIM)
                        .with_bias(true),
                ),
            )
            .with_kl_weight(1e0)
            .with_latent_dim(LATENT_DIM),
            AdamWConfig::new(),
        )
        .with_num_epochs(1000)
        .with_batch_size(256)
        .with_num_workers(4)
        .with_warmup_steps(500)
        .with_early_stop_patience(15)
        .with_learning_rate(4e1),
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
    let mut gen_colors = Vec::new();
    (0..N).step_by(MAX_SIZE).for_each(|_| {
        let (gen, t) = model.generate(MAX_SIZE, &device);
        generated.extend(gen);
        gen_colors.extend(t);
    });

    let true_data = get_data(N);
    let mut true_colors = Vec::new();
    let true_pts = true_data
        .into_iter()
        .map(|mut v| {
            true_colors.push(v.pop().unwrap());
            v
        })
        .collect();

    plot(&[
        Trace::new(generated, gen_colors, "generated"),
        Trace::new(true_pts, true_colors, "true"),
    ]);
}
