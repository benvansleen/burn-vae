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
    data::{get_data, INPUT_DIM, LABEL_DIM},
    mlp::MLPBlockConfig,
    model::{DecoderConfig, EncoderConfig, VAEConfig},
    train::{train, TrainingConfig},
    visualization::{plot, Trace},
};
use rand::Rng;

type Backend = Wgpu<AutoGraphicsApi, f32, i32>;

fn main() {
    let device = WgpuDevice::BestAvailable;
    let args: Vec<String> = std::env::args().collect();
    let artifacts_dir = match args.last() {
        Some(dir) => dir,
        None => "artifacts",
    };

    const LATENT_DIM: usize = 2;
    train::<Autodiff<Backend>>(
        artifacts_dir,
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
                    MLPBlockConfig::new(
                        6,
                        128,
                        LATENT_DIM + LABEL_DIM,
                        64,
                    )
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
        .with_batch_size(512)
        .with_num_workers(4)
        .with_warmup_steps(500)
        .with_early_stop_patience(15)
        .with_learning_rate(4e1),
        &device,
    );

    let config = TrainingConfig::load(format!(
        "{artifacts_dir}/config.json"
    ))
    .expect("Config should exist for model");
    let record = CompactRecorder::new()
        .load(format!("{artifacts_dir}/model").into())
        .expect("Model should exist for model");
    let model = config
        .model
        .init_with::<Backend>(record)
        .to_device(&device);

    const N: usize = 5000;
    const MAX_SIZE: usize = 128;

    let (true_pts, true_colors) = get_data(N);

    let mut rng = rand::thread_rng();
    let max_t = true_colors
        .iter()
        .fold(f32::NEG_INFINITY, |acc, &i| f32::max(acc, i));
    let min_t = true_colors
        .iter()
        .fold(f32::INFINITY, |acc, &i| f32::min(acc, i));

    let mut generated = Vec::new();
    let mut gen_colors = Vec::new();
    (0..N).step_by(MAX_SIZE).for_each(|_| {
        let t = rng.gen_range(min_t..max_t);
        let gen = model.generate(t, MAX_SIZE, &device);
        let n = gen.len();
        generated.extend(gen);
        gen_colors.extend(std::iter::repeat(t).take(n));
    });

    plot(&[
        Trace::new(generated, gen_colors, "generated"),
        Trace::new(true_pts, true_colors, "true"),
    ]);
}
