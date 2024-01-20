use train::{train, TrainingConfig};

use burn::{
    nn::LinearConfig,
    optim::AdamWConfig,
    tensor::{backend::AutodiffBackend, Device},
};
use dataset::{INPUT_DIM, LABEL_DIM};
use vae::{DecoderConfig, EncoderConfig, MLPBlockConfig, ModelConfig};

const LATENT_DIM: usize = 2;

pub fn train_model<B: AutodiffBackend>(
    artifact_dir: &str,
    device: &Device<B>,
) {
    train::<B>(
        artifact_dir,
        TrainingConfig::new(
            ModelConfig::new(
                EncoderConfig::new(
                    MLPBlockConfig::new(3, 128, INPUT_DIM, 32)
                        .with_dropout(0.1),
                    LinearConfig::new(32, LATENT_DIM).with_bias(true),
                    LinearConfig::new(32, LATENT_DIM).with_bias(true),
                ),
                DecoderConfig::new(
                    MLPBlockConfig::new(
                        6,
                        128,
                        LATENT_DIM + LABEL_DIM,
                        64,
                    )
                    .with_dropout(0.1),
                    LinearConfig::new(64, INPUT_DIM).with_bias(true),
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
        .with_early_stop_patience(50)
        .with_learning_rate(8e1),
        device,
    );
}
