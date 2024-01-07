use crate::{
    data::{SpiralBatch, SpiralBatcher, SpiralDataset},
    metric::{
        KLLossMetric, ReconstructionLossMetric, VAEOutput,
    },
    model::{VAEConfig, VAE},
};
use burn::{
    config::Config,
    data::dataloader::DataLoaderBuilder,
    lr_scheduler::noam::NoamLrSchedulerConfig,
    module::Module,
    optim::AdamWConfig,
    record::CompactRecorder,
    tensor::backend::{AutodiffBackend, Backend},
    train::{
        metric::{
            store::{Aggregate, Direction, Split},
            LossMetric,
        },
        LearnerBuilder, MetricEarlyStoppingStrategy,
        StoppingCondition, TrainOutput, TrainStep, ValidStep,
    },
};

#[derive(Config)]
pub struct TrainingConfig {
    pub model: VAEConfig,
    pub optimizer: AdamWConfig,
    #[config(default = 10)]
    pub num_epochs: usize,
    #[config(default = 64)]
    pub batch_size: usize,
    #[config(default = 1)]
    pub num_workers: usize,
    #[config(default = 42)]
    pub seed: u64,
    #[config(default = 3e-3)]
    pub learning_rate: f64,
    #[config(default = 1000)]
    pub warmup_steps: usize,
    #[config(default = 10)]
    pub early_stop_patience: usize,
}

pub fn train<B: AutodiffBackend>(
    artifact_dir: &str,
    config: TrainingConfig,
    device: &B::Device,
) {
    std::fs::create_dir_all(artifact_dir).ok();
    config
        .save(format!("{artifact_dir}/config.json"))
        .expect("Configuration should be saved successfully");

    B::seed(config.seed);

    let train_batcher = SpiralBatcher::<B>::new(device.clone());
    let valid_batcher =
        SpiralBatcher::<B::InnerBackend>::new(device.clone());

    let train_loader = DataLoaderBuilder::new(train_batcher)
        .batch_size(config.batch_size)
        .num_workers(config.num_workers)
        .build(SpiralDataset::new(10_000));
    let valid_loader = DataLoaderBuilder::new(valid_batcher)
        .batch_size(config.batch_size)
        .num_workers(config.num_workers)
        .build(SpiralDataset::new(100));

    let scheduler =
        NoamLrSchedulerConfig::new(config.learning_rate)
            .with_warmup_steps(config.warmup_steps)
            .with_model_size(
                config.model.encoder.block_config.hidden_dim,
            )
            .init();

    let learner = LearnerBuilder::new(artifact_dir)
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .metric_train_numeric(ReconstructionLossMetric::new())
        .metric_valid_numeric(ReconstructionLossMetric::new())
        .metric_train_numeric(KLLossMetric::new())
        .metric_valid_numeric(KLLossMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .early_stopping(MetricEarlyStoppingStrategy::new::<
            ReconstructionLossMetric<B>,
        >(
            Aggregate::Mean,
            Direction::Lowest,
            Split::Valid,
            StoppingCondition::NoImprovementSince {
                n_epochs: config.early_stop_patience,
            },
        ))
        .devices(vec![device.clone()])
        .num_epochs(config.num_epochs)
        .build(
            config.model.init::<B>(),
            config.optimizer.init(),
            scheduler,
        );

    let model = learner.fit(train_loader, valid_loader);
    model
        .save_file(
            format!("{artifact_dir}/model"),
            &CompactRecorder::new(),
        )
        .expect("Trained model should be saved successfully");
}

impl<B: AutodiffBackend> TrainStep<SpiralBatch<B>, VAEOutput<B>>
    for VAE<B>
{
    fn step(
        &self,
        batch: SpiralBatch<B>,
    ) -> TrainOutput<VAEOutput<B>> {
        let prediction = self.forward(batch.points);
        let loss = prediction.recon_loss.clone()
            + prediction.kl_loss.clone();

        TrainOutput::new(self, loss.backward(), prediction)
    }
}

impl<B: Backend> ValidStep<SpiralBatch<B>, VAEOutput<B>>
    for VAE<B>
{
    fn step(&self, batch: SpiralBatch<B>) -> VAEOutput<B> {
        self.forward(batch.points)
    }
}
