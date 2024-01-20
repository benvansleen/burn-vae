use crate::{
    loss::KLLoss,
    metric::VAEOutput,
    mlp::{MLPBlock, MLPBlockConfig},
};
use burn::{
    config::Config,
    module::Module,
    nn::{loss::MSELoss, loss::Reduction, Linear, LinearConfig},
    tensor::{
        backend::{AutodiffBackend, Backend},
        Distribution, ElementConversion, Tensor,
    },
    train::{TrainOutput, TrainStep, ValidStep},
};
use dataset::{LatentTensor, Point, PointTensor, SpiralBatch};

#[derive(Module, Debug)]
pub struct VAE<B: Backend> {
    encoder: Encoder<B>,
    decoder: Decoder<B>,
    pub kl_weight: f64,
    latent_dim: usize,
}

#[derive(Config, Debug)]
pub struct VAEConfig {
    pub encoder: EncoderConfig,
    decoder: DecoderConfig,
    #[config(default = 1e0)]
    kl_weight: f64,
    #[config(default = 2)]
    latent_dim: usize,
}

impl VAEConfig {
    pub fn init<B: Backend>(&self) -> VAE<B> {
        VAE {
            encoder: self.encoder.init(),
            decoder: self.decoder.init(),
            kl_weight: self.kl_weight,
            latent_dim: self.latent_dim,
        }
    }

    pub fn init_with<B: Backend>(
        &self,
        record: VAERecord<B>,
    ) -> VAE<B> {
        VAE {
            encoder: self.encoder.init_with(record.encoder),
            decoder: self.decoder.init_with(record.decoder),
            kl_weight: self.kl_weight,
            latent_dim: self.latent_dim,
        }
    }
}

impl<B: Backend> VAE<B> {
    pub fn forward(
        &self,
        x: PointTensor<B>,
        y: Tensor<B, 2>,
    ) -> VAEOutput<B> {
        let batchsize = x.dims()[0] as i32;

        let (mu, logvar) = self.encoder.forward(x.clone());
        let kl_loss =
            KLLoss::new().forward(mu.clone(), logvar.clone());

        let std = logvar.exp() / 2;
        let eps = Tensor::random_like(
            &std,
            Distribution::Normal(0., 1.),
        )
        .to_device(&x.device());
        let z = mu + eps * std;

        let y = y.reshape([batchsize, -1, 1]);
        let z = Tensor::cat(vec![z, y], 2);

        let output = self.decoder.forward(z);
        let recon_loss =
            MSELoss::new().forward(output, x, Reduction::Mean);

        VAEOutput::new(
            recon_loss,
            kl_loss.mul_scalar(self.kl_weight),
        )
    }

    pub fn generate(
        &self,
        t: f32,
        n: usize,
        device: &B::Device,
    ) -> Vec<Point> {
        let latent = Tensor::random(
            [n, 1, self.latent_dim],
            Distribution::Normal(0., 1.),
        )
        .to_device(device);
        let t = Tensor::from_floats([t])
            .to_device(device)
            .unsqueeze::<3>()
            .repeat(0, n);

        let latent = Tensor::cat(vec![latent, t], 2);
        let output = self.decoder.forward(latent);
        let last_dim = output.dims()[2];

        output
            .into_data()
            .value
            .chunks(last_dim)
            .map(|chunk| {
                chunk.iter().map(|x| x.elem()).collect()
            })
            .map(|v: Vec<f32>| {
                v.try_into().expect("expected 3D point")
            })
            .collect()
    }

    pub fn encode(
        &self,
        x: Vec<Point>,
    ) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
        let x = Tensor::cat(
            x.into_iter()
                .map(|pt| {
                    Tensor::from_floats(pt).unsqueeze::<3>()
                })
                .collect(),
            0,
        );
        let (mu, log_var) = self.encoder.forward(x);
        let dim = mu.dims()[2];
        let to_vec = |t: Tensor<B, 3>| {
            t.into_data()
                .value
                .chunks(dim)
                .map(|chunk| {
                    chunk.iter().map(|x| x.elem()).collect()
                })
                .collect()
        };

        (to_vec(mu), to_vec(log_var))
    }
}

#[derive(Module, Debug)]
pub struct Encoder<B: Backend> {
    block: MLPBlock<B, 3>,
    fc_mu: Linear<B>,
    fc_logvar: Linear<B>,
}

#[derive(Config, Debug)]
pub struct EncoderConfig {
    pub block_config: MLPBlockConfig,
    fc_mu: LinearConfig,
    fc_logvar: LinearConfig,
}

impl EncoderConfig {
    pub fn init<B: Backend>(&self) -> Encoder<B> {
        Encoder {
            block: self.block_config.init(),
            fc_mu: self.fc_mu.init(),
            fc_logvar: self.fc_logvar.init(),
        }
    }

    pub fn init_with<B: Backend>(
        &self,
        record: EncoderRecord<B>,
    ) -> Encoder<B> {
        Encoder {
            block: self.block_config.init_with(record.block),
            fc_mu: self.fc_mu.init_with(record.fc_mu),
            fc_logvar: self
                .fc_logvar
                .init_with(record.fc_logvar),
        }
    }
}

impl<B: Backend> Encoder<B> {
    pub fn forward(
        &self,
        input: PointTensor<B>,
    ) -> (LatentTensor<B>, LatentTensor<B>) {
        let x = self.block.forward(input);
        let mu = self.fc_mu.forward(x.clone());
        let logvar = self.fc_logvar.forward(x.clone());

        (mu, logvar)
    }
}

#[derive(Module, Debug)]
pub struct Decoder<B: Backend> {
    block: MLPBlock<B, 3>,
    fc: Linear<B>,
}

#[derive(Config, Debug)]
pub struct DecoderConfig {
    block_config: MLPBlockConfig,
    fc: LinearConfig,
}

impl DecoderConfig {
    pub fn init<B: Backend>(&self) -> Decoder<B> {
        Decoder {
            block: self.block_config.init(),
            fc: self.fc.init(),
        }
    }

    pub fn init_with<B: Backend>(
        &self,
        record: DecoderRecord<B>,
    ) -> Decoder<B> {
        Decoder {
            block: self.block_config.init_with(record.block),
            fc: self.fc.init_with(record.fc),
        }
    }
}

impl<B: Backend> Decoder<B> {
    pub fn forward(
        &self,
        input: PointTensor<B>,
    ) -> LatentTensor<B> {
        let x = self.block.forward(input);
        self.fc.forward(x)
    }
}

impl<B: AutodiffBackend> TrainStep<SpiralBatch<B>, VAEOutput<B>>
    for VAE<B>
{
    fn step(
        &self,
        batch: SpiralBatch<B>,
    ) -> TrainOutput<VAEOutput<B>> {
        let prediction =
            self.forward(batch.points, batch.labels);
        let loss = prediction.recon_loss.clone()
            + prediction.kl_loss.clone();

        TrainOutput::new(self, loss.backward(), prediction)
    }
}

impl<B: Backend> ValidStep<SpiralBatch<B>, VAEOutput<B>>
    for VAE<B>
{
    fn step(&self, batch: SpiralBatch<B>) -> VAEOutput<B> {
        self.forward(batch.points, batch.labels)
    }
}
