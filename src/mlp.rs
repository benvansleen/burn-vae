use burn::{
    config::Config,
    module::Module,
    nn::{
        Dropout, DropoutConfig, LayerNorm, LayerNormConfig,
        Linear, LinearConfig, GELU,
    },
    tensor::{backend::Backend, Tensor},
};

#[derive(Module, Debug)]
pub struct MLPBlock<B: Backend, const D: usize> {
    layers: Vec<Linear<B>>,
    dropout: Dropout,
    activation: GELU,
    norm: LayerNorm<B>,
    norm_last: LayerNorm<B>,
}

#[derive(Config, Debug)]
pub struct MLPBlockConfig {
    n_layers: usize,
    pub hidden_dim: usize,
    input_dim: usize,
    pub output_dim: usize,
    #[config(default = "0.5")]
    dropout: f64,
}

impl<B: Backend, const D: usize> MLPBlock<B, D> {
    pub fn forward(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        self.layers.iter().enumerate().fold(
            x,
            |x, (i, layer)| {
                let norm = if i < self.layers.len() - 1 {
                    &self.norm
                } else {
                    &self.norm_last
                };

                let orig = x.clone();
                let mut x = layer.forward(x);
                x = self.dropout.forward(x);
                x = norm.forward(x);
                x = self.activation.forward(x);

                if i > 0 && i < self.layers.len() - 1 {
                    x = x + orig;
                }

                x
            },
        )
    }
}

impl MLPBlockConfig {
    fn build_layers(&self) -> Vec<LinearConfig> {
        let mut layers = (0..self.n_layers - 1).fold(
            vec![LinearConfig::new(
                self.input_dim,
                self.hidden_dim,
            )],
            |mut layers, _| {
                layers.push(LinearConfig::new(
                    self.hidden_dim,
                    self.hidden_dim,
                ));
                layers
            },
        );

        layers.push(LinearConfig::new(
            self.hidden_dim,
            self.output_dim,
        ));

        layers
    }
    pub fn init<B: Backend, const D: usize>(
        &self,
    ) -> MLPBlock<B, D> {
        MLPBlock {
            layers: self
                .build_layers()
                .into_iter()
                .map(|config| config.init())
                .collect(),
            dropout: DropoutConfig::new(self.dropout).init(),
            activation: GELU::new(),
            norm: LayerNormConfig::new(self.hidden_dim).init(),
            norm_last: LayerNormConfig::new(self.output_dim)
                .init(),
        }
    }

    pub fn init_with<B: Backend, const D: usize>(
        &self,
        record: MLPBlockRecord<B, D>,
    ) -> MLPBlock<B, D> {
        MLPBlock {
            layers: self
                .build_layers()
                .into_iter()
                .zip(record.layers)
                .map(|(config, layer)| config.init_with(layer))
                .collect(),
            dropout: DropoutConfig::new(self.dropout).init(),
            activation: GELU::new(),
            norm: LayerNormConfig::new(self.hidden_dim)
                .init_with(record.norm),
            norm_last: LayerNormConfig::new(self.output_dim)
                .init_with(record.norm_last),
        }
    }
}
