use burn::{
    config::Config,
    module::Module,
    nn::{
        Dropout, DropoutConfig, LayerNorm, LayerNormConfig, Linear,
        LinearConfig, GELU,
    },
    tensor::{backend::Backend, Tensor},
};

#[derive(Module, Debug)]
pub struct MLPBlock<B: Backend, const D: usize> {
    layers: Vec<Linear<B>>,
    final_layer: Linear<B>,
    dropout: Dropout,
    activation: GELU,
    norm: LayerNorm<B>,
    final_norm: LayerNorm<B>,
}

#[derive(Config, Debug)]
pub struct MLPBlockConfig {
    n_layers: usize,
    pub hidden_dim: usize,
    input_dim: usize,
    pub output_dim: usize,
    #[config(default = 0.5)]
    dropout: f64,
}

impl<B: Backend, const D: usize> MLPBlock<B, D> {
    pub fn forward(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        let x = self.layers.iter().fold(x, |x, layer| {
            let input = x.clone();
            let mut x = layer.forward(x);
            x = self.dropout.forward(x);
            x = self.norm.forward(x);
            x = self.activation.forward(x);
            if x.shape() == input.shape() {
                x = x + input;
            }

            x
        });

        let mut x = self.final_layer.forward(x);
        x = self.dropout.forward(x);
        x = self.final_norm.forward(x);
        self.activation.forward(x)
    }
}

impl MLPBlockConfig {
    fn build_layers(&self) -> Vec<LinearConfig> {
        (0..self.n_layers - 1).fold(
            vec![LinearConfig::new(self.input_dim, self.hidden_dim)],
            |mut layers, _| {
                layers.push(LinearConfig::new(
                    self.hidden_dim,
                    self.hidden_dim,
                ));
                layers
            },
        )
    }

    pub fn init<B: Backend, const D: usize>(&self) -> MLPBlock<B, D> {
        MLPBlock {
            layers: self
                .build_layers()
                .into_iter()
                .map(|config| config.init())
                .collect(),
            final_layer: LinearConfig::new(
                self.hidden_dim,
                self.output_dim,
            )
            .init(),
            dropout: DropoutConfig::new(self.dropout).init(),
            activation: GELU::new(),
            norm: LayerNormConfig::new(self.hidden_dim).init(),
            final_norm: LayerNormConfig::new(self.output_dim).init(),
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
            final_layer: LinearConfig::new(
                self.hidden_dim,
                self.output_dim,
            )
            .init_with(record.final_layer),
            dropout: DropoutConfig::new(self.dropout).init(),
            activation: GELU::new(),
            norm: LayerNormConfig::new(self.hidden_dim)
                .init_with(record.norm),
            final_norm: LayerNormConfig::new(self.output_dim)
                .init_with(record.final_norm),
        }
    }
}
