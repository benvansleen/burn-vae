use burn::tensor::{backend::Backend, ElementConversion, Tensor};
#[cfg(not(target_family = "wasm"))]
use burn::train::metric::{
    state::{FormatOptions, NumericMetricState},
    Adaptor, LossInput, Metric, MetricEntry, MetricMetadata, Numeric,
};

pub struct VAEOutput<B: Backend> {
    pub recon_loss: Tensor<B, 1>,
    pub kl_loss: Tensor<B, 1>,
}

impl<B: Backend> VAEOutput<B> {
    pub fn new(recon_loss: Tensor<B, 1>, kl_loss: Tensor<B, 1>) -> Self {
        Self {
            recon_loss,
            kl_loss,
        }
    }
}

#[cfg(not(target_family = "wasm"))]
impl<B: Backend> Adaptor<LossInput<B>> for VAEOutput<B> {
    fn adapt(&self) -> LossInput<B> {
        LossInput::new(self.kl_loss.clone() + self.recon_loss.clone())
    }
}

#[cfg(not(target_family = "wasm"))]
impl<B: Backend> Adaptor<ReconstructionLossInput<B>> for VAEOutput<B> {
    fn adapt(&self) -> ReconstructionLossInput<B> {
        ReconstructionLossInput::new(self.recon_loss.clone())
    }
}

#[cfg(not(target_family = "wasm"))]
impl<B: Backend> Adaptor<KLLossInput<B>> for VAEOutput<B> {
    fn adapt(&self) -> KLLossInput<B> {
        KLLossInput::new(self.kl_loss.clone())
    }
}

pub struct ReconstructionLossInput<B: Backend> {
    tensor: Tensor<B, 1>,
}

impl<B: Backend> ReconstructionLossInput<B> {
    pub fn new(tensor: Tensor<B, 1>) -> Self {
        Self { tensor }
    }
}

#[derive(Default)]
pub struct ReconstructionLossMetric<B: Backend> {
    #[cfg(not(target_family = "wasm"))]
    state: NumericMetricState,
    _b: B,
}

impl<B: Backend> ReconstructionLossMetric<B> {
    pub fn new() -> Self {
        Self::default()
    }
}

#[cfg(not(target_family = "wasm"))]
impl<B: Backend> Metric for ReconstructionLossMetric<B> {
    type Input = ReconstructionLossInput<B>;
    const NAME: &'static str = "Reconstruction Loss";

    fn update(
        &mut self,
        loss: &Self::Input,
        _metadata: &MetricMetadata,
    ) -> MetricEntry {
        let loss = f64::from_elem(
            loss.tensor.clone().mean().into_data().value[0],
        );
        self.state.update(
            loss,
            1,
            FormatOptions::new(Self::NAME).precision(2),
        )
    }

    fn clear(&mut self) {
        self.state.reset();
    }
}

#[cfg(not(target_family = "wasm"))]
impl<B: Backend> Numeric for ReconstructionLossMetric<B> {
    fn value(&self) -> f64 {
        self.state.value()
    }
}

pub struct KLLossInput<B: Backend> {
    tensor: Tensor<B, 1>,
}

impl<B: Backend> KLLossInput<B> {
    pub fn new(tensor: Tensor<B, 1>) -> Self {
        Self { tensor }
    }
}

#[derive(Default)]
pub struct KLLossMetric<B: Backend> {
    #[cfg(not(target_family = "wasm"))]
    state: NumericMetricState,
    _b: B,
}

impl<B: Backend> KLLossMetric<B> {
    pub fn new() -> Self {
        Self::default()
    }
}

#[cfg(not(target_family = "wasm"))]
impl<B: Backend> Metric for KLLossMetric<B> {
    type Input = KLLossInput<B>;
    const NAME: &'static str = "KL Loss";

    fn update(
        &mut self,
        loss: &Self::Input,
        _metadata: &MetricMetadata,
    ) -> MetricEntry {
        let loss = f64::from_elem(
            loss.tensor.clone().mean().into_data().value[0],
        );
        self.state.update(
            loss,
            1,
            FormatOptions::new(Self::NAME).precision(2),
        )
    }

    fn clear(&mut self) {
        self.state.reset();
    }
}

#[cfg(not(target_family = "wasm"))]
impl<B: Backend> Numeric for KLLossMetric<B> {
    fn value(&self) -> f64 {
        self.state.value()
    }
}
