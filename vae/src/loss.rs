use burn::tensor::{backend::Backend, Tensor};

pub struct KLLoss {}

impl KLLoss {
    pub fn new() -> Self {
        Self {}
    }

    pub fn forward<B: Backend, const D: usize>(
        &self,
        mean: Tensor<B, D>,
        log_var: Tensor<B, D>,
    ) -> Tensor<B, 1> {
        (log_var.clone().add_scalar(1.) - mean.powf(2.) - log_var.exp())
            .sum_dim(1)
            .mean()
            .mul_scalar(-0.5)
    }
}

impl Default for KLLoss {
    fn default() -> Self {
        Self::new()
    }
}
