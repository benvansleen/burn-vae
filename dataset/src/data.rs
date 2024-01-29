use crate::{
    point::{Point, SpiralItem},
    workers,
};
#[cfg(not(target_family = "wasm"))]
use burn::data::{dataloader::batcher::Batcher, dataset::Dataset};
use burn::tensor::{backend::Backend, Tensor};
use flume::Receiver;

#[derive(Debug, Clone)]
pub struct SpiralDataset {
    pub epoch_size: usize,
    ch: Receiver<SpiralItem>,
}

impl SpiralDataset {
    pub fn new(epoch_size: usize) -> Self {
        Self {
            epoch_size,
            ch: workers::init(),
        }
    }
}

#[cfg(not(target_family = "wasm"))]
impl Dataset<SpiralItem> for SpiralDataset {
    fn get(&self, _idx: usize) -> Option<SpiralItem> {
        self.ch.recv().ok()
    }

    fn len(&self) -> usize {
        self.epoch_size
    }
}

#[derive(Debug, Clone)]
pub struct SpiralBatcher<B: Backend> {
    device: B::Device,
}

impl<B: Backend> SpiralBatcher<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }
}

#[derive(Debug, Clone)]
pub struct SpiralBatch<B: Backend> {
    pub points: Tensor<B, 3>,
    pub labels: Tensor<B, 2>,
}

#[cfg(not(target_family = "wasm"))]
impl<B: Backend> Batcher<SpiralItem, SpiralBatch<B>> for SpiralBatcher<B> {
    fn batch(&self, items: Vec<SpiralItem>) -> SpiralBatch<B> {
        let batch = items
            .iter()
            .map(|item| {
                (
                    Tensor::<B, 1>::from_floats(item.point),
                    Tensor::<B, 1>::from_floats([item.label]),
                )
            })
            .map(|(p, l)| {
                let (p_dim, l_dim) = (p.dims()[0], l.dims()[0]);
                (p.reshape([1, 1, p_dim]), l.reshape([1, l_dim]))
            });

        let points =
            Tensor::cat(batch.clone().map(|(p, _)| p).collect(), 0)
                .to_device(&self.device);
        let labels = Tensor::cat(batch.map(|(_, l)| l).collect(), 0)
            .to_device(&self.device);

        SpiralBatch { points, labels }
    }
}

#[cfg(not(target_family = "wasm"))]
pub fn get_data(n: usize) -> (Vec<Point>, Vec<f32>) {
    let dataset = SpiralDataset::new(n);
    (0..n)
        .map(|i| dataset.get(i).unwrap())
        .map(|item| (item.point, item.label))
        .unzip()
}
