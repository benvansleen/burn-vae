use burn::{
    data::{dataloader::batcher::Batcher, dataset::Dataset},
    tensor::{backend::Backend, Data, Tensor},
};
use crossbeam_channel::{bounded, Receiver, Sender};
use once_cell::sync::OnceCell;
use pyo3::{prelude::*, types::IntoPyDict};

pub const INPUT_DIM: usize = 4;
type Point = [f32; INPUT_DIM];
pub type PointTensor<B> = Tensor<B, 3>;
pub type LatentTensor<B> = Tensor<B, 3>;

static CHAN: OnceCell<(
    Sender<SpiralItem>,
    Receiver<SpiralItem>,
)> = OnceCell::new();
static T: OnceCell<std::thread::JoinHandle<()>> =
    OnceCell::new();

#[derive(Debug, Clone)]
pub struct SpiralItem {
    pub point: Point,
}

#[derive(Debug, Clone)]
pub struct SpiralDataset {
    pub epoch_size: usize,
    ch: Receiver<SpiralItem>,
}

impl SpiralDataset {
    pub fn new(epoch_size: usize) -> Self {
        let (tx, rx) = CHAN.get_or_init(|| bounded(100_000));
        T.get_or_init(|| {
            std::thread::spawn(move || {
                Python::with_gil(|py| {
                    let ds =
                        py.import("sklearn.datasets").unwrap();
                    let swiss =
                        ds.getattr("make_swiss_roll").unwrap();
                    let kwargs =
                        [("n_samples", 10_000)].into_py_dict(py);

                    loop {
                        let (points, ts): (
                            Vec<[f32; 3]>,
                            Vec<f32>,
                        ) = swiss
                            .call((), Some(kwargs))
                            .unwrap()
                            .extract()
                            .unwrap();

                        points.into_iter().zip(ts).for_each(
                            |(pt, t)| {
                                if tx
                                    .send(SpiralItem {
                                        point: [
                                            pt[0], pt[1], pt[2],
                                            t,
                                        ],
                                    })
                                    .is_err()
                                {
                                    std::thread::yield_now();
                                }
                            },
                        );
                    }
                })
            })
        });

        Self {
            epoch_size,
            ch: rx.clone(),
        }
    }
}

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
    pub points: PointTensor<B>,
}

impl<B: Backend> Batcher<SpiralItem, SpiralBatch<B>>
    for SpiralBatcher<B>
{
    fn batch(&self, items: Vec<SpiralItem>) -> SpiralBatch<B> {
        let points = items
            .iter()
            .map(|item| Data::<f32, 1>::from(item.point))
            .map(|data| {
                Tensor::<B, 1>::from_data(data.convert())
            })
            .map(|tensor| {
                tensor.clone().reshape([1, 1, tensor.dims()[0]])
            })
            .collect();

        let points =
            Tensor::cat(points, 0).to_device(&self.device);

        SpiralBatch { points }
    }
}

pub fn get_data(n: usize) -> Vec<Vec<f32>> {
    let dataset = SpiralDataset::new(n);
    (0..n)
        .map(|i| dataset.get(i).unwrap().point.to_vec())
        .collect()
}
