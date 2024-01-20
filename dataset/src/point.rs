use burn::tensor::{backend::Backend, ElementConversion, Tensor};

pub const INPUT_DIM: usize = 3;
pub const LABEL_DIM: usize = 1;
pub type Point = [f32; INPUT_DIM];

#[derive(Debug, Clone)]
pub struct SpiralItem {
    pub point: Point,
    pub label: f32,
}

pub trait ToPoints {
    fn to_points(self) -> Vec<Point>;
}

pub trait ToVec {
    fn to_vec(self) -> Vec<Vec<f32>>;
}

impl<B: Backend, const D: usize> ToPoints for Tensor<B, D> {
    fn to_points(self) -> Vec<Point> {
        let chunk_size = *self.dims().last().expect("at least 1 dim");
        self.into_data()
            .value
            .chunks(chunk_size)
            .map(|chunk| std::array::from_fn(|i| chunk[i].elem()))
            .collect()
    }
}

impl<B: Backend, const D: usize> ToVec for Tensor<B, D> {
    fn to_vec(self) -> Vec<Vec<f32>> {
        let chunk_size = *self.dims().last().expect("at least 1 dim");
        self.into_data()
            .value
            .chunks(chunk_size)
            .map(|chunk| chunk.iter().map(|e| e.elem()).collect())
            .collect()
    }
}
