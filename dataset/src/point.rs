use burn::tensor::{backend::Backend, ElementConversion, Tensor};
#[cfg(target_family = "wasm")]
use std::future::Future;


pub const INPUT_DIM: usize = 3;
pub const LABEL_DIM: usize = 1;
pub type Point = [f32; INPUT_DIM];

#[derive(Debug, Clone)]
pub struct SpiralItem {
    pub point: Point,
    pub label: f32,
}

#[cfg(not(target_family = "wasm"))]
pub trait ToPoints {
    fn to_points(self) -> Vec<Point>;
}

#[cfg(not(target_family = "wasm"))]
pub trait ToVec {
    fn to_vec(self) -> Vec<Vec<f32>>;
}

#[cfg(not(target_family = "wasm"))]
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

#[cfg(not(target_family = "wasm"))]
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

#[cfg(target_family = "wasm")]
pub trait ToPoints {
    fn to_points(self) -> impl Future<Output=Vec<Point>>;
}

#[cfg(target_family = "wasm")]
pub trait ToVec {
    fn to_vec(self) -> impl Future<Output=Vec<Vec<f32>>>;
}

#[cfg(target_family = "wasm")]
impl<B: Backend, const D: usize> ToPoints for Tensor<B, D> {
    fn to_points(self) -> impl Future<Output=Vec<Point>> {
        let chunk_size = *self.dims().last().expect("at least 1 dim");
        async move {
            self.into_data()
                .await
                .value
                .chunks(chunk_size)
                .map(|chunk| std::array::from_fn(|i| chunk[i].elem()))
                .collect()
        }
    }
}

#[cfg(target_family = "wasm")]
impl<B: Backend, const D: usize> ToVec for Tensor<B, D> {
    fn to_vec(self) -> impl Future<Output=Vec<Vec<f32>>> {
        let chunk_size = *self.dims().last().expect("at least 1 dim");
        async move {
            self.into_data()
                .await
                .value
                .chunks(chunk_size)
                .map(|chunk| chunk.iter().map(|e| e.elem()).collect())
                .collect()
        }
    }
}
