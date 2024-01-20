use flume::Sender;
use pyo3::{prelude::*, types::IntoPyDict};

pub const INPUT_DIM: usize = 3;
pub const LABEL_DIM: usize = 1;
pub type Point = [f32; INPUT_DIM];

#[derive(Debug, Clone)]
pub struct SpiralItem {
    pub point: Point,
    pub label: f32,
}

pub fn generate_data(n_samples: u32, tx: Sender<SpiralItem>) {
    Python::with_gil(|py| {
        let ds = py.import("sklearn.datasets").unwrap();
        let swiss = ds.getattr("make_swiss_roll").unwrap();
        let kwargs = [("n_samples", n_samples)].into_py_dict(py);

        loop {
            let (points, ts): (Vec<Point>, Vec<f32>) = swiss
                .call((), Some(kwargs))
                .expect("to generate swiss roll")
                .extract()
                .expect("to extract swiss roll");

            points.into_iter().zip(ts).for_each(|(pt, t)| {
                tx.send(SpiralItem {
                    point: pt,
                    label: t,
                })
                .expect("to send item to channel");
            });
        }
    })
}