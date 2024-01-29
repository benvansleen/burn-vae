use crate::point::{Point, SpiralItem};
use flume::Sender;

pub fn generate_data(n_samples: u32, tx: Sender<SpiralItem>) {
    #[cfg(not(target_family = "wasm"))]
    {
        use pyo3::{prelude::*, types::IntoPyDict};

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
}
