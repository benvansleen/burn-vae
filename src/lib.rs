pub mod loss;
pub mod metric;
pub mod mlp;
pub mod model;
pub mod train;
pub mod visualization;

use pyo3::prelude::*;
pub mod data;

#[pymodule]
fn _burn_vae(_py: Python, m: &PyModule) -> PyResult<()> {
    #[pyfn(m)]
    fn get_data() -> Vec<Vec<f32>> {
        data::get_data(100)
    }
    Ok(())
}
