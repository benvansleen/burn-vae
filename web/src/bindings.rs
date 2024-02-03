use wasm_bindgen::prelude::*;
use plotly::Plot;

#[wasm_bindgen(module = "/public/plot.js")]
extern "C" {
    fn _update(id: &str, p: String);
}

pub async fn update(id: &str, p: &Plot) {
    _update(id, p.data().to_json());
}
