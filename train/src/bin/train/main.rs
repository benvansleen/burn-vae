use burn::backend::{
    wgpu::{AutoGraphicsApi, WgpuDevice},
    Autodiff, Wgpu,
};

use dataset::get_data;
use rand::Rng;
use train::{
    load_model,
    visualization::{plot, Trace},
};
mod config;
use config::train_model;

type Backend = Wgpu<AutoGraphicsApi, f32, i32>;

fn main() {
    let device = WgpuDevice::BestAvailable;
    let artifacts_dir =
        &std::env::args().nth(1).unwrap_or("artifacts".to_string());

    train_model::<Autodiff<Backend>>(artifacts_dir, &device);
    let model = load_model::<Backend>(artifacts_dir, &device);

    const N: usize = 5000;
    const MAX_SIZE: usize = 128;

    let (true_pts, true_colors) = get_data(N);

    let mut rng = rand::thread_rng();
    let max_t = true_colors
        .iter()
        .fold(f32::NEG_INFINITY, |acc, &i| f32::max(acc, i));
    let min_t = true_colors
        .iter()
        .fold(f32::INFINITY, |acc, &i| f32::min(acc, i));

    let mut generated = Vec::new();
    let mut gen_colors = Vec::new();
    (0..N).step_by(MAX_SIZE).for_each(|_| {
        let t = rng.gen_range(min_t..max_t);
        let gen = model.generate(t, MAX_SIZE, &device);
        let n = gen.len();
        generated.extend(gen);
        gen_colors.extend(std::iter::repeat(t).take(n));
    });

    plot(&[
        Trace::new(generated, gen_colors, "generated"),
        Trace::new(true_pts, true_colors, "true"),
    ]);
}
