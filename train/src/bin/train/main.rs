mod config;

#[cfg(not(target_family = "wasm"))]
fn main() {
    use burn::backend::{
        Autodiff, Fusion, Wgpu,
        wgpu::{AutoGraphicsApi, WgpuDevice},
    };

    use config::config;
    use dataset::get_data;
    use rand::Rng;
    use train::train;
    use train::{
        load_model,
        visualization::{Trace, plot},
    };
    type Backend = Fusion<Wgpu<AutoGraphicsApi, f32, i32>>;
    const DEVICE: WgpuDevice = WgpuDevice::BestAvailable;

    let artifacts_dir = &std::env::args()
        .nth(1)
        .unwrap_or("model_artifacts".to_string());

    train::<Autodiff<Backend>>(artifacts_dir, config(), &DEVICE);
    let model = load_model::<Backend>(artifacts_dir, &DEVICE);

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
        let current_generated = model.generate(t, MAX_SIZE, &DEVICE);
        let n = current_generated.len();
        generated.extend(current_generated);
        gen_colors.extend(std::iter::repeat(t).take(n));
    });

    plot(&[
        Trace::new(generated, gen_colors, "generated"),
        Trace::new(true_pts, true_colors, "true"),
    ])
    .show();
}

#[cfg(target_family = "wasm")]
fn main() {
    panic!("This is not supported on wasm");
}
