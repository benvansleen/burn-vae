use crate::error_template::{AppError, ErrorTemplate};
use leptos::*;
use leptos_dom::log;
use leptos_meta::*;
use leptos_router::*;
use thaw::{Button, Slider};
use std::path::Path;
use std::sync::Arc;

use burn::config::Config;
use inference::ModelConfig;
use train::visualization::{plot, Trace};
    use rand::Rng;

type Points = Vec<inference::Point>;

#[component]
pub fn App() -> impl IntoView {
    provide_meta_context();

    view! {
        <head>
            <Stylesheet id="leptos" href="/pkg/web.css"/>
            <Script src="https://cdn.plot.ly/plotly-2.27.0.min.js"/>
            <Title text="C-VAE"/>
        </head>

        <Router fallback=|| {
            let mut outside_errors = Errors::default();
            outside_errors.insert_with_default_key(AppError::NotFound);
            view! { <ErrorTemplate outside_errors/> }.into_view()
        }>
            <main>
                <Routes>
                    <Route path="" view=Main/>
                </Routes>
            </main>
        </Router>
    }
}

static MODEL_BYTES: &[u8] =
    include_bytes!("../../model_artifacts/model.bin");

#[component]
fn Main() -> impl IntoView {
    let (true_pts, true_color) = dataset::get_data(2000);
    let model_config = ModelConfig::load(
        Path::new("model_artifacts").join("config.json"),
    ).expect("Model not found");

    view! {
        <Plot model_config true_pts true_color/>
    }
}

async fn generate(
    r: f32,
) -> (Points, Vec<f32>) {
    log!("generating");

    const MAX_SIZE: usize = 2;
    let mut generated = Vec::new();
    let mut gen_colors = Vec::new();

    let (min_t, max_t) = (6., 20.);
    for _ in (0..50).step_by(MAX_SIZE) {
        let r = r + rand::thread_rng().gen_range(-0.25..0.25);
        #[cfg(target_family = "wasm")]
        let gen = inference::generate(r, MAX_SIZE).await;
        #[cfg(not(target_family = "wasm"))]
        let gen = inference::generate(r, MAX_SIZE);
        let n = gen.len();
        generated.extend(gen);
        gen_colors.extend(std::iter::repeat(r).take(n));
    }

    log!("generated");
    (generated, gen_colors)
}

#[island]
fn Plot(
    model_config: inference::ModelConfig,
    true_pts: Points,
    true_color: Vec<f32>,
) -> impl IntoView {
    let (pt_buf, set_pt_buf) = create_signal(Points::new());
    let (col_buf, set_col_buf) = create_signal(Vec::<f32>::new());
    let r = create_rw_signal(6.);

    inference::load_bytes(
        model_config,
        MODEL_BYTES.to_vec(),
    );
    let generated =
        create_local_resource(|| (), move |_| {
            generate(r() as f32)
        });

    let id = "plot-div";
    let (true_pts, true_color) =
        (Arc::new(true_pts), Arc::new(true_color));
    #[cfg(target_family = "wasm")]
    let _ = create_local_resource(
        || (), move |_| {
            let (g, c) = generated.get().unwrap_or((vec![], vec![]));
            set_pt_buf.update(|pt| {
                pt.extend(g);
            });
            set_col_buf.update(|col| {
                col.extend(c);
            });
            render_plot(
                id,
                pt_buf(),
                col_buf(),
                true_pts.to_vec(),
                true_color.to_vec(),
            )
        },
    );

    let clear = move |_| {
        set_pt_buf.update(|pt| pt.clear());
        set_col_buf.update(|col| col.clear());
    };
    view! {
        <div id={id} />
        <Slider value=r max=20. step=0.5/>
        <Button on:click=clear>
        "Clear generated points"
        </Button>
    }
}

#[cfg(target_family = "wasm")]
async fn render_plot(
    id: &str,
    generated: Vec<inference::Point>,
    gen_colors: Vec<f32>,
    true_pts: Vec<inference::Point>,
    true_colors: Vec<f32>,
) {
    log!("plotting");
    let p = plot(&[
        Trace::new(generated, gen_colors, "generated"),
        Trace::new(true_pts, true_colors, "true"),
    ]);
    crate::bindings::update(id, &p).await;
    log!("finished plotting");
}
