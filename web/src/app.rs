use crate::error_template::{AppError, ErrorTemplate};
use burn::{
    backend::{ndarray::NdArrayDevice, NdArray},
    config::Config,
    module::Module,
    record::{BinBytesRecorder, FullPrecisionSettings, Recorder},
};
use leptos::*;
use leptos_dom::log;
use leptos_meta::*;
use leptos_router::*;
type Backend = NdArray<f32>;
use inference::ModelConfig;
use train::visualization::{plot, Trace};

#[component]
pub fn App() -> impl IntoView {
    // Provides context that manages stylesheets, titles, meta tags, etc.
    provide_meta_context();

    view! {
        <head>
        <Stylesheet id="leptos" href="/pkg/hello-leptos.css"/>
        <Script src="https://cdn.plot.ly/plotly-2.14.0.min.js"/>
        <Title text="Welcome to Leptos"/>
        </head>

        // content for this welcome page
        <Router fallback=|| {
            let mut outside_errors = Errors::default();
            outside_errors.insert_with_default_key(AppError::NotFound);
            view! { <ErrorTemplate outside_errors/> }.into_view()
        }>
            <main>
                <Routes>
                    <Route path="" view=HomePage/>
                </Routes>
            </main>
        </Router>
    }
}

static MODEL_BYTES: &[u8] =
    include_bytes!("../../model_artifacts/model.bin");

const ASSETS: &str = "target/site";
const N: usize = 5000;

fn asset(path: &str) -> String {
    std::path::Path::new(ASSETS)
        .join(path)
        .to_str()
        .unwrap()
        .to_string()
}

/// Renders the home page of your application.
#[component]
fn HomePage() -> impl IntoView {
    let config = ModelConfig::load(
        std::path::Path::new("model_artifacts").join("config.json"),
    )
    .expect("Model not found");

    let count_start = 1;
    let files = ["a.txt", "b.txt", "c.txt"];
    let labels: Vec<_> = files.iter().map(|f| f.to_string()).collect();
    let tabs = move || {
        files
            .into_iter()
            .enumerate()
            .map(|(index, f)| {
                let content = std::fs::read_to_string(asset(f))
                    .expect("that file exists");
                view! {
                    <Tab index>
                    <h2>{f}</h2>
                    <p>{content}</p>
                    </Tab>
                }
            })
            .collect_view()
    };

    // let x = vec![1, 2, 3];
    // let y = vec![1, 2, 3];
    // let z = vec![1, 2, 3];
    #[cfg(not(target_family = "wasm"))]
    let (true_pts, true_color) = dataset::get_data(N);
    #[cfg(target_family = "wasm")]
    let (true_pts, true_color) = (vec![], vec![]);

    view! {
        <h1>"Hi, Ben!"</h1>
        <Counter {count_start}/>
        <Tabs labels>
        <div>{tabs()}</div>
        </Tabs>
        <Plot config true_pts true_color/>
    }
}

// async fn plot(id: &str, x: Vec<i32>, y: Vec<i32>, z: Vec<i32>) {
//     use plotly::{Plot, Scatter3D};
//     let trace = Scatter3D::new(x, y, z);
//     let mut plot = Plot::new();
//     plot.add_trace(trace);
//     log!("plotting");
//     plotly::bindings::new_plot(id, &plot).await;
// }

use std::sync::Arc;
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
    plotly::bindings::new_plot(id, &p).await;
    log!("finished plotting");
}

async fn generate(
    model: inference::Model<Backend>,
) -> (Vec<inference::Point>, Vec<f32>) {
    use rand::Rng;
    log!("generating");
    let device = NdArrayDevice::default();

    let mut rng = rand::thread_rng();
    const MAX_SIZE: usize = 10;
    let mut generated = Vec::new();
    let mut gen_colors = Vec::new();

    let (min_t, max_t) = (6., 20.);
    for _ in (0..N).step_by(MAX_SIZE) {
        let t = rng.gen_range(min_t..max_t);
        #[cfg(target_family = "wasm")]
        let gen = model.generate(t, MAX_SIZE, &device).await;
        #[cfg(not(target_family = "wasm"))]
        let gen = model.generate(t, MAX_SIZE, &device);
        let n = gen.len();
        generated.extend(gen);
        gen_colors.extend(std::iter::repeat(t).take(n));
    }

    log!("generated");
    (generated, gen_colors)
}

#[island]
fn Plot(
    config: ModelConfig,
    true_pts: Vec<inference::Point>,
    true_color: Vec<f32>,
) -> impl IntoView {
    let record = BinBytesRecorder::<FullPrecisionSettings>::default()
        .load(MODEL_BYTES.to_vec())
        .expect("to load model from bytes");
    let model = config.model.init_with::<Backend>(record);

    let generated =
        create_local_resource(|| (), move |_| generate(model.clone()));

    let id = "plot-div";
    // let true_dist = Arc::new(true_dist);
    let (true_pts, true_color) =
        (Arc::new(true_pts), Arc::new(true_color));
    // let x = Arc::new(x);
    // let y = Arc::new(y);
    // let z = Arc::new(z);

    // let _ = create_local_resource(|| (), move |_| plot(id, x.to_vec(), y.to_vec(), z.to_vec()));

    let _ = create_local_resource(
        || (),
        move |_| match generated.get() {
            Some((g, c)) => render_plot(
                id,
                g,
                c,
                true_pts.to_vec(),
                true_color.to_vec(),
            ),
            None => render_plot(
                id,
                vec![],
                vec![],
                true_pts.to_vec(),
                true_color.to_vec(),
            ),
        },
    );

    view! {
        <div id={id}></div>
    }
}

#[island]
fn Tabs(labels: Vec<String>, children: Children) -> impl IntoView {
    let (selected, set_selected) = create_signal(0usize);
    provide_context(selected);

    log!("rendering tabs");
    log!("{:?}", selected);

    let buttons = labels
        .into_iter()
        .enumerate()
        .map(|(index, label)| {
            let select = move |_| set_selected(index);
            view! { <button on:click=select>{label}</button> }
        })
        .collect_view();
    view! {
        <div style="display: flex; width: 100%; justify-content: space-between;">
            {buttons}
        </div>
        {children()}
    }
}

#[island]
fn Tab(index: usize, children: Children) -> impl IntoView {
    let selected: ReadSignal<usize> = expect_context();
    view! {
        <div style:display=move || {
            if selected() == index { "block" } else { "none" }
        }>{children()}</div>
    }
}

#[island]
fn Counter(#[prop(optional)] start: i32) -> impl IntoView {
    let (count, set_count) = create_signal(start);
    let on_click = move |_| set_count.update(|count| *count += 1);

    view! {
        <button on:click=on_click>"Click Me: " {count}</button>
    }
}
