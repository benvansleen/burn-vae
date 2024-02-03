use dataset::Point;
use plotly::{
    color,
    common::{Marker, Mode},
    Layout, Plot, Scatter3D,
};

type Points = Vec<Point>;

pub struct Trace {
    pts: Points,
    colors: Vec<f32>,
    name: &'static str,
}

impl Trace {
    pub fn new(pts: Points, colors: Vec<f32>, name: &'static str) -> Self {
        Self { pts, colors, name }
    }
}

pub fn plot(traces: &[Trace]) -> Plot {
    let mut plt = Plot::new();
    plt.set_layout(Layout::new().height(1000).width(1000));
    traces.iter().for_each(|t| {
        let t = trace(&t.pts, &t.colors, t.name);
        plt.add_trace(t);
    });

    plt
}

fn trace(
    pts: &Points,
    color: &[f32],
    name: &str,
) -> Box<Scatter3D<f32, f32, f32>> {
    let (mut x, mut y, mut z) = (
        Vec::with_capacity(pts.len()),
        Vec::with_capacity(pts.len()),
        Vec::with_capacity(pts.len()),
    );
    pts.iter().for_each(|pt| {
        x.push(pt[0]);
        y.push(pt[1]);
        z.push(pt[2]);
    });

    let color_fn = |c| match name {
        "generated" => color::Rgb::new(
            (c * 10.) as u8,
            (c * 128.) as u8,
            (c * 255.) as u8,
        ),
        "true" => color::Rgb::new(
            (c * 200.) as u8,
            (c * 100.) as u8,
            (c * 100.) as u8,
        ),
        _ => color::Rgb::new(
            (c * 0.) as u8,
            (c * 0.) as u8,
            (c * 200.) as u8,
        ),
    };

    let c_max: f32 = color.iter().fold(0., |acc, &x| acc.max(x));
    let color: Vec<_> =
        color.iter().map(|&c| c / c_max).map(color_fn).collect();

    Scatter3D::new(x, y, z)
        .marker(Marker::new().size(2).color_array(color))
        .mode(Mode::Markers)
        .name(name)
}
