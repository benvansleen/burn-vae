use plotly::{
    color,
    common::{Marker, Mode},
    Layout, Plot, Scatter3D,
};

type Points = Vec<Vec<f32>>;
pub fn plot(pts: (&Points, &Points), color: (&[f32], &[f32])) {
    let (label, pred) = pts;
    let (label_color, pred_color) = color;

    let labels = trace(label, label_color, false);
    let preds = trace(pred, pred_color, true);

    let mut plot = Plot::new();
    plot.set_layout(Layout::new().height(1000).width(1000));
    plot.add_trace(labels);
    plot.add_trace(preds);
    plot.show();
}

fn trace(
    pts: &Points,
    color: &[f32],
    generated: bool,
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

    let color_fn = |c| {
        if generated {
            color::Rgb::new(
                (c * 255.) as u8,
                (c * 128.) as u8,
                0,
            )
        } else {
            color::Rgb::new(
                0,
                (c * 128.) as u8,
                (c * 255.) as u8,
            )
        }
    };

    let c_max: f32 = color.iter().fold(0., |acc, &x| acc.max(x));
    let color: Vec<_> =
        color.iter().map(|&c| c / c_max).map(color_fn).collect();

    Scatter3D::new(x, y, z)
        .marker(Marker::new().size(2).color_array(color))
        .mode(Mode::Markers)
}
