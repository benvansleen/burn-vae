export async function _update(id, p) {
    const div = document.getElementById(id);
    const layout = div? div.layout : {
        scene: {
            camera: {
                up: {x: 1, y: 1, z: 1},
            },
        },
    };

    const plot = JSON.parse(p);
    Plotly.react(id, plot, {
        showlegend: false,
        ...layout,
    });
}
