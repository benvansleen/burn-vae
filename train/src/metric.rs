use burn::train::metric::{
    Metric, MetricEntry, MetricMetadata, Numeric,
    state::{FormatOptions, NumericMetricState},
};

pub struct NvidiaUtilMetric {
    state: NumericMetricState,
    has_gpu: bool,
}

impl NvidiaUtilMetric {
    pub fn new() -> Self {
        Self {
            state: NumericMetricState::new(),
            has_gpu: true,
        }
    }
}

impl Default for NvidiaUtilMetric {
    fn default() -> Self {
        Self::new()
    }
}

impl Metric for NvidiaUtilMetric {
    const NAME: &'static str = "GPU Utilization";
    type Input = ();

    fn update(
        &mut self,
        _item: &(),
        _metadata: &MetricMetadata,
    ) -> MetricEntry {
        let fmt = FormatOptions::new(Self::NAME).precision(2);
        let util = self
            .has_gpu
            .then(|| {
                std::process::Command::new("nvidia-smi")
                    .arg("--query-gpu=utilization.gpu")
                    .arg("--format=csv,noheader,nounits")
                    .output()
                    .ok()
                    .and_then(|output| {
                        String::from_utf8(output.stdout).ok()
                    })
                    .and_then(|output| output.trim().parse::<f64>().ok())
                    .unwrap_or_else(|| {
                        self.has_gpu = false;
                        0.
                    })
            })
            .unwrap_or(0.);

        self.state.update(util, 1, fmt)
    }

    fn clear(&mut self) {
        self.state.reset();
    }
}

impl Numeric for NvidiaUtilMetric {
    fn value(&self) -> f64 {
        self.state.value()
    }
}
