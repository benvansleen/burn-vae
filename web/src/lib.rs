use cfg_if::cfg_if;
pub mod app;
pub mod error_template;
pub mod fileserv;

cfg_if! { if #[cfg(feature = "hydrate")] {
    pub mod bindings;
    use leptos::*;
    use wasm_bindgen::prelude::wasm_bindgen;

    #[wasm_bindgen]
    pub fn hydrate() {
        _ = console_log::init_with_level(log::Level::Debug);
        console_error_panic_hook::set_once();

        leptos::leptos_dom::HydrationCtx::stop_hydrating();
    }
}}
