# Variational Autoencoder Implementation

This project uses the Burn library to learn the the Scikit-Learn Swiss Roll function. The image below shows the true (orange) and learned (blue) distributions:

![image](https://github.com/benvansleen/burn-vae/assets/78059325/6ac56ac3-1edd-41cf-ba57-77ba95a845d5)


The VAE can be compiled to WASM and run in the browser. The `web` crate deploys the VAE via Leptos.

https://github.com/benvansleen/burn-vae/assets/78059325/71f40d8d-38b8-41e4-8b17-af26ab22e12c


The VAE is conditioned on radius from center of spiral. As a result, the trained model generates points at a provided radius (`burn_vae.generate(r, n_points)` using the python bindings).

https://github.com/benvansleen/burn-vae/assets/78059325/065c9e8c-9f1d-4bb9-acc6-f9dbe2af32e2
