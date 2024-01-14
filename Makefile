LIB_DIR := burn_vae
BUILD_DIR := model_artifacts
NOTEBOOK_DIR := notebooks
MODEL := $(BUILD_DIR)/model.mpk.gz

notebook: build
	@echo "Starting notebook..."
	@poetry run jupyter notebook --notebook-dir=$(NOTEBOOK_DIR)


build: $(MODEL) python_deps
	@echo "Building python package"
	@poetry run maturin develop --release -m $(LIB_DIR)/Cargo.toml

$(MODEL):
	@echo "Training model"
	@cargo run --release $(BUILD_DIR)

.PHONY: python_deps
python_deps:
	@poetry install --no-root -q

.PHONY: clean
clean:
	rm -rf $(BUILD_DIR)

.PHONY: fclean
fclean: clean
	cargo clean
	rm -rf .venv
