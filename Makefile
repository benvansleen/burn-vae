LIB_DIR := inference
BUILD_DIR := model_artifacts
NOTEBOOK_DIR := notebooks
PYTHON_DIR := python/burn_vae
MODEL := $(BUILD_DIR)/model.mpk.gz

.PHONY: notebook
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
	@echo "Installing python dependencies"
	@poetry install --no-root -q

.PHONY: clean
clean:
	rm -rf $(BUILD_DIR)

.PHONY: fclean
fclean: clean
	cargo clean
	rm -rf .venv
	rm -rf $(PYTHON_DIR)/__pycache__
	rm $(PYTHON_DIR)/*.so
