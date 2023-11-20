.PHONY: install-python-package setup clean

PYTHON := python3.10

install-python-package:
	@echo "Installing requirements python package"
	@$(PYTHON) -m venv .venv
	@.venv/bin/pip install -r requirements.txt
	@.venv/bin/pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
	@.venv/bin/pip install torchsummary==1.5.1 pytorch_model_summary==0.2.4

clean:
	@echo "Cleaning up"
	@rm -rf .venv

# Add a target for setup that depends on installing Python packages
setup: install-python-package
