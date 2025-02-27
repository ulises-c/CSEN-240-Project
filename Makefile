# Makefile for setting up Python environment using pyenv

# Usage:
# make setup
# make clean
# make activate

# Define paths and variables
VENV_NAME = venv
VENV_DIR = $(VENV_NAME)
PYTHON_VERSION = 3.12
PYENV = pyenv
INSTALL = pip install -r helper_tools/requirements.txt

install-python:
	@echo "Installing Python $(PYTHON_VERSION) using pyenv..."
	$(PYENV) install -s $(PYTHON_VERSION)

create-venv: install-python
	@echo "Creating virtual environment in $(VENV_DIR)..."
	PYENV_VERSION=$(PYTHON_VERSION) python -m venv $(VENV_DIR)

install: create-venv
	@echo "Installing dependencies..."
	$(VENV_DIR)/bin/pip install --upgrade pip
	$(VENV_DIR)/bin/$(INSTALL)

activate:
	@echo "To activate your virtual environment, run:"
	@echo "source $(VENV_DIR)/bin/activate"

clean:
	rm -rf $(VENV_DIR)

setup: install
	@echo "Python environment setup is complete!"

