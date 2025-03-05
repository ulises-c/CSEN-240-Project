# Makefile for setting up Python environment using pyenv

# Usage:
# make setup
# make clean
# make activate

# Define paths and variables
VENV_NAME = py311
PYTHON_VERSION = 3.11
PYENV = pyenv
INSTALL = pip install -r helper_tools/requirements.txt
PY_FILE = knee-osteo.py

install-python:
	@echo "Installing Python $(PYTHON_VERSION) using pyenv..."
	$(PYENV) install -s $(PYTHON_VERSION)
	$(PYENV) local $(PYTHON_VERSION)

create-venv: install-python
	@echo "Using Python version: $$(pyenv which python)"  # Check which Python is being used
	@echo "Creating virtual environment in $(VENV_NAME)..."
	$(PYENV) exec python3 -m venv $(VENV_NAME)  # Create the venv inside the venvs/ directory

install: create-venv
	@echo "Installing dependencies..."
	$(VENV_NAME)/bin/pip install --upgrade pip
	$(VENV_NAME)/bin/$(INSTALL)

activate:
	@echo "To activate your virtual environment, run:"
	@echo "source $(VENV_NAME)/bin/activate"

clean:
	rm -rf $(VENV_NAME)

setup: install
	@echo "Python environment setup is complete!"

run: 
	@echo "Running $(PY_FILE), using Python version: $$(pyenv which python)"  # Check which Python is being used
	$(VENV_NAME)/bin/python knee-osteo.py