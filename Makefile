# Makefile for setting up Python environment using pyenv

# Usage:
# make setup
# make clean
# make activate

# Define paths and variables
VENV_NAME = py311
PYTHON_VERSION = 3.11
PYENV = pyenv
INSTALL = pip install -r requirements.txt
PY_FILE = knee-osteo.py
ZIP_FILE = Knee_Osteoarthritis_Classification_Camino.zip
EXTRACT_DIR = images/

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

extract-images:
	@echo "Extracting $(ZIP_FILE) to $(EXTRACT_DIR)..."
	@mkdir -p $(EXTRACT_DIR)
	@unzip -n $(ZIP_FILE) -d $(EXTRACT_DIR)

activate:
	@echo "To activate your virtual environment, run:"
	@echo "source $(VENV_NAME)/bin/activate"

clean:
	rm -rf $(VENV_NAME)

full-clean:
	rm -rf $(VENV_NAME)
	rm -rf $(EXTRACT_DIR)

setup: install extract-images
	@echo "Python environment setup is complete, and images have been extracted!"

run: setup
	@echo "Running $(PY_FILE), using Python version: $$(pyenv which python)"  # Check which Python is being used
	$(VENV_NAME)/bin/python knee-osteo.py
