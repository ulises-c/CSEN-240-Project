#!/bin/bash

# Simple script to create a Python virtual environment using venv and pyenv
# Tested on Ubuntu (Linux) with pyenv installed

# Usage:
# 1. chmod +x create_venv.sh
# 2. ./create_venv.sh
# Optional: Specify the virtual environment name and Python version
# 2. ./create_venv.sh -n [venv_name] -v [python_version]
# Example: ./create_venv.sh -n myenv -v 3.9

### Script Start ###

# Default virtual environment name
VENV_NAME="venv"

# Default Python version
PYTHON_VERSION="3.12"

# Parse command line arguments
while getopts "n:v:" opt; do
  case $opt in
    n) VENV_NAME=$OPTARG ;;
    v) PYTHON_VERSION=$OPTARG ;;
    *) echo "Usage: $0 [-n venv_name] [-v python_version]" >&2
       exit 1 ;;
  esac
done

# Ensure pyenv is installed
if ! command -v pyenv &> /dev/null; then
    echo "Error: pyenv is not installed. Install it first: https://github.com/pyenv/pyenv"
    exit 1
fi

# Check if the specified Python version is installed with pyenv
if ! pyenv versions --bare | grep -q "^$PYTHON_VERSION$"; then
    echo "Python $PYTHON_VERSION is not installed. Installing it via pyenv..."
    pyenv install $PYTHON_VERSION
fi

# Set the Python version globally or locally for the virtual environment
pyenv global $PYTHON_VERSION  # Or use pyenv local $PYTHON_VERSION for a specific project directory

# Ensure the desired Python version is in the path
PYTHON_PATH=$(pyenv which python)
if [ ! -x "$PYTHON_PATH" ]; then
    echo "Error: Python $PYTHON_VERSION not found after installation. Ensure it is installed correctly."
    exit 1
fi

# Create virtual environment inside the src directory
echo "Creating virtual environment: $VENV_NAME using Python $PYTHON_VERSION"
VENV_PATH="../$VENV_NAME"  # Adjust path to create the venv in the parent (src) directory
$PYTHON_PATH -m venv "$VENV_PATH"

# Activate the virtual environment
source "$VENV_PATH/bin/activate"

# Upgrade essential tools
pip install --upgrade pip setuptools wheel

# Install dependencies if requirements.txt exists
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    echo "No requirements.txt found. Skipping dependencies installation."
fi

# Sanity check: Verify the Python version in the virtual environment
echo "Virtual environment '$VENV_NAME' is ready with Python $(python --version) using pyenv."
