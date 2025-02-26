#!/bin/bash

# Simple script to create a Python virtual environment using venv

# Not yet tested on Ubuntu, but should work with minor adjustments

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

# Ensure Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python is not installed. Please install Python $PYTHON_VERSION."
    exit 1
fi

# Ensure the specified Python version is installed
if ! python3 -c "import sys; assert sys.version_info >= ('$PYTHON_VERSION')" &> /dev/null; then
    echo "Error: Python $PYTHON_VERSION is not installed. Please install Python $PYTHON_VERSION."
    exit 1
fi

# Create virtual environment inside the src directory
echo "Creating virtual environment: $VENV_NAME using Python $PYTHON_VERSION"
VENV_PATH="../$VENV_NAME"  # Adjust path to create the venv in the parent (src) directory
python$PYTHON_VERSION -m venv "$VENV_PATH"

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
echo "Virtual environment '$VENV_NAME' is ready with Python $(python --version) on Ubuntu."
