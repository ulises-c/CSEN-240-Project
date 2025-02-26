"""
Helper script for installing required Python packages from requirements.txt.

Handles extra index URLs and ignores commented or empty lines.
"""

import subprocess
import sys
from typing import List, Tuple


def install_packages(packages: List[str], extra_index_url: str = None) -> None:
    """
    Install a list of packages using pip.

    Args:
        packages (list): List of package names to install.
        extra_index_url (str, optional): Additional package index URL.
    """
    if not packages:
        print("No packages to install.")
        return

    pip_command = [sys.executable, "-m", "pip", "install"]

    # Add extra index URL if provided
    if extra_index_url:
        pip_command += ["--extra-index-url", extra_index_url]

    # Install all packages in one command for efficiency
    pip_command += packages

    try:
        subprocess.check_call(pip_command)
        print(f"Successfully installed: {', '.join(packages)}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to install packages: {e}")


def parse_requirements(file_path: str) -> Tuple[List[str], str | None]:
    """
    Parses a requirements.txt file to extract package names and extra index URLs.

    Args:
        file_path (str): Path to the requirements file.

    Returns:
        tuple: (List of packages, extra index URL or None)
    """
    packages = []
    extra_index_url: str | None = None

    with open(file_path, "r") as file:
        for line in file:
            line = line.strip()

            if not line or line.startswith("#"):  # Skip empty and commented lines
                continue
            elif line.startswith("--extra-index-url"):  # Capture extra index URL
                parts = line.split()
                if len(parts) > 1:
                    extra_index_url = parts[1]
            elif not line.startswith("--"):  # Ignore other pip options
                packages.append(line)

    return packages, extra_index_url


def main() -> None:
    """
    Main function to install packages from requirements.txt.
    """
    requirements_file = "helper_tools/requirements.txt"

    # Parse the requirements file
    packages, extra_index_url = parse_requirements(requirements_file)

    print(f"Installing the following packages: {packages}")
    if extra_index_url:
        print(f"Using extra index URL: {extra_index_url}")

    # Install packages with optional extra index URL
    install_packages(packages, extra_index_url)


if __name__ == "__main__":
    main()
