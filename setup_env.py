import yaml
import subprocess
import sys
import re

# Mapping from Conda names to PyPI names
CONDA_TO_PIP_MAPPING = {
    "fastai": "fastai",
    "scikit-learn": "scikit-learn",
    "gensim": "gensim",
    "transformers": "transformers",
    "huggingface_hub": "huggingface_hub",
    "datasets": "datasets",
    "hydra-core": "hydra-core",
}

# Packages to skip on Kaggle (already preinstalled)
SKIP_PACKAGES = ["pytorch", "torch", "torchvision", "torchaudio", "python"]

def install(package):
    """Install a Python package via pip."""
    print(f"Installing {package} ...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def conda_to_pip(pkg):
    """
    Convert conda-style package to pip-style and map names.
    Returns None if package should be skipped.
    """
    # Skip packages explicitly
    for skip in SKIP_PACKAGES:
        if pkg.startswith(skip):
            return None

    # Replace single '=' with '==', leave >= or <= unchanged
    pkg = re.sub(r"(?<![<>])=(?!=)", "==", pkg)

    # Split name and version
    name_version = pkg.split("==")
    name = name_version[0].strip()
    version = name_version[1].strip() if len(name_version) > 1 else None

    # Map package name if needed
    pip_name = CONDA_TO_PIP_MAPPING.get(name, name)

    return f"{pip_name}=={version}" if version else pip_name

def main():
    # Upgrade pip first
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])

    # Load environment.yaml
    env_file = "environment.yaml"
    with open(env_file, "r") as f:
        env = yaml.safe_load(f)

    dependencies = env.get("dependencies", [])
    pip_packages = []

    # Convert Conda dependencies to pip
    for dep in dependencies:
        if isinstance(dep, dict) and "pip" in dep:
            pip_packages.extend(dep["pip"])
        elif isinstance(dep, str):
            pip_pkg = conda_to_pip(dep)
            if pip_pkg:
                pip_packages.append(pip_pkg)

    # Install all packages
    for pkg in pip_packages:
        install(pkg)

    print("\n✅ All packages installed successfully!")

if __name__ == "__main__":
    main()
