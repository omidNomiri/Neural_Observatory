"""
Neural Observatory setup configuration.
"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="Neural_Observatory",
    version="0.1.0b1",
    author="Omid Nomiri",
    author_email="omidnomiri@gmail.com",
    description="A tool for neural network observability and diagnostics for PyTorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/omidNomiri/Neural_Observatory",
    project_urls={
        "Bug Tracker": "https://github.com/omidNomiri/Neural_Observatory/issues",
        "Documentation": "https://github.com/omidNomiri/Neural_Observatory#readme",
        "Source Code": "https://github.com/omidNomiri/Neural_Observatory",
    },
    packages=find_packages(exclude=["tests", "examples", "docs"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.9.0",
        "numpy>=1.19.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.10",
            "black>=21.0",
            "isort>=5.0",
            "flake8>=3.9",
            "mypy>=0.900",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
        ],
    },
)