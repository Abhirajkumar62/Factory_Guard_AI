from setuptools import setup, find_packages

setup(
    name="factory_guard_ai",
    version="0.1.0",
    description="Factory Guard AI - Advanced monitoring and anomaly detection system",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "xgboost>=2.0.0",
        "lightgbm>=4.0.0",
        "tensorflow>=2.14.0",
        "mlflow>=2.10.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.10.0",
            "flake8>=6.1.0",
            "jupyter>=1.0.0",
        ],
        "spark": [
            "pyspark>=3.5.0",
        ],
    },
)
