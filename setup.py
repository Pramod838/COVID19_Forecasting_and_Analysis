from setuptools import setup, find_packages

setup(
    name="disease-dynamics-analysis",
    version="1.0.0",
    description="COVID-19 disease dynamics and forecasting analysis",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "plotly>=5.15.0",
        "scikit-learn>=1.3.0",
        "xgboost>=2.0.0",
        "prophet>=1.1.4",
        "torch>=2.0.0",
        "requests>=2.31.0",
        "pyyaml>=6.0",
        "jupyter>=1.0.0",
        "optuna>=3.3.0",
        "shap>=0.42.0",
        "folium>=0.14.0",
        "tqdm>=4.65.0",
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "download-data=scripts.download_data:main",
            "train-models=scripts.train_models:main",
            "generate-report=scripts.generate_report:main",
        ]
    },
)
