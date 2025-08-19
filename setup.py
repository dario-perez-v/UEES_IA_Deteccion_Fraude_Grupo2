from setuptools import setup, find_packages

setup(
    name="ProyectoFraude",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "matplotlib",
        "seaborn",
        "scikit-learn",
        "xgboost",
        "lightgbm",
        "joblib",
        "shap"
    ]
)