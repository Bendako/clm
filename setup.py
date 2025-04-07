from setuptools import setup, find_packages

setup(
    name="clm",
    version="0.1.0",
    packages=find_packages(),
    description="Continual Learning for Models",
    author="Ben",
    author_email="ben@example.com",
    install_requires=[
        "torch",
        "numpy",
        "matplotlib"
    ],
) 