from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name = "Colorectal Cancer Detection Project",
    version = "0.1",
    author = "Pratik Suchak",
    packages = find_packages(),
    install_requires = requirements,
)