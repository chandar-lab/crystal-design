import os

import setuptools

with open("version.txt") as f:
    VERSION = f.read().strip()

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="crystal_design",
    version=VERSION,
    description="Research code on using designing crystal structures using RL-like algorithms",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/chandar-lab/crystal-design",
    project_urls={
        "Bug Tracker": "https://github.com/chandar-lab/crystal-design/issues",
    },
    python_requires=">=3.6",
    install_requires=[
        "numpy",
        "torch",
    ],
    packages=['crystal_design']
)
