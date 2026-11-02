#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="src",
    version="0.0.1",
    description="Self-supervised pretraining of 3D ViTs on radiotherapy CBCT",
    # dependencies are listed in environment.yaml
    packages=find_packages(include=["src", "src.*"]),
)
