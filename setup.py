#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from setuptools import setup, Extension

import numpy
import pybind11

ext_modules = [
    Extension(
        "dr25.quad",
        [os.path.join("dr25", "quad.cpp")],
        include_dirs=[
            pybind11.get_include(False),
            pybind11.get_include(True),
            numpy.get_include(),
        ],
        language="c++",
        extra_compile_args=["-std=c++14", "-stdlib=libc++"],
    ),
]

setup(
    name="dr25",
    version="0.0.0",
    author="Dan Foreman-Mackey",
    ext_modules=ext_modules,
    install_requires=["pybind11", "numpy"],
    zip_safe=False,
)
