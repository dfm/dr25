#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from setuptools import setup, Extension

import numpy
import pybind11
import tensorflow as tf

args = ["-O2", "-std=c++14", "-stdlib=libc++"]
ext_modules = [
    Extension(
        "dr25.quad",
        [os.path.join("dr25", "quad.cc")],
        include_dirs=[
            pybind11.get_include(False),
            pybind11.get_include(True),
            numpy.get_include(),
            "/usr/local/include/eigen3",
            "dr25",
        ],
        language="c++",
        extra_compile_args=args,
    ),
    Extension(
        "dr25.ops",
        [os.path.join("dr25", "quad_op.cc"),
         os.path.join("dr25", "quad_rev_op.cc"),
         os.path.join("dr25", "interp_op.cc")],
        include_dirs=["dr25", ],
        language="c++",
        extra_compile_args=args+tf.sysconfig.get_compile_flags(),
        extra_link_args=tf.sysconfig.get_link_flags(),
    ),
]

setup(
    name="dr25",
    version="0.0.0",
    author="Dan Foreman-Mackey",
    ext_modules=ext_modules,
    install_requires=["tensorflow", "pybind11", "numpy"],
    zip_safe=False,
)
