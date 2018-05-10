# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["quad"]

import os
import sysconfig
import tensorflow as tf


# Load the ops library
suffix = sysconfig.get_config_var("EXT_SUFFIX")
dirname = os.path.dirname(os.path.abspath(__file__))
libfile = os.path.join(dirname, "quad_op")
if suffix is not None:
    libfile += suffix
else:
    libfile += ".so"
quad_op = tf.load_op_library(libfile)


def quad(g1, g2, p, z):
    return quad_op.quad(g1, g2, p, z)


@tf.RegisterGradient("Quad")
def _quad_grad(op, *grads):
    g1, g2, p, z = op.inputs
    bf = grads[0]
    return quad_op.quad_rev(g1, g2, p, z, bf)
