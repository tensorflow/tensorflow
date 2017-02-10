"""Fact user op Python library."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path

import tensorflow as tf

_fact_module = tf.load_op_library(os.path.join(tf.resource_loader.get_data_files_path(), 'fact.so'))
fact = _fact_module.fact
