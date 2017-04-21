
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# go/tf-wildcard-import
# pylint: disable=wildcard-import
# from tensorflow.contrib.shuffle.python.ops.gen_shuffle_op import *
# pylint: enable=wildcard-import
from tensorflow.contrib.util import loader
from tensorflow.python.platform import resource_loader

_shuffle_op = loader.load_op_library(
    resource_loader.get_path_to_datafile('_shuffle_op.so'))
