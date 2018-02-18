from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf
from tensorflow.python.framework import common_shapes
from tensorflow.python.framework import errors
from tensorflow.python.framework import load_library
from tensorflow.python.framework import ops
from tensorflow.python.platform import resource_loader
from tensorflow.python.platform import tf_logging as logging

library = None

def bilinear_initializer(shape, dtype=tf.float32, partition_info=None):
  """
  Bilinear Initializer Operation
  
  Computes batch filters with bilinear weights. Result is a
  4D tensor of dimension [H, W, C, N] where:
  
  (1) H = height of kernel
  (2) W = width of kernel
  (3) C = number of input channels
  (4) N = number of output channels (number of filters)
  Typical use case is weight initialization for deconvolution layer.
  """
  return library.bilinear_initializer(shape)

ops.NotDifferentiable('BilinearInitializer')


def _load_library(name, op_list=None):
  """Loads a .so file containing the specified operators.
  Args:
    name: The name of the .so file to load.
    op_list: A list of names of operators that the library should have. If None
         then the .so file's contents will not be verified.
  Raises:
    NameError if one of the required ops is missing.
  """
  try:
    filename = resource_loader.get_path_to_datafile(name)
    library = load_library.load_op_library(filename)
    for expected_op in (op_list or []):
      for lib_op in library.OP_LIST.op:
        if lib_op.name == expected_op:
          break
        else:
          raise NameError('Could not find operator %s in dynamic library %s' %
                  (expected_op, name))
    return library
  except errors.NotFoundError:
    logging.warning('%s file could not be loaded.', name)


if os.name != 'nt':
  library = _load_library('bilinear_initializer.so', ['BilinearInitializer'])
