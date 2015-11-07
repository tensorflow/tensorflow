# pylint: disable=wildcard-import,unused-import,g-bad-import-order,line-too-long
"""Import core names of TensorFlow.

Programs that want to build Brain Ops and Graphs without having to import the
constructors and utilities individually can import this file:

import tensorflow.python.platform
import tensorflow as tf

"""

import tensorflow.python.platform
from tensorflow.core.framework.graph_pb2 import *
from tensorflow.core.framework.summary_pb2 import *
from tensorflow.core.framework.config_pb2 import *
from tensorflow.core.util.event_pb2 import *

# Framework
from tensorflow.python.framework.framework_lib import *

# Session
from tensorflow.python.client.client_lib import *

# Ops
from tensorflow.python.ops.standard_ops import *

# Bring nn, image_ops, user_ops as a subpackages
from tensorflow.python.ops import nn
from tensorflow.python.ops import image_ops as image
from tensorflow.python.user_ops import user_ops

# Import the names from python/training.py as train.Name.
from tensorflow.python.training import training as train

# Sub-package for performing i/o directly instead of via ops in a graph.
from tensorflow.python.lib.io import python_io

# Make some application and test modules available.
from tensorflow.python.platform import app
from tensorflow.python.platform import flags
from tensorflow.python.platform import logging
from tensorflow.python.platform import test
