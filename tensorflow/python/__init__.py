# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# pylint: disable=wildcard-import,unused-import,g-bad-import-order,line-too-long
"""Import core names of TensorFlow.

Programs that want to build Brain Ops and Graphs without having to import the
constructors and utilities individually can import this file:

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.python.platform
import tensorflow as tf

"""

import inspect
import traceback

try:
  # pylint: disable=g-import-not-at-top
  import tensorflow.python.platform
  from tensorflow.core.framework.graph_pb2 import *
except ImportError:
  msg = """%s\n\nError importing tensorflow.  Unless you are using bazel,
you should not try to import tensorflow from its source directory;
please exit the tensorflow source tree, and relaunch your python interpreter
from there.""" % traceback.format_exc()
  raise ImportError(msg)

from tensorflow.core.framework.summary_pb2 import *
from tensorflow.core.framework.config_pb2 import *
from tensorflow.core.util.event_pb2 import *

# Framework
from tensorflow.python.framework.framework_lib import *
from tensorflow.python.framework.versions import *
from tensorflow.python.framework import errors

# Session
from tensorflow.python.client.client_lib import *

# Ops
from tensorflow.python.ops.standard_ops import *

# Bring nn, image_ops, user_ops, compat as a subpackages
from tensorflow.python.ops import nn
from tensorflow.python.ops import image_ops as image
from tensorflow.python.user_ops import user_ops
from tensorflow.python.util import compat

# Import the names from python/training.py as train.Name.
from tensorflow.python.training import training as train

# Sub-package for performing i/o directly instead of via ops in a graph.
from tensorflow.python.lib.io import python_io

# Make some application and test modules available.
from tensorflow.python.platform import app
from tensorflow.python.platform import flags
from tensorflow.python.platform import logging
from tensorflow.python.platform import test

# Don't export modules except for the few we really want
_whitelist = set([app, compat, errors, flags, image, logging, nn,
                  python_io, test, train, user_ops])
# TODO(b/25561952): tf.ops and tf.tensor_util are DEPRECATED.  Please avoid.
_whitelist.update([ops, tensor_util])  # pylint: disable=undefined-variable
__all__ = [name for name, x in locals().items() if not name.startswith('_') and
           (not inspect.ismodule(x) or x in _whitelist)]
__all__.append('__version__')
