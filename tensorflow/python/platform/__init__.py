"""Setup system-specific platform environment for TensorFlow."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from . import control_imports
if control_imports.USE_OSS:
  from tensorflow.python.platform.default._init import *
else:
  from tensorflow.python.platform.google._init import *
