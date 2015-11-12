"""Switch between depending on pyglib.gfile or an OSS replacement."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=unused-import
# pylint: disable=g-import-not-at-top
# pylint: disable=wildcard-import
import tensorflow.python.platform
from . import control_imports
if control_imports.USE_OSS and control_imports.OSS_GFILE:
  from tensorflow.python.platform.default._gfile import *
else:
  from tensorflow.python.platform.google._gfile import *
