"""Load a file resource and return the contents."""
from __future__ import absolute_import
# pylint: disable=unused-import
# pylint: disable=g-import-not-at-top
# pylint: disable=wildcard-import
from . import control_imports
import tensorflow.python.platform
if control_imports.USE_OSS:
  from tensorflow.python.platform.default._resource_loader import *
else:
  from tensorflow.python.platform.google._resource_loader import *
