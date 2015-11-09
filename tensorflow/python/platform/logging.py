"""Switch between depending on pyglib.logging or regular logging."""
from __future__ import absolute_import
# pylint: disable=unused-import
# pylint: disable=g-import-not-at-top
# pylint: disable=wildcard-import
import tensorflow.python.platform
from . import control_imports
if control_imports.USE_OSS and control_imports.OSS_LOGGING:
  from tensorflow.python.platform.default._logging import *
else:
  from tensorflow.python.platform.google._logging import *
