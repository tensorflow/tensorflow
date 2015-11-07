"""Switch between depending on pyglib.gfile or an OSS replacement."""
# pylint: disable=unused-import
# pylint: disable=g-import-not-at-top
# pylint: disable=wildcard-import
import tensorflow.python.platform
import control_imports
if control_imports.USE_OSS and control_imports.OSS_GFILE:
  from tensorflow.python.platform.default._gfile import *
else:
  from tensorflow.python.platform.google._gfile import *
