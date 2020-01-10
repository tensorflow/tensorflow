"""Switch between depending on googletest or unittest."""
# pylint: disable=unused-import
# pylint: disable=g-import-not-at-top
# pylint: disable=wildcard-import
import tensorflow.python.platform
import control_imports
if control_imports.USE_OSS and control_imports.OSS_GOOGLETEST:
  from tensorflow.python.platform.default._googletest import *
else:
  from tensorflow.python.platform.google._googletest import *
