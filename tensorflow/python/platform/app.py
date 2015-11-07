"""Switch between depending on pyglib.app or an OSS replacement."""
# pylint: disable=unused-import
# pylint: disable=g-import-not-at-top
# pylint: disable=wildcard-import
import tensorflow.python.platform
import control_imports
if control_imports.USE_OSS and control_imports.OSS_APP:
  from tensorflow.python.platform.default._app import *
else:
  from tensorflow.python.platform.google._app import *

# Import 'flags' into this module
from tensorflow.python.platform import flags
