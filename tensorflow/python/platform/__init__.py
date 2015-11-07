"""Setup system-specific platform environment for TensorFlow."""
import control_imports
if control_imports.USE_OSS:
  from tensorflow.python.platform.default._init import *
else:
  from tensorflow.python.platform.google._init import *
