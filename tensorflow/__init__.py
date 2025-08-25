#  checing of python-version-compatibility 
import sys

# If an older TF wheel is installed on a newer Python (e.g. TF 2.7.1 + Python 3.12),
# the native import fails with a cryptic DLL error. Fail early with a clear message.
_UNSUPPORTED_PYTHON_MIN = (3, 12) 

if sys.version_info >= _UNSUPPORTED_PYTHON_MIN:
    ver = ".".join(map(str, sys.version_info[:3]))
    raise ImportError(
        f"Failed to import the native TensorFlow runtime because this TensorFlow "
        f"release is incompatible with Python {ver}.\n\n"
        "Common causes:\n"
        "- You installed a TensorFlow wheel built for a different Python version.\n"
        "- This TensorFlow version does not yet support your Python.\n\n"
        "Suggested fixes:\n"
        "- Create/use a virtual environment with a supported Python (e.g., Python 3.8/3.9 for older TF releases),\n"
        "- or upgrade TensorFlow to a release that supports your Python.\n\n"
        "See https://www.tensorflow.org/install/errors for more help and the compatibility matrix."
    )
# --- end: python-version-compatibility-check ---

from tensorflow.python import pywrap_tensorflow  # pylint: disable=unused-import

from tensorflow.python.platform import flags  # pylint: disable=g-import-not-at-top
from tensorflow.python.platform import app  # pylint: disable=g-import-not-at-top
app.flags = flags

# These symbols appear because we import the python package which
# in turn imports from tensorflow.core and tensorflow.python. They
# must come from this module. So python adds these symbols for the
# resolution to succeed.
# pylint: disable=undefined-variable
del python
del core
# pylint: enable=undefined-variable
