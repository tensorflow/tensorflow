"""This file is only retained for backwards compatibility.
It will be removed in the future.  sre was moved to re in version 2.5.
"""

import warnings
warnings.warn("The sre module is deprecated, please import re.",
              DeprecationWarning, 2)

from re import *
from re import __all__

# old pickles expect the _compile() reconstructor in this module
from re import _compile
