# Copyright 2016 Google Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

""" A tiny version of `six` to help with backwards compability. Also includes
 compatibility helpers for numpy. """

import sys
import imp

PY2 = sys.version_info[0] == 2
PY26 = sys.version_info[0:2] == (2, 6)
PY27 = sys.version_info[0:2] == (2, 7)
PY275 = sys.version_info[0:3] >= (2, 7, 5)
PY3 = sys.version_info[0] == 3
PY34 = sys.version_info[0:2] >= (3, 4)

if PY3:
    string_types = (str,)
    binary_types = (bytes,bytearray)
    range_func = range
    memoryview_type = memoryview
    struct_bool_decl = "?"
else:
    string_types = (unicode,)
    if PY26 or PY27:
        binary_types = (str,bytearray)
    else:
        binary_types = (str,)
    range_func = xrange
    if PY26 or (PY27 and not PY275):
        memoryview_type = buffer
        struct_bool_decl = "<b"
    else:
        memoryview_type = memoryview
        struct_bool_decl = "?"

# Helper functions to facilitate making numpy optional instead of required

def import_numpy():
    """
    Returns the numpy module if it exists on the system,
    otherwise returns None.
    """
    try:
        imp.find_module('numpy')
        numpy_exists = True
    except ImportError:
        numpy_exists = False

    if numpy_exists:
        # We do this outside of try/except block in case numpy exists
        # but is not installed correctly. We do not want to catch an
        # incorrect installation which would manifest as an
        # ImportError.
        import numpy as np
    else:
        np = None

    return np


class NumpyRequiredForThisFeature(RuntimeError):
    """
    Error raised when user tries to use a feature that
    requires numpy without having numpy installed.
    """
    pass


# NOTE: Future Jython support may require code here (look at `six`).
