# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Autograph compiles Python code into equivalent TensorFlow code.

Equivalent here means that they have the same effect when executed.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# TODO(mdan): Bring only the relevant symbols to the top level.
from tensorflow.contrib.autograph import utils
from tensorflow.contrib.autograph import operators
from tensorflow.contrib.autograph.impl.api import convert
from tensorflow.contrib.autograph.impl.api import converted_call
from tensorflow.contrib.autograph.impl.api import do_not_convert
from tensorflow.contrib.autograph.impl.api import RunMode
from tensorflow.contrib.autograph.impl.api import to_code
from tensorflow.contrib.autograph.impl.api import to_graph
from tensorflow.contrib.autograph.pyct.transformer import AutographParseError
from tensorflow.python.util.all_util import remove_undocumented

_allowed_symbols = [
    # Main API
    'RunMode',
    'convert',
    'converted_call',
    'do_not_convert',
    'to_code',
    'to_graph',
    # Special functions and overloaded operators
    'operators',
    'stack',
    # Exceptions
    'AutographParseError',
    # Utilities: to be removed
    'utils',
]

remove_undocumented(__name__, _allowed_symbols)
