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
"""Conversion of eager-style Python into TensorFlow graph code.

NOTE: In TensorFlow 2.0, AutoGraph is automatically applied when using
`tf.function`. This module contains lower-level APIs for advanced use.

AutoGraph transforms a subset of Python which operates on TensorFlow objects
into equivalent TensorFlow graph code. When executing the graph, it has the same
effect as if you ran the original code in eager mode.
Python code which doesn't operate on TensorFlow objects remains functionally
unchanged, but keep in mind that `tf.function` only executes such code at trace
time, and generally will not be consistent with eager execution.

For more information, see the
[AutoGraph reference documentation](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/autograph/g3doc/reference/index.md),
and the [tf.function guide](https://www.tensorflow.org/guide/function#autograph_transformations).
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# TODO(mdan): Bring only the relevant symbols to the top level.
from tensorflow.python.autograph import operators
from tensorflow.python.autograph import utils
from tensorflow.python.autograph.core.converter import ConversionOptions
from tensorflow.python.autograph.core.converter import Feature
from tensorflow.python.autograph.impl.api import AutoGraphError
from tensorflow.python.autograph.impl.api import convert
from tensorflow.python.autograph.impl.api import converted_call
from tensorflow.python.autograph.impl.api import do_not_convert
from tensorflow.python.autograph.impl.api import StackTraceMapper
from tensorflow.python.autograph.impl.api import to_code
from tensorflow.python.autograph.impl.api import to_graph
from tensorflow.python.autograph.lang.directives import set_element_type
from tensorflow.python.autograph.lang.directives import set_loop_options
from tensorflow.python.autograph.lang.special_functions import stack
from tensorflow.python.autograph.utils import ag_logging
from tensorflow.python.util.all_util import remove_undocumented

# TODO(mdan): Revisit this list once we finalize the generated code mechanism.
_allowed_symbols = [
    # Main API
    'AutoGraphError',
    'ConversionOptions',
    'Feature',
    'StackTraceMapper',
    'convert',
    'converted_call',
    'do_not_convert',
    'to_code',
    'to_graph',
    # Overloaded operators
    'operators',
    # Python language "extensions"
    'set_element_type',
    'set_loop_options',
    'stack',
    'tensor_list',
    # Utilities: to be removed
    'utils',
]

remove_undocumented(__name__, _allowed_symbols)
