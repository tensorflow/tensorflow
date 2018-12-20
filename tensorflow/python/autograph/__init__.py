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
"""Conversion of plain Python into TensorFlow graph code.

NOTE: In TensorFlow 2.0, AutoGraph is automatically applied when using
`tf.function`. This module contains lower-level APIs for advanced use.

For more information, see the
[AutoGraph guide](https://www.tensorflow.org/guide/autograph).

By equivalent graph code we mean code that generates a TensorFlow graph when
run. The generated graph has the same effects as the original code when executed
(for example with `tf.function` or `tf.compat.v1.Session.run`). In other words,
using AutoGraph can be thought of as running Python in TensorFlow.
"""
# TODO(b/119833526): Link to the new tf.function + autograph tutorial.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# TODO(mdan): Bring only the relevant symbols to the top level.
from tensorflow.python.autograph import operators
from tensorflow.python.autograph import utils
from tensorflow.python.autograph.core.converter import ConversionOptions
from tensorflow.python.autograph.core.converter import Feature
from tensorflow.python.autograph.core.converter import Verbosity
from tensorflow.python.autograph.core.errors import GraphConstructionError
from tensorflow.python.autograph.core.errors import improved_errors
from tensorflow.python.autograph.core.errors import TfRuntimeError
from tensorflow.python.autograph.impl.api import convert
from tensorflow.python.autograph.impl.api import converted_call
from tensorflow.python.autograph.impl.api import do_not_convert
from tensorflow.python.autograph.impl.api import RunMode
from tensorflow.python.autograph.impl.api import to_code
from tensorflow.python.autograph.impl.api import to_graph
from tensorflow.python.autograph.lang.directives import set_element_type
from tensorflow.python.autograph.lang.directives import set_loop_options
from tensorflow.python.autograph.lang.special_functions import stack
from tensorflow.python.autograph.lang.special_functions import tensor_list
from tensorflow.python.autograph.pyct.transformer import AutographParseError
from tensorflow.python.util.all_util import remove_undocumented

# TODO(mdan): Revisit this list once we finalize the generated code mechanism.
_allowed_symbols = [
    # Main API
    'ConversionOptions',
    'Feature',
    'RunMode',
    'convert',
    'converted_call',
    'do_not_convert',
    'to_code',
    'to_graph',
    # Overloaded operators
    'operators',
    # Errors
    'improved_errors',
    'GraphConstructionError',
    'TfRuntimeError',
    'Verbosity',
    # Python language "extensions"
    'set_element_type',
    'set_loop_options',
    'stack',
    'tensor_list',
    # Exceptions
    'AutographParseError',
    # Utilities: to be removed
    'utils',
]

remove_undocumented(__name__, _allowed_symbols)
