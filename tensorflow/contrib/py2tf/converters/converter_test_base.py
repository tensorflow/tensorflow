# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Base class for tests in this module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.py2tf.pyct import context
from tensorflow.contrib.py2tf.pyct import parser
from tensorflow.contrib.py2tf.pyct.static_analysis import access
from tensorflow.contrib.py2tf.pyct.static_analysis import live_values
from tensorflow.contrib.py2tf.pyct.static_analysis import type_info
from tensorflow.python.platform import test


class TestCase(test.TestCase):

  def parse_and_analyze(self,
                        test_fn,
                        namespace,
                        arg_types=None,
                        include_type_analysis=True):
    ctx = context.EntityContext(
        namer=None,
        source_code=None,
        source_file=None,
        namespace=namespace,
        arg_values=None,
        arg_types=arg_types)
    node = parser.parse_object(test_fn)
    node = access.resolve(node)
    node = live_values.resolve(node, namespace, {})
    if include_type_analysis:
      node = type_info.resolve(node, ctx)
    return node
