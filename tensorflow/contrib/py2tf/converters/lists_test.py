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
"""Tests for lists module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.py2tf import utils
from tensorflow.contrib.py2tf.converters import converter_test_base
from tensorflow.contrib.py2tf.converters import lists
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.platform import test


class ListTest(converter_test_base.TestCase):

  def test_empty_annotated_list(self):

    def test_fn():
      l = []
      utils.set_element_type(l, dtypes.int32)
      l.append(1)
      return l

    node = self.parse_and_analyze(test_fn, {'dtypes': dtypes, 'utils': utils})
    node = lists.transform(node, self.ctx)

    with self.compiled(node, tensor_array_ops.TensorArray,
                       dtypes.int32) as result:
      # TODO(mdan): Attach these additional modules automatically.
      result.utils = utils
      result.dtypes = dtypes
      with self.test_session() as sess:
        self.assertEqual(test_fn(), sess.run(result.test_fn().stack()))


if __name__ == '__main__':
  test.main()
