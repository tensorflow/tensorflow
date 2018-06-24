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
"""Tests for slices module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.autograph import utils
from tensorflow.contrib.autograph.converters import converter_test_base
from tensorflow.contrib.autograph.converters import slices
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import list_ops
from tensorflow.python.platform import test


class SliceTest(converter_test_base.TestCase):

  def test_index_access(self):

    def test_fn(l):
      utils.set_element_type(l, dtypes.int32)
      return l[1]

    node = self.parse_and_analyze(
        test_fn,
        {
            'utils': utils,
            'dtypes': dtypes
        },
        include_type_analysis=True,
    )
    node = slices.transform(node, self.ctx)

    with self.compiled(node, dtypes.int32) as result:
      result.utils = utils
      result.dtypes = dtypes
      with self.test_session() as sess:
        tl = list_ops.tensor_list_from_tensor(
            [1, 2], element_shape=constant_op.constant([], dtype=dtypes.int32))
        y = result.test_fn(tl)
        self.assertEqual(2, sess.run(y))


if __name__ == '__main__':
  test.main()
