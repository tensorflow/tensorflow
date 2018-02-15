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
"""Tests for the experimental input pipeline ops."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.data.python.ops import get_single_element
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test


class GetSingleElementTest(test.TestCase):

  def testGetSingleElement(self):
    skip_value = array_ops.placeholder(dtypes.int64, shape=[])
    take_value = array_ops.placeholder_with_default(
        constant_op.constant(1, dtype=dtypes.int64), shape=[])

    dataset = (dataset_ops.Dataset.range(100)
               .skip(skip_value)
               .map(lambda x: x * x)
               .take(take_value))

    element = get_single_element.get_single_element(dataset)

    with self.test_session() as sess:
      self.assertEqual(0, sess.run(element, feed_dict={skip_value: 0}))
      self.assertEqual(25, sess.run(element, feed_dict={skip_value: 5}))
      self.assertEqual(100, sess.run(element, feed_dict={skip_value: 10}))

      with self.assertRaisesRegexp(errors.InvalidArgumentError,
                                   "Dataset was empty."):
        sess.run(element, feed_dict={skip_value: 100})

      with self.assertRaisesRegexp(errors.InvalidArgumentError,
                                   "Dataset had more than one element."):
        sess.run(element, feed_dict={skip_value: 0, take_value: 2})


if __name__ == "__main__":
  test.main()
