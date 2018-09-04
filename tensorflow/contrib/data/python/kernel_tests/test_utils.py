# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Test utilities for tf.data functionality."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re

from tensorflow.python.data.util import nest
from tensorflow.python.framework import errors
from tensorflow.python.platform import test


class DatasetTestBase(test.TestCase):
  """Base class for dataset tests."""

  def _assert_datasets_equal(self, dataset1, dataset2):
    # TODO(rachelim): support sparse tensor outputs
    next1 = dataset1.make_one_shot_iterator().get_next()
    next2 = dataset2.make_one_shot_iterator().get_next()
    with self.test_session() as sess:
      while True:
        try:
          op1 = sess.run(next1)
        except errors.OutOfRangeError:
          with self.assertRaises(errors.OutOfRangeError):
            sess.run(next2)
          break
        op2 = sess.run(next2)

        op1 = nest.flatten(op1)
        op2 = nest.flatten(op2)
        assert len(op1) == len(op2)
        for i in range(len(op1)):
          self.assertAllEqual(op1[i], op2[i])

  def _assert_datasets_raise_same_error(self,
                                        dataset1,
                                        dataset2,
                                        exception_class,
                                        replacements=None):
    next1 = dataset1.make_one_shot_iterator().get_next()
    next2 = dataset2.make_one_shot_iterator().get_next()
    with self.test_session() as sess:
      try:
        sess.run(next1)
        raise ValueError(
            "Expected dataset to raise an error of type %s, but it did not." %
            repr(exception_class))
      except exception_class as e:
        expected_message = e.message
        for old, new, count in replacements:
          expected_message = expected_message.replace(old, new, count)
        # Check that the first segment of the error messages are the same.
        with self.assertRaisesRegexp(exception_class,
                                     re.escape(expected_message)):
          sess.run(next2)
