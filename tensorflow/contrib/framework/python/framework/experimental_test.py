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
"""@experimental tests."""

# pylint: disable=unused-import
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.framework.python.framework import experimental
from tensorflow.python.platform import tf_logging as logging


class ExperimentalTest(tf.test.TestCase):

  @tf.test.mock.patch.object(logging, "warning", autospec=True)
  def test_warning(self, mock_warning):
    @experimental
    def _fn(arg0, arg1):
      """fn doc.

      Args:
        arg0: Arg 0.
        arg1: Arg 1.

      Returns:
        Sum of args.
      """
      return arg0 + arg1

    # Assert function docs are properly updated.
    self.assertEqual("_fn", _fn.__name__)
    self.assertEqual(
        "fn doc. (experimental)"
        "\n"
        "\nTHIS FUNCTION IS EXPERIMENTAL. It may change or "
        "be removed at any time, and without warning."
        "\n"
        "\n"
        "\nArgs:"
        "\n  arg0: Arg 0."
        "\n  arg1: Arg 1."
        "\n"
        "\nReturns:"
        "\n  Sum of args.", _fn.__doc__)

    # Assert calling new fn issues log warning.
    self.assertEqual(3, _fn(1, 2))
    self.assertEqual(1, mock_warning.call_count)
    (args, _) = mock_warning.call_args
    self.assertRegexpMatches(args[0], r"is experimental and may change")


if __name__ == "__main__":
  tf.test.main()
