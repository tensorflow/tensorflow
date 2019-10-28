# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for `multi_process_runner` for non-initialization."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.distribute import multi_process_runner
from tensorflow.python.distribute.multi_process_runner import MultiProcessRunner
from tensorflow.python.eager import test


class MultiProcessRunnerNoInitTest(test.TestCase):

  def test_stdout_captured(self):

    def simple_func():
      return 'foobar'

    job_count_dict = {'worker': 1}
    with self.assertRaisesRegexp(RuntimeError,
                                 '`multi_process_runner` is not initialized.'):
      MultiProcessRunner().run(
          simple_func,
          multi_process_runner.job_count_to_cluster_spec(job_count_dict))


if __name__ == '__main__':
  # Intentionally not using `multi_process_runner.test_main()` so the error
  # would occur.
  test.main()
