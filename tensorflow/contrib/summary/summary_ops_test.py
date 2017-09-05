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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tempfile

from tensorflow.contrib.summary import summary_ops
from tensorflow.python.eager import test
from tensorflow.python.framework import test_util
from tensorflow.python.platform import gfile
from tensorflow.python.training import training_util


class TargetTest(test_util.TensorFlowTestCase):

  def testShouldRecordSummary(self):
    self.assertFalse(summary_ops.should_record_summaries().numpy())
    summary_ops.always_record_summaries()
    self.assertTrue(summary_ops.should_record_summaries().numpy())

  def testSummaryOps(self):
    training_util.get_or_create_global_step()
    logdir = tempfile.mkdtemp()
    summary_ops.create_summary_file_writer(logdir, max_queue=0)
    summary_ops.always_record_summaries()
    summary_ops.generic('tensor', 1, '')
    summary_ops.scalar('scalar', 2.0)
    summary_ops.histogram('histogram', [1.0])
    summary_ops.image('image', [[[[1.0]]]])
    summary_ops.audio('audio', [[1.0]], 1.0, 1)
    # The working condition of the ops is tested in the C++ test so we just
    # test here that we're calling them correctly.
    self.assertTrue(gfile.Exists(logdir))


if __name__ == '__main__':
  test.main()
