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
"""Tests for the gcs_config_ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.cloud.python.ops import gcs_config_ops
from tensorflow.python.platform import test


class GcsConfigOpsTest(test.TestCase):

  def testSetBlockCache(self):
    cfg = gcs_config_ops.BlockCacheParams(max_bytes=1024*1024*1024)
    with self.test_session() as sess:
      gcs_config_ops.configure_gcs(sess, block_cache=cfg)

  def testConfigureGcsHook(self):
    creds = {'client_id': 'fake_client',
             'refresh_token': 'fake_token',
             'client_secret': 'fake_secret',
             'type': 'authorized_user'}
    hook = gcs_config_ops.ConfigureGcsHook(credentials=creds)
    hook.begin()
    with self.test_session() as sess:
      sess.run = lambda _, feed_dict=None, options=None, run_metadata=None: None
      hook.after_create_session(sess, None)

if __name__ == '__main__':
  test.main()
