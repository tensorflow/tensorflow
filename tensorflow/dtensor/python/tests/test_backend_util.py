# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
"""Utility to set up DTensor backend in tests."""

from tensorflow.dtensor.python import accelerator_util
from tensorflow.python.platform import test as tf_test


class DTensorTestBackendConfigurator:
  """Configurate test backends."""

  def __init__(self, test_case: tf_test.TestCase):
    self._test_case = test_case
    # TODO(b/260771689): Refactor common backend set up logic to here.

  def tearDown(self):
    # Only need to explicitly shuts down TPU system in TFRT since in current
    # runtime, the shutdown is done in initialization process.
    if accelerator_util.is_initialized():
      accelerator_util.shutdown_accelerator_system()
