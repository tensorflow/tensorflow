# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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
"""Python test mode enabler.

Enables test mode for tf.data.

The test mode can be used to set up custom values for features and
experiments as required in the unit tests.

For example, if `warm_start` feature needs to be enabled exclusively for the
unit tests, the tests can enable the test mode using `toggle_test_mode` and
the default value of `warm_start` can be set as per the value of `TEST_MODE`.
"""

TEST_MODE = False


def toggle_test_mode(test_mode):
  global TEST_MODE
  TEST_MODE = test_mode
