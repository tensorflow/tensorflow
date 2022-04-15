# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Keras injection tests."""

import tensorflow as tf


# This test is used for OSS since the Keras package import is different
# between internal g3 and external OSS. We need to make sure when TensorFlow
# is imported, Keras should be loaded as well (not just lazy load). Some of the
# Keras code load should be triggered so that it will inject proper
# functionality like registering the optimizer class for SavedModel.
class KerasInjectionTest(tf.test.TestCase):

  def test_keras_optimizer_injected(self):
    # Make sure keras optimizers are registed without accessing keras code
    self.assertIn('optimizer',
                  tf.__internal__.saved_model.load.registered_identifiers())


if __name__ == '__main__':
  tf.test.main()
