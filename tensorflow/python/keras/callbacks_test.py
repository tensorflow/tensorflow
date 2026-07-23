# Copyright 2026 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for Keras callbacks validation."""

from tensorflow.python.keras import callbacks
from tensorflow.python.platform import test as test_lib


class _MockModel:
  """Minimal mock model for testing EarlyStopping on_epoch_end."""

  def __init__(self):
    self.stop_training = False
    self.weights = [0.0]

  def get_weights(self):
    return list(self.weights)

  def set_weights(self, w):
    self.weights = list(w)


class EarlyStoppingValidationTest(test_lib.TestCase):

  def testPatienceNegative(self):
    with self.assertRaisesRegex(ValueError, r'patience.*must be >= 0'):
      callbacks.EarlyStopping(patience=-1)

  def testPatienceZero(self):
    callbacks.EarlyStopping(patience=0)

  def testPatiencePositive(self):
    callbacks.EarlyStopping(patience=3)


class EarlyStoppingOnEpochEndTest(test_lib.TestCase):
  """Tests for on_epoch_end behavior, especially patience=0."""

  def testPatienceZeroContinuesOnImprovement(self):
    """patience=0 should NOT stop training when the monitored value improves."""
    stopper = callbacks.EarlyStopping(monitor='val_loss', patience=0)
    stopper.model = _MockModel()
    stopper.on_train_begin()

    # First call: improvement (loss goes down)
    stopper.on_epoch_end(0, logs={'val_loss': 0.5})
    self.assertFalse(stopper.model.stop_training)

    # Second call: further improvement
    stopper.on_epoch_end(1, logs={'val_loss': 0.4})
    self.assertFalse(stopper.model.stop_training)

  def testPatienceZeroStopsOnNoImprovement(self):
    """patience=0 should stop training when the monitored value does NOT improve."""
    stopper = callbacks.EarlyStopping(monitor='val_loss', patience=0)
    stopper.model = _MockModel()
    stopper.on_train_begin()

    # First call: initial value (improvement over +inf for mode='min')
    stopper.on_epoch_end(0, logs={'val_loss': 0.5})
    self.assertFalse(stopper.model.stop_training)

    # Second call: same value, no improvement -> should stop
    stopper.on_epoch_end(1, logs={'val_loss': 0.5})
    self.assertTrue(stopper.model.stop_training)

  def testPatienceZeroStopsOnWorseValue(self):
    """patience=0 should stop when the monitored value gets worse."""
    stopper = callbacks.EarlyStopping(monitor='val_loss', patience=0)
    stopper.model = _MockModel()
    stopper.on_train_begin()

    # First call: initial value
    stopper.on_epoch_end(0, logs={'val_loss': 0.5})
    self.assertFalse(stopper.model.stop_training)

    # Second call: worse value (loss increased) -> should stop
    stopper.on_epoch_end(1, logs={'val_loss': 0.6})
    self.assertTrue(stopper.model.stop_training)

  def testPatienceZeroImprovementOverBestButNotBaseline(self):
    """Improvement over best but NOT baseline: wait increments, training continues."""
    # baseline=0.3 means only values < 0.3 reset the wait counter.
    stopper = callbacks.EarlyStopping(
        monitor='val_loss', patience=0, baseline=0.3)
    stopper.model = _MockModel()
    stopper.on_train_begin()

    # Loss improves from inf to 0.5, but 0.5 is NOT better than baseline 0.3
    # so wait is NOT reset (stays at 1 after increment).
    # However, the return prevents checking wait >= patience.
    stopper.on_epoch_end(0, logs={'val_loss': 0.5})
    self.assertFalse(stopper.model.stop_training)

    # Loss improves from 0.5 to 0.4, but 0.4 is still NOT better than 0.3.
    # wait increments (from 1 to 2) + return = no stop.
    stopper.on_epoch_end(1, logs={'val_loss': 0.4})
    self.assertFalse(stopper.model.stop_training)


if __name__ == '__main__':
  test_lib.main()
