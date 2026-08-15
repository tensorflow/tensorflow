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
    """Improvement over best always resets wait, even if below baseline."""
    # baseline=0.3: improvements that don't beat baseline still reset wait.
    stopper = callbacks.EarlyStopping(
        monitor='val_loss', patience=0, baseline=0.3)
    stopper.model = _MockModel()
    stopper.on_train_begin()

    # Loss improves from inf to 0.5 — improvement over best resets wait to 0.
    stopper.on_epoch_end(0, logs={'val_loss': 0.5})
    self.assertFalse(stopper.model.stop_training)

    # Loss improves from 0.5 to 0.4 — improvement resets wait to 0 again.
    stopper.on_epoch_end(1, logs={'val_loss': 0.4})
    self.assertFalse(stopper.model.stop_training)


class EarlyStoppingPatienceGreaterThanZeroTest(test_lib.TestCase):
  """Tests for patience > 0 with mixed improvement/non-improvement sequences."""

  def testPatienceTwoStopsAfterTwoNonImprovements(self):
    """patience=2 stops after 2 consecutive non-improving epochs."""
    stopper = callbacks.EarlyStopping(monitor='val_loss', patience=2)
    stopper.model = _MockModel()
    stopper.on_train_begin()

    # Epoch 0: initial improvement (inf -> 0.5)
    stopper.on_epoch_end(0, logs={'val_loss': 0.5})
    self.assertFalse(stopper.model.stop_training)

    # Epoch 1: no improvement (0.5 -> 0.5), wait=1
    stopper.on_epoch_end(1, logs={'val_loss': 0.5})
    self.assertFalse(stopper.model.stop_training)

    # Epoch 2: no improvement (0.5 -> 0.6 worse), wait=2 >= patience=2 -> stop
    stopper.on_epoch_end(2, logs={'val_loss': 0.6})
    self.assertTrue(stopper.model.stop_training)

  def testPatienceTwoResetsOnImprovement(self):
    """patience=2: improvement resets wait counter."""
    stopper = callbacks.EarlyStopping(monitor='val_loss', patience=2)
    stopper.model = _MockModel()
    stopper.on_train_begin()

    # Epoch 0: initial improvement (inf -> 0.5), wait=0
    stopper.on_epoch_end(0, logs={'val_loss': 0.5})
    self.assertFalse(stopper.model.stop_training)

    # Epoch 1: no improvement (0.5 -> 0.5), wait=1
    stopper.on_epoch_end(1, logs={'val_loss': 0.5})
    self.assertFalse(stopper.model.stop_training)

    # Epoch 2: improvement (0.5 -> 0.4), wait resets to 0
    stopper.on_epoch_end(2, logs={'val_loss': 0.4})
    self.assertFalse(stopper.model.stop_training)

    # Epoch 3: no improvement (0.4 -> 0.5 worse), wait=1
    stopper.on_epoch_end(3, logs={'val_loss': 0.5})
    self.assertFalse(stopper.model.stop_training)

    # Epoch 4: no improvement (0.5 -> 0.5), wait=2 >= patience=2 -> stop
    stopper.on_epoch_end(4, logs={'val_loss': 0.5})
    self.assertTrue(stopper.model.stop_training)

  def testPatienceThreeImprovementBelowBaselineResetsWait(self):
    """Improvement below baseline still resets wait; patience counts properly."""
    stopper = callbacks.EarlyStopping(
        monitor='val_loss', patience=3, baseline=0.1)
    stopper.model = _MockModel()
    stopper.on_train_begin()

    # Epoch 0: improvement (inf -> 0.5), wait=0 (below baseline but improved)
    stopper.on_epoch_end(0, logs={'val_loss': 0.5})
    self.assertFalse(stopper.model.stop_training)

    # Epoch 1: no improvement (0.5 -> 0.5), wait=1
    stopper.on_epoch_end(1, logs={'val_loss': 0.5})
    self.assertFalse(stopper.model.stop_training)

    # Epoch 2: improvement (0.5 -> 0.4), wait resets to 0
    stopper.on_epoch_end(2, logs={'val_loss': 0.4})
    self.assertFalse(stopper.model.stop_training)

    # Epoch 3: no improvement (0.4 -> 0.5), wait=1
    stopper.on_epoch_end(3, logs={'val_loss': 0.5})
    self.assertFalse(stopper.model.stop_training)

    # Epoch 4: no improvement (0.5 -> 0.5), wait=2
    stopper.on_epoch_end(4, logs={'val_loss': 0.5})
    self.assertFalse(stopper.model.stop_training)

    # Epoch 5: no improvement, wait=3 >= patience=3 -> stop
    stopper.on_epoch_end(5, logs={'val_loss': 0.6})
    self.assertTrue(stopper.model.stop_training)


if __name__ == '__main__':
  test_lib.main()
