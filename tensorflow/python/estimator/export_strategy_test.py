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
"""Tests for `make_export_strategy`."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tempfile
import time

from tensorflow.python.estimator import estimator as estimator_lib
from tensorflow.python.estimator import export_strategy as export_strategy_lib
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import gfile
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import compat


class ExportStrategyTest(test.TestCase):

  def testAcceptsNameAndFn(self):
    def export_fn(estimator, export_path):
      del estimator, export_path

    export_strategy = export_strategy_lib.ExportStrategy(
        name="test", export_fn=export_fn)

    self.assertEqual("test", export_strategy.name)
    self.assertEqual(export_fn, export_strategy.export_fn)

  def testCallsExportFnThatDoesntKnowExtraArguments(self):
    expected_estimator = {}

    def export_fn(estimator, export_path):
      self.assertEqual(expected_estimator, estimator)
      self.assertEqual("expected_path", export_path)

    export_strategy = export_strategy_lib.ExportStrategy(
        name="test", export_fn=export_fn)

    export_strategy.export(
        estimator=expected_estimator, export_path="expected_path")

    # Also works with additional arguments that `export_fn` doesn't support.
    # The lack of support is detected and the arguments aren't passed.
    export_strategy.export(
        estimator=expected_estimator,
        export_path="expected_path",
        checkpoint_path="unexpected_checkpoint_path")
    export_strategy.export(
        estimator=expected_estimator,
        export_path="expected_path",
        eval_result=())
    export_strategy.export(
        estimator=expected_estimator,
        export_path="expected_path",
        checkpoint_path="unexpected_checkpoint_path",
        eval_result=())

  def testCallsExportFnThatKnowsAboutCheckpointPathButItsNotGiven(self):
    expected_estimator = {}

    def export_fn(estimator, export_path, checkpoint_path):
      self.assertEqual(expected_estimator, estimator)
      self.assertEqual("expected_path", export_path)
      self.assertEqual(None, checkpoint_path)

    export_strategy = export_strategy_lib.ExportStrategy(
        name="test", export_fn=export_fn)

    export_strategy.export(
        estimator=expected_estimator, export_path="expected_path")
    export_strategy.export(
        estimator=expected_estimator,
        export_path="expected_path",
        eval_result=())

  def testCallsExportFnWithCheckpointPath(self):
    expected_estimator = {}

    def export_fn(estimator, export_path, checkpoint_path):
      self.assertEqual(expected_estimator, estimator)
      self.assertEqual("expected_path", export_path)
      self.assertEqual("expected_checkpoint_path", checkpoint_path)

    export_strategy = export_strategy_lib.ExportStrategy(
        name="test", export_fn=export_fn)

    export_strategy.export(
        estimator=expected_estimator,
        export_path="expected_path",
        checkpoint_path="expected_checkpoint_path")
    export_strategy.export(
        estimator=expected_estimator,
        export_path="expected_path",
        checkpoint_path="expected_checkpoint_path",
        eval_result=())

  def testCallsExportFnThatKnowsAboutEvalResultButItsNotGiven(self):
    expected_estimator = {}

    def export_fn(estimator, export_path, checkpoint_path, eval_result):
      self.assertEqual(expected_estimator, estimator)
      self.assertEqual("expected_path", export_path)
      self.assertEqual(None, checkpoint_path)
      self.assertEqual(None, eval_result)

    export_strategy = export_strategy_lib.ExportStrategy(
        name="test", export_fn=export_fn)

    export_strategy.export(
        estimator=expected_estimator, export_path="expected_path")

  def testCallsExportFnThatAcceptsEvalResultButNotCheckpoint(self):
    expected_estimator = {}

    def export_fn(estimator, export_path, eval_result):
      del estimator, export_path, eval_result
      raise RuntimeError("Should raise ValueError before this.")

    export_strategy = export_strategy_lib.ExportStrategy(
        name="test", export_fn=export_fn)

    expected_error_message = (
        "An export_fn accepting eval_result must also accept checkpoint_path")

    with self.assertRaisesRegexp(ValueError, expected_error_message):
      export_strategy.export(
          estimator=expected_estimator, export_path="expected_path")

    with self.assertRaisesRegexp(ValueError, expected_error_message):
      export_strategy.export(
          estimator=expected_estimator,
          export_path="expected_path",
          checkpoint_path="unexpected_checkpoint_path")

    with self.assertRaisesRegexp(ValueError, expected_error_message):
      export_strategy.export(
          estimator=expected_estimator,
          export_path="expected_path",
          eval_result=())

    with self.assertRaisesRegexp(ValueError, expected_error_message):
      export_strategy.export(
          estimator=expected_estimator,
          export_path="expected_path",
          checkpoint_path="unexpected_checkpoint_path",
          eval_result=())

  def testCallsExportFnWithEvalResultAndCheckpointPath(self):
    expected_estimator = {}
    expected_eval_result = {}

    def export_fn(estimator, export_path, checkpoint_path, eval_result):
      self.assertEqual(expected_estimator, estimator)
      self.assertEqual("expected_path", export_path)
      self.assertEqual("expected_checkpoint_path", checkpoint_path)
      self.assertEqual(expected_eval_result, eval_result)

    export_strategy = export_strategy_lib.ExportStrategy(
        name="test", export_fn=export_fn)

    export_strategy.export(
        estimator=expected_estimator,
        export_path="expected_path",
        checkpoint_path="expected_checkpoint_path",
        eval_result=expected_eval_result)


class MakeExportStrategyTest(test.TestCase):

  def test_make_export_strategy(self):
    def _serving_input_fn():
      return array_ops.constant([1]), None

    export_strategy = export_strategy_lib.make_export_strategy(
        serving_input_fn=_serving_input_fn,
        assets_extra={"from/path": "to/path"},
        as_text=False,
        exports_to_keep=5)
    self.assertTrue(
        isinstance(export_strategy, export_strategy_lib.ExportStrategy))

  def test_garbage_collect_exports(self):
    export_dir_base = tempfile.mkdtemp() + "export/"
    gfile.MkDir(export_dir_base)
    export_dir_1 = _create_test_export_dir(export_dir_base)
    export_dir_2 = _create_test_export_dir(export_dir_base)
    export_dir_3 = _create_test_export_dir(export_dir_base)
    export_dir_4 = _create_test_export_dir(export_dir_base)

    self.assertTrue(gfile.Exists(export_dir_1))
    self.assertTrue(gfile.Exists(export_dir_2))
    self.assertTrue(gfile.Exists(export_dir_3))
    self.assertTrue(gfile.Exists(export_dir_4))

    def _serving_input_fn():
      return array_ops.constant([1]), None
    export_strategy = export_strategy_lib.make_export_strategy(
        _serving_input_fn, exports_to_keep=2)
    estimator = test.mock.Mock(spec=estimator_lib.Estimator)
    # Garbage collect all but the most recent 2 exports,
    # where recency is determined based on the timestamp directory names.
    export_strategy.export(estimator, export_dir_base)

    self.assertFalse(gfile.Exists(export_dir_1))
    self.assertFalse(gfile.Exists(export_dir_2))
    self.assertTrue(gfile.Exists(export_dir_3))
    self.assertTrue(gfile.Exists(export_dir_4))


def _create_test_export_dir(export_dir_base):
  export_dir = _get_timestamped_export_dir(export_dir_base)
  gfile.MkDir(export_dir)
  time.sleep(2)
  return export_dir


def _get_timestamped_export_dir(export_dir_base):
  # When we create a timestamped directory, there is a small chance that the
  # directory already exists because another worker is also writing exports.
  # In this case we just wait one second to get a new timestamp and try again.
  # If this fails several times in a row, then something is seriously wrong.
  max_directory_creation_attempts = 10

  attempts = 0
  while attempts < max_directory_creation_attempts:
    export_timestamp = int(time.time())

    export_dir = os.path.join(
        compat.as_bytes(export_dir_base),
        compat.as_bytes(str(export_timestamp)))
    if not gfile.Exists(export_dir):
      # Collisions are still possible (though extremely unlikely): this
      # directory is not actually created yet, but it will be almost
      # instantly on return from this function.
      return export_dir
    time.sleep(1)
    attempts += 1
    logging.warn("Export directory {} already exists; retrying (attempt {}/{})".
                 format(export_dir, attempts, max_directory_creation_attempts))
  raise RuntimeError("Failed to obtain a unique export directory name after "
                     "{} attempts.".format(max_directory_creation_attempts))


if __name__ == "__main__":
  test.main()
