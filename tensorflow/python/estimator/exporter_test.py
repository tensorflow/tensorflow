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
"""Tests for `Exporter`s."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tempfile
import time

from tensorflow.python.estimator import estimator as estimator_lib
from tensorflow.python.estimator import exporter as exporter_lib
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import gfile
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import compat


class LatestExporterTest(test.TestCase):

  def test_error_out_if_exports_to_keep_is_zero(self):
    def _serving_input_receiver_fn():
      pass

    with self.assertRaisesRegexp(ValueError, "positive number"):
      exporter = exporter_lib.LatestExporter(
          name="latest_exporter",
          serving_input_receiver_fn=_serving_input_receiver_fn,
          exports_to_keep=0)
      self.assertEqual("latest_exporter", exporter.name)

  def test_latest_exporter(self):

    def _serving_input_receiver_fn():
      pass

    export_dir_base = tempfile.mkdtemp() + "export/"
    gfile.MkDir(export_dir_base)

    exporter = exporter_lib.LatestExporter(
        name="latest_exporter",
        serving_input_receiver_fn=_serving_input_receiver_fn,
        assets_extra={"from/path": "to/path"},
        as_text=False,
        exports_to_keep=5)
    estimator = test.mock.Mock(spec=estimator_lib.Estimator)
    estimator.export_savedmodel.return_value = "export_result_path"

    export_result = exporter.export(estimator, export_dir_base,
                                    "checkpoint_path", {}, False)

    self.assertEqual("export_result_path", export_result)
    estimator.export_savedmodel.assert_called_with(
        export_dir_base,
        _serving_input_receiver_fn,
        assets_extra={"from/path": "to/path"},
        as_text=False,
        checkpoint_path="checkpoint_path",
        strip_default_attrs=True)

  def test_only_the_last_export_is_saved(self):

    def _serving_input_receiver_fn():
      pass

    export_dir_base = tempfile.mkdtemp() + "export/"
    gfile.MkDir(export_dir_base)

    exporter = exporter_lib.FinalExporter(
        name="latest_exporter",
        serving_input_receiver_fn=_serving_input_receiver_fn,
        assets_extra={"from/path": "to/path"},
        as_text=False)
    estimator = test.mock.Mock(spec=estimator_lib.Estimator)
    estimator.export_savedmodel.return_value = "export_result_path"

    export_result = exporter.export(estimator, export_dir_base,
                                    "checkpoint_path", {}, False)

    self.assertFalse(estimator.export_savedmodel.called)
    self.assertEqual(None, export_result)

    export_result = exporter.export(estimator, export_dir_base,
                                    "checkpoint_path", {}, True)

    self.assertEqual("export_result_path", export_result)
    estimator.export_savedmodel.assert_called_with(
        export_dir_base,
        _serving_input_receiver_fn,
        assets_extra={"from/path": "to/path"},
        as_text=False,
        checkpoint_path="checkpoint_path",
        strip_default_attrs=True)

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

    def _serving_input_receiver_fn():
      return array_ops.constant([1]), None

    exporter = exporter_lib.LatestExporter(
        name="latest_exporter",
        serving_input_receiver_fn=_serving_input_receiver_fn,
        exports_to_keep=2)
    estimator = test.mock.Mock(spec=estimator_lib.Estimator)
    # Garbage collect all but the most recent 2 exports,
    # where recency is determined based on the timestamp directory names.
    exporter.export(estimator, export_dir_base, None, None, False)

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
        compat.as_bytes(export_dir_base), compat.as_bytes(
            str(export_timestamp)))
    if not gfile.Exists(export_dir):
      # Collisions are still possible (though extremely unlikely): this
      # directory is not actually created yet, but it will be almost
      # instantly on return from this function.
      return export_dir
    time.sleep(1)
    attempts += 1
    logging.warn(
        "Export directory {} already exists; retrying (attempt {}/{})".format(
            export_dir, attempts, max_directory_creation_attempts))
  raise RuntimeError("Failed to obtain a unique export directory name after "
                     "{} attempts.".format(max_directory_creation_attempts))


if __name__ == "__main__":
  test.main()
