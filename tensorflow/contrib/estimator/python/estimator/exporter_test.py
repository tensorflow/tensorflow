# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for `StepsExporter`."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil
import tempfile

from tensorflow.contrib.estimator.python.estimator import exporter as exporter_lib
from tensorflow.python.estimator import estimator as estimator_lib
from tensorflow.python.platform import gfile
from tensorflow.python.platform import test


class StepsExporterTest(test.TestCase):

  def test_error_out_if_steps_to_keep_has_no_positive_integers(self):

    def _serving_input_receiver_fn():
      pass

    with self.assertRaisesRegexp(ValueError, "positive integer"):
      exporter = exporter_lib.StepsExporter(
          name="specified_steps_exporter",
          serving_input_receiver_fn=_serving_input_receiver_fn,
          steps_to_keep=[-1, 0, 1.1])
      self.assertEqual("specified_steps_exporter", exporter.name)

  def test_steps_exporter(self):

    def _serving_input_receiver_fn():
      pass

    export_dir_base = tempfile.mkdtemp()
    gfile.MkDir(export_dir_base)
    gfile.MkDir(export_dir_base + "/export")
    gfile.MkDir(export_dir_base + "/eval")

    exporter = exporter_lib.StepsExporter(
        name="steps_exporter",
        serving_input_receiver_fn=_serving_input_receiver_fn,
        assets_extra={"from/path": "to/path"},
        as_text=False,
        steps_to_keep=[1])
    estimator = test.mock.Mock(spec=estimator_lib.Estimator)
    estimator.export_savedmodel.return_value = "export_result_path"
    estimator.model_dir = export_dir_base

    export_result = exporter.export(estimator, export_dir_base,
                                    "checkpoint_path", {"global_step": 1},
                                    False)

    self.assertEqual("export_result_path", export_result)
    estimator.export_savedmodel.assert_called_with(
        export_dir_base,
        _serving_input_receiver_fn,
        assets_extra={"from/path": "to/path"},
        as_text=False,
        checkpoint_path="checkpoint_path",
        strip_default_attrs=True)

    shutil.rmtree(export_dir_base, ignore_errors=True)

  def test_steps_exporter_with_preemption(self):

    def _serving_input_receiver_fn():
      pass

    export_dir_base = tempfile.mkdtemp()
    gfile.MkDir(export_dir_base)
    gfile.MkDir(export_dir_base + "/export")
    gfile.MkDir(export_dir_base + "/eval")

    eval_dir_base = os.path.join(export_dir_base, "eval_continuous")
    estimator_lib._write_dict_to_summary(eval_dir_base, {}, 1)
    estimator_lib._write_dict_to_summary(eval_dir_base, {}, 2)

    exporter = exporter_lib.StepsExporter(
        name="steps_exporter",
        serving_input_receiver_fn=_serving_input_receiver_fn,
        event_file_pattern="eval_continuous/*.tfevents.*",
        assets_extra={"from/path": "to/path"},
        as_text=False,
        steps_to_keep=[1, 2, 6, 8])

    estimator = test.mock.Mock(spec=estimator_lib.Estimator)
    estimator.model_dir = export_dir_base
    estimator.export_savedmodel.return_value = "export_result_path"

    export_result = exporter.export(estimator, export_dir_base,
                                    "checkpoint_path", {"global_step": 3},
                                    False)
    self.assertEqual(None, export_result)

    export_result = exporter.export(estimator, export_dir_base,
                                    "checkpoint_path", {"global_step": 6},
                                    False)
    self.assertEqual("export_result_path", export_result)

    export_result = exporter.export(estimator, export_dir_base,
                                    "checkpoint_path", {"global_step": 7},
                                    False)
    self.assertEqual(None, export_result)

    shutil.rmtree(export_dir_base, ignore_errors=True)

  def test_specified_step_is_saved(self):

    def _serving_input_receiver_fn():
      pass

    export_dir_base = tempfile.mkdtemp()
    gfile.MkDir(export_dir_base)
    gfile.MkDir(export_dir_base + "/export")
    gfile.MkDir(export_dir_base + "/eval")

    exporter = exporter_lib.StepsExporter(
        name="steps_exporter",
        serving_input_receiver_fn=_serving_input_receiver_fn,
        assets_extra={"from/path": "to/path"},
        as_text=False,
        steps_to_keep=[1, 5, 8, 10, 11])
    estimator = test.mock.Mock(spec=estimator_lib.Estimator)
    estimator.export_savedmodel.return_value = "export_result_path"
    estimator.model_dir = export_dir_base

    export_result = exporter.export(estimator, export_dir_base,
                                    "checkpoint_path", {"global_step": 1},
                                    False)

    self.assertTrue(estimator.export_savedmodel.called)
    self.assertEqual("export_result_path", export_result)

    export_result = exporter.export(estimator, export_dir_base,
                                    "checkpoint_path", {"global_step": 2},
                                    False)
    self.assertEqual(None, export_result)

    export_result = exporter.export(estimator, export_dir_base,
                                    "checkpoint_path", {"global_step": 5},
                                    False)
    self.assertTrue(estimator.export_savedmodel.called)
    self.assertEqual("export_result_path", export_result)

    export_result = exporter.export(estimator, export_dir_base,
                                    "checkpoint_path", {"global_step": 10},
                                    False)
    self.assertTrue(estimator.export_savedmodel.called)
    self.assertEqual("export_result_path", export_result)

    export_result = exporter.export(estimator, export_dir_base,
                                    "checkpoint_path", {"global_step": 15},
                                    False)
    self.assertTrue(estimator.export_savedmodel.called)
    self.assertEqual("export_result_path", export_result)

    export_result = exporter.export(estimator, export_dir_base,
                                    "checkpoint_path", {"global_step": 20},
                                    False)
    self.assertEqual(None, export_result)

    shutil.rmtree(export_dir_base, ignore_errors=True)

  def test_steps_exporter_with_no_global_step_key(self):

    def _serving_input_receiver_fn():
      pass

    export_dir_base = tempfile.mkdtemp()
    gfile.MkDir(export_dir_base)
    gfile.MkDir(export_dir_base + "/export")
    gfile.MkDir(export_dir_base + "/eval")

    exporter = exporter_lib.StepsExporter(
        name="steps_exporter",
        serving_input_receiver_fn=_serving_input_receiver_fn,
        assets_extra={"from/path": "to/path"},
        as_text=False,
        steps_to_keep=[1])
    estimator = test.mock.Mock(spec=estimator_lib.Estimator)
    estimator.export_savedmodel.return_value = "export_result_path"
    estimator.model_dir = export_dir_base

    with self.assertRaisesRegexp(ValueError, "does not have global step"):
      exporter.export(estimator, export_dir_base, "checkpoint_path", {}, False)

    shutil.rmtree(export_dir_base, ignore_errors=True)


if __name__ == "__main__":
  test.main()
