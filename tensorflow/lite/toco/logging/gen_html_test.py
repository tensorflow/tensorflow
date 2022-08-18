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
"""Tests for gen_html.py."""

import os
import shutil

from tensorflow.lite.toco.logging import gen_html
from tensorflow.lite.toco.logging import toco_conversion_log_pb2 as _toco_conversion_log_pb2
from tensorflow.python.framework import test_util
from tensorflow.python.lib.io import file_io as _file_io
from tensorflow.python.platform import resource_loader
from tensorflow.python.platform import test


class GenHtmlTest(test_util.TensorFlowTestCase):

  def test_generate_html(self):
    toco_conversion_log_before = _toco_conversion_log_pb2.TocoConversionLog()
    toco_conversion_log_after = _toco_conversion_log_pb2.TocoConversionLog()

    toco_conversion_log_before.op_list.extend([
        "Conv1", "Conv2", "Identity", "Reshape", "Dense", "Dense", "CustomOp",
        "AvgPool3D", "Softmax"
    ])
    toco_conversion_log_before.model_size = 9

    toco_conversion_log_after.op_list.extend([
        "Conv1", "Conv2", "Dense", "Dense", "CustomOp", "AvgPool3D", "Softmax"
    ])
    toco_conversion_log_after.built_in_ops["Conv1"] = 1
    toco_conversion_log_after.built_in_ops["Conv2"] = 1
    toco_conversion_log_after.built_in_ops["Dense"] = 2
    toco_conversion_log_after.built_in_ops["Softmax"] = 1
    toco_conversion_log_after.custom_ops["CustomOp"] = 1
    toco_conversion_log_after.select_ops["AvgPool3D"] = 1
    toco_conversion_log_after.model_size = 7

    export_path = os.path.join(self.get_temp_dir(), "generated.html")
    html_generator = gen_html.HTMLGenerator(
        html_template_path=resource_loader.get_path_to_datafile(
            "template.html"),
        export_report_path=export_path)

    html_generator.generate(toco_conversion_log_before,
                            toco_conversion_log_after, True,
                            "digraph  {a -> b}", "digraph  {a -> b}", "",
                            "/path/to/flatbuffer")

    with _file_io.FileIO(export_path, "r") as f_export, _file_io.FileIO(
        resource_loader.get_path_to_datafile("testdata/generated.html"),
        "r") as f_expect:
      expected = f_expect.read()
      exported = f_export.read()
      self.assertEqual(exported, expected)

  def test_gen_conversion_log_html(self):
    # Copies all required data files into a temporary folder for testing.
    export_path = self.get_temp_dir()
    toco_log_before_path = resource_loader.get_path_to_datafile(
        "testdata/toco_log_before.pb")
    toco_log_after_path = resource_loader.get_path_to_datafile(
        "testdata/toco_log_after.pb")
    dot_before = resource_loader.get_path_to_datafile(
        "testdata/toco_tf_graph.dot")
    dot_after = resource_loader.get_path_to_datafile(
        "testdata/toco_tflite_graph.dot")
    shutil.copy(toco_log_before_path, export_path)
    shutil.copy(toco_log_after_path, export_path)
    shutil.copy(dot_before, export_path)
    shutil.copy(dot_after, export_path)

    # Generate HTML content based on files in the test folder.
    gen_html.gen_conversion_log_html(export_path, True, "/path/to/flatbuffer")

    result_html = os.path.join(export_path, "toco_conversion_summary.html")

    with _file_io.FileIO(result_html, "r") as f_export, _file_io.FileIO(
        resource_loader.get_path_to_datafile("testdata/generated.html"),
        "r") as f_expect:
      expected = f_expect.read()
      exported = f_export.read()
      self.assertEqual(exported, expected)

  def test_get_input_type_from_signature(self):
    op_signatures = [
        ("INPUT:[1,73,73,160]::float::[64,1,1,160]::float::[64]::float::"
         "OUTPUT:[1,73,73,64]::float::NAME:Conv::VERSION:1")
    ]
    expect_input_types = [
        ("shape:[1,73,73,160],type:float,shape:[64,1,1,160],type:float,"
         "shape:[64],type:float")
    ]
    for i in range(len(op_signatures)):
      self.assertEqual(
          gen_html.get_input_type_from_signature(op_signatures[i]),
          expect_input_types[i])


if __name__ == "__main__":
  test.main()
