# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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

# Lint as: python3
"""Tests for input_pipeline_proto_to_gviz."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import io
import enum

import gviz_api
# pylint: disable=g-importing-member
from google.protobuf.any_pb2 import Any
# pylint: enable=g-importing-member

# pylint: disable=g-direct-tensorflow-import
from tensorflow.core.profiler.protobuf import hardware_types_pb2
from tensorflow.core.profiler.protobuf import input_pipeline_pb2
from tensorflow.python.platform import test
from tensorflow.python.profiler import input_pipeline_proto_to_gviz
# pylint: enable=g-direct-tensorflow-import


class MockValues(enum.IntEnum):
  STEP_NUMBER = 1
  STEP_TIME_MS = 2
  UNKNOWN_TIME_MS = 3
  HOST_WAIT_INPUT_MS = 11
  HOST_TO_DEVICE_MS = 12
  OUTPUT_MS = 5
  DEVICE_COMPUTE_MS = 6
  DEVICE_TO_DEVICE_MS = 7
  HOST_COMPUTE_MS = 8
  HOST_PREPARE_MS = 9
  HOST_COMPILE_MS = 10


class ProtoToGvizTest(test.TestCase):

  def create_empty_input_pipeline(self):
    return input_pipeline_pb2.InputPipelineAnalysisResult()

  def create_mock_step_summary(self, base):
    step_summary = input_pipeline_pb2.StepSummary()
    step_summary.average = 1 + base
    step_summary.standard_deviation = 2 + base
    step_summary.minimum = 3 + base
    step_summary.maximum = 4 + base
    return step_summary

  def create_mock_input_pipeline(self):
    ipa = input_pipeline_pb2.InputPipelineAnalysisResult()
    ipa.hardware_type = hardware_types_pb2.HardwareType.CPU_ONLY
    ipa.step_time_summary.CopyFrom(self.create_mock_step_summary(10))
    ipa.input_percent_summary.CopyFrom(self.create_mock_step_summary(20))

    # Add 3 rows
    for _ in range(0, 3):
      step_details = input_pipeline_pb2.PerGenericStepDetails()
      step_details.step_number = MockValues.STEP_NUMBER
      step_details.step_time_ms = MockValues.STEP_TIME_MS
      step_details.unknown_time_ms = MockValues.UNKNOWN_TIME_MS
      step_details.host_wait_input_ms = MockValues.HOST_WAIT_INPUT_MS
      step_details.host_to_device_ms = MockValues.HOST_TO_DEVICE_MS
      step_details.output_ms = MockValues.OUTPUT_MS
      step_details.device_compute_ms = MockValues.DEVICE_COMPUTE_MS
      step_details.device_to_device_ms = MockValues.DEVICE_TO_DEVICE_MS
      step_details.host_compute_ms = MockValues.HOST_COMPUTE_MS
      step_details.host_prepare_ms = MockValues.HOST_PREPARE_MS
      step_details.host_compile_ms = MockValues.HOST_COMPILE_MS

      step_details_any = Any()
      step_details_any.Pack(step_details)
      ipa.step_details.append(step_details_any)

    input_time_breakdown = input_pipeline_pb2.InputTimeBreakdown()
    input_time_breakdown.demanded_file_read_us = 1
    input_time_breakdown.advanced_file_read_us = 2
    input_time_breakdown.preprocessing_us = 3
    input_time_breakdown.enqueue_us = 4
    input_time_breakdown.unclassified_non_enqueue_us = 5
    ipa.input_time_breakdown.CopyFrom(input_time_breakdown)

    for _ in range(0, 3):
      input_op_details = input_pipeline_pb2.InputOpDetails()
      input_op_details.op_name = str(1)
      input_op_details.count = 2
      input_op_details.time_in_ms = 3
      input_op_details.time_in_percent = 4
      input_op_details.self_time_in_ms = 5
      input_op_details.self_time_in_percent = 6
      input_op_details.category = str(7)
      ipa.input_op_details.append(input_op_details)

    recommendation = input_pipeline_pb2.InputPipelineAnalysisRecommendation()
    for ss in ["a", "b", "c", "d", "e"]:
      recommendation.details.append(ss)
    ipa.recommendation.CopyFrom(recommendation)

    step_time_breakdown = input_pipeline_pb2.GenericStepTimeBreakdown()
    step_time_breakdown.unknown_time_ms_summary.CopyFrom(
        self.create_mock_step_summary(1))
    step_time_breakdown.host_wait_input_ms_summary.CopyFrom(
        self.create_mock_step_summary(9))
    step_time_breakdown.host_to_device_ms_summary.CopyFrom(
        self.create_mock_step_summary(10))
    step_time_breakdown.input_ms_summary.CopyFrom(
        self.create_mock_step_summary(11))
    step_time_breakdown.output_ms_summary.CopyFrom(
        self.create_mock_step_summary(3))
    step_time_breakdown.device_compute_ms_summary.CopyFrom(
        self.create_mock_step_summary(4))
    step_time_breakdown.device_to_device_ms_summary.CopyFrom(
        self.create_mock_step_summary(5))
    step_time_breakdown.host_compute_ms_summary.CopyFrom(
        self.create_mock_step_summary(6))
    step_time_breakdown.host_prepare_ms_summary.CopyFrom(
        self.create_mock_step_summary(7))
    step_time_breakdown.host_compile_ms_summary.CopyFrom(
        self.create_mock_step_summary(8))

    step_time_breakdown_any = Any()
    step_time_breakdown_any.Pack(step_time_breakdown)
    ipa.step_time_breakdown.CopyFrom(step_time_breakdown_any)

    return ipa

  def test_input_pipeline_empty(self):
    ipa = self.create_empty_input_pipeline()
    data_table = input_pipeline_proto_to_gviz.generate_chart_table(ipa)

    self.assertEqual(0, data_table.NumberOfRows(),
                     "Empty table should have 0 rows.")
    # Input pipeline chart data table has 9 columns.
    self.assertLen(data_table.columns, 9)

  def test_input_pipeline_simple(self):
    ipa = self.create_mock_input_pipeline()
    (table_description, data,
     custom_properties) = input_pipeline_proto_to_gviz.get_chart_table_args(ipa)
    data_table = gviz_api.DataTable(table_description, data, custom_properties)

    # Data is a list of 3 rows.
    self.assertLen(data, 3)
    self.assertEqual(3, data_table.NumberOfRows(), "Simple table has 3 rows.")
    # Table descriptor is a list of 9 columns.
    self.assertLen(table_description, 9)
    # DataTable also has 9 columns.
    self.assertLen(data_table.columns, 9)

    csv_file = io.StringIO(data_table.ToCsv())
    reader = csv.reader(csv_file)

    expected = [
        str(int(MockValues.STEP_NUMBER)),
        int(MockValues.DEVICE_COMPUTE_MS),
        int(MockValues.DEVICE_TO_DEVICE_MS),
        int(MockValues.HOST_COMPUTE_MS),
        int(MockValues.HOST_PREPARE_MS),
        int(MockValues.HOST_WAIT_INPUT_MS) + int(MockValues.HOST_TO_DEVICE_MS),
        int(MockValues.OUTPUT_MS),
        int(MockValues.HOST_COMPILE_MS),
        int(MockValues.UNKNOWN_TIME_MS),
    ]

    for (rr, row_values) in enumerate(reader):
      if rr == 0:
        # DataTable columns match schema defined in table_description.
        for (cc, column_header) in enumerate(row_values):
          self.assertEqual(table_description[cc][2], column_header)
      else:
        for (cc, cell_str) in enumerate(row_values):
          raw_value = data[rr - 1][cc]
          value_type = table_description[cc][1]

          # Only number and strings are used in our DataTable schema.
          self.assertIn(value_type, ["number", "string"])

          # Encode in similar fashion as DataTable.ToCsv().
          expected_value = gviz_api.DataTable.CoerceValue(raw_value, value_type)
          self.assertNotIsInstance(expected_value, tuple)
          self.assertEqual(expected_value, raw_value)
          self.assertEqual(str(expected_value), cell_str)

          # Check against expected values we have set in our mock table.
          if isinstance(expected[cc], str):
            self.assertEqual(expected[cc], cell_str)
          else:
            self.assertEqual(str(float(expected[cc])), cell_str)


if __name__ == "__main__":
  test.main()
