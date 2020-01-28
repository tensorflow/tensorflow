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
"""Tests for overview_page_proto_to_gviz."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import csv
import io

import gviz_api

# pylint: disable=g-direct-tensorflow-import
from tensorflow.core.profiler.protobuf import op_stats_pb2
from tensorflow.core.profiler.protobuf import overview_page_pb2
from tensorflow.python.platform import test
from tensorflow.python.profiler import overview_page_proto_to_gviz
# pylint: enable=g-direct-tensorflow-import


class ProtoToGvizTest(test.TestCase):

  @classmethod
  def setUpClass(cls):
    super(ProtoToGvizTest, cls).setUpClass()

    ProtoToGvizTest.run_env_num_columns = 4
    MockRunEnvironment = collections.namedtuple(  # pylint: disable=invalid-name
        "MockRunEnvironment", [
            "host_id", "command_line", "start_time", "bns_address",
            "host_count", "task_count", "device_type", "device_core_count"
        ])

    ProtoToGvizTest.mock_run_env = MockRunEnvironment(
        # Columns
        host_id="1",
        command_line="2",
        start_time=1582202096,
        bns_address="4",
        # Custom properties
        host_count=1,
        task_count=10,
        device_type="GPU",
        device_core_count=20,
    )

    MockOverviewTfOp = collections.namedtuple(  # pylint: disable=invalid-name
        "MockOverviewTfOp", [
            "self_time_fraction",
            "cumulative_time_fraction",
            "category",
            "name",
            "flop_rate",
        ])

    ProtoToGvizTest.mock_tf_op = MockOverviewTfOp(
        self_time_fraction=3.0,
        cumulative_time_fraction=4.0,
        category="2",
        name="1",
        flop_rate=5.0,
    )

    MockTip = collections.namedtuple(  # pylint: disable=invalid-name
        "MockTip", [
            "tip_type",
            "link",
        ])

    ProtoToGvizTest.mock_tips = []
    for tip in ["faq", "host", "device", "doc"]:
      for idx in range(0, 3):
        ProtoToGvizTest.mock_tips.append(MockTip(tip, tip + "_link" + str(idx)))

  # Checks that DataTable columns match schema defined in table_description.
  def check_header_row(self, data, table_description, row_values):
    for (cc, column_header) in enumerate(row_values):
      self.assertEqual(table_description[cc][2], column_header)

  # Checks that DataTable row value representation matches number or string.
  def check_row_types(self, data, table_description, row_values, row_idx):
    for (cc, cell_str) in enumerate(row_values):
      raw_value = data[row_idx - 1][cc]
      value_type = table_description[cc][1]

      # Only number and strings are used in our DataTable schema.
      self.assertIn(value_type, ["number", "string"])

      # Encode in similar fashion as DataTable.ToCsv().
      expected_value = gviz_api.DataTable.CoerceValue(raw_value, value_type)
      self.assertNotIsInstance(expected_value, tuple)
      self.assertEqual(expected_value, raw_value)
      self.assertEqual(str(expected_value), cell_str)

  def create_empty_run_environment(self):
    return op_stats_pb2.RunEnvironment()

  def create_empty_overview_page_analysis(self):
    return overview_page_pb2.OverviewPageAnalysis()

  def create_empty_recommendation(self):
    return overview_page_pb2.OverviewPageRecommendation()

  def create_mock_run_environment(self):
    run_env = op_stats_pb2.RunEnvironment()

    # Add 3 rows
    for _ in range(0, 3):
      job = op_stats_pb2.HostDependentJobInfoResult()
      job.host_id = self.mock_run_env.host_id
      job.command_line = self.mock_run_env.command_line
      job.start_time = self.mock_run_env.start_time
      job.bns_address = self.mock_run_env.bns_address
      run_env.host_dependent_job_info.append(job)

    run_env.host_count = self.mock_run_env.host_count
    run_env.task_count = self.mock_run_env.task_count
    run_env.device_type = self.mock_run_env.device_type
    run_env.device_core_count = self.mock_run_env.device_core_count
    return run_env

  def test_run_environment_empty(self):
    run_env = self.create_empty_run_environment()
    data_table = overview_page_proto_to_gviz.generate_run_environment_table(
        run_env)

    self.assertEqual(0, data_table.NumberOfRows(),
                     "Empty table should have 0 rows.")
    # Check the number of columns in Run environment data table.
    self.assertLen(data_table.columns, self.run_env_num_columns)
    # Check custom properties default values.
    self.assertEqual("0", data_table.custom_properties["host_count"])
    self.assertEqual("0", data_table.custom_properties["task_count"])
    self.assertEqual("", data_table.custom_properties["device_type"])
    self.assertEqual("0", data_table.custom_properties["device_core_count"])

  def test_run_environment_simple(self):
    run_env = self.create_mock_run_environment()
    (table_description, data, custom_properties) = \
        overview_page_proto_to_gviz.get_run_environment_table_args(run_env)
    data_table = gviz_api.DataTable(table_description, data, custom_properties)

    # Data is a list of 3 rows.
    self.assertLen(data, 3)
    self.assertEqual(3, data_table.NumberOfRows(), "Simple table has 3 rows.")
    # Check the number of columns in table descriptor and data table.
    self.assertLen(table_description, self.run_env_num_columns)
    self.assertLen(data_table.columns, self.run_env_num_columns)

    # Prepare expectation to check against.
    # get_run_environment_table_args() formats ns to RFC3339_full format.
    mock_data_run_env = self.mock_run_env._replace(
        start_time="2020-02-20 12:34:56")
    # Check data against mock values.
    for row in data:
      self.assertEqual(list(mock_data_run_env[:self.run_env_num_columns]), row)

    # Check DataTable against mock values.
    # Only way to access DataTable contents is by CSV
    csv_file = io.StringIO(data_table.ToCsv())
    reader = csv.reader(csv_file)

    for (rr, row_values) in enumerate(reader):
      if rr == 0:
        self.check_header_row(data, table_description, row_values)
      else:
        self.check_row_types(data, table_description, row_values, rr)

        self.assertEqual(
            list(mock_data_run_env[:self.run_env_num_columns]), row_values)

    # Check custom properties
    self.assertTrue(data_table.custom_properties["host_count"].startswith(
        str(self.mock_run_env.host_count)))
    self.assertTrue(data_table.custom_properties["task_count"].startswith(
        str(self.mock_run_env.task_count)))
    self.assertTrue(data_table.custom_properties["device_type"].startswith(
        self.mock_run_env.device_type))
    self.assertTrue(
        data_table.custom_properties["device_core_count"].startswith(
            str(self.mock_run_env.device_core_count)))

  def create_mock_overview_page_analysis(self):
    analysis = overview_page_pb2.OverviewPageAnalysis()

    # Add 3 rows
    for _ in range(0, 3):
      op = overview_page_pb2.OverviewTfOp()
      op.self_time_fraction = self.mock_tf_op.self_time_fraction
      op.cumulative_time_fraction = self.mock_tf_op.cumulative_time_fraction
      op.category = self.mock_tf_op.category
      op.name = self.mock_tf_op.name
      op.flop_rate = self.mock_tf_op.flop_rate
      analysis.top_device_ops.append(op)

    return analysis

  def test_overview_page_analysis_empty(self):
    analysis = self.create_empty_overview_page_analysis()
    data_table = \
        overview_page_proto_to_gviz.generate_overview_page_analysis_table(
            analysis)

    self.assertEqual(0, data_table.NumberOfRows(),
                     "Empty table should have 0 rows.")
    # Check the number of Overview Page Analysis data table columns.
    self.assertLen(data_table.columns, len(list(self.mock_tf_op)))

  def test_overview_page_analysis_simple(self):
    analysis = self.create_mock_overview_page_analysis()
    (table_description, data, custom_properties) = \
        overview_page_proto_to_gviz.get_overview_page_analysis_table_args(
            analysis)
    data_table = gviz_api.DataTable(table_description, data, custom_properties)

    # Data is a list of 3 rows.
    self.assertLen(data, 3)
    self.assertEqual(3, data_table.NumberOfRows(), "Simple table has 3 rows.")
    # Check the number of columns in table descriptor and data table.
    self.assertLen(table_description, len(list(self.mock_tf_op)))
    self.assertLen(data_table.columns, len(list(self.mock_tf_op)))

    # Prepare expectation to check against.
    mock_csv_tf_op = [str(x) for x in list(self.mock_tf_op)]

    # Check data against mock values.
    for row in data:
      self.assertEqual(list(self.mock_tf_op), row)

    # Check DataTable against mock values.
    # Only way to access DataTable contents is by CSV
    csv_file = io.StringIO(data_table.ToCsv())
    reader = csv.reader(csv_file)

    for (rr, row_values) in enumerate(reader):
      if rr == 0:
        self.check_header_row(data, table_description, row_values)
      else:
        self.check_row_types(data, table_description, row_values, rr)

        self.assertEqual(mock_csv_tf_op, row_values)

  def create_mock_recommendation(self):
    recommendation = overview_page_pb2.OverviewPageRecommendation()

    for idx in range(0, 3):
      recommendation.faq_tips.add().link = "faq_link" + str(idx)
      recommendation.host_tips.add().link = "host_link" + str(idx)
      recommendation.device_tips.add().link = "device_link" + str(idx)
      recommendation.documentation_tips.add().link = "doc_link" + str(idx)

    return recommendation

  def test_recommendation_empty(self):
    recommendation = self.create_empty_recommendation()
    data_table = overview_page_proto_to_gviz.generate_recommendation_table(
        recommendation)

    self.assertEqual(0, data_table.NumberOfRows(),
                     "Empty table should have 0 rows.")
    # Check the number of Overview Page Recommendation data table columns.
    # One for tip_type, and one for link
    self.assertLen(data_table.columns, 2)

  def test_recommendation_simple(self):
    recommendation = self.create_mock_recommendation()
    (table_description, data, custom_properties) = \
        overview_page_proto_to_gviz.get_recommendation_table_args(
            recommendation)
    data_table = gviz_api.DataTable(table_description, data, custom_properties)

    # Data is a list of 12 rows: 3 rows for each tip type.
    self.assertLen(data, len(list(self.mock_tips)))
    self.assertLen(
        list(self.mock_tips), data_table.NumberOfRows(),
        "Simple table has 12 rows.")
    # Check the number of columns in table descriptor and data table.
    self.assertLen(table_description, 2)
    self.assertLen(data_table.columns, 2)

    # Check data against mock values.
    for idx, row in enumerate(data):
      self.assertEqual(list(self.mock_tips[idx]), row)

    # Check DataTable against mock values.
    # Only way to access DataTable contents is by CSV
    csv_file = io.StringIO(data_table.ToCsv())
    reader = csv.reader(csv_file)

    for (rr, row_values) in enumerate(reader):
      if rr == 0:
        self.check_header_row(data, table_description, row_values)
      else:
        self.check_row_types(data, table_description, row_values, rr)

        self.assertEqual(list(self.mock_tips[rr - 1]), row_values)


if __name__ == "__main__":
  test.main()
