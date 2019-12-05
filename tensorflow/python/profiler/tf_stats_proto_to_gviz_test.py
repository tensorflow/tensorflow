# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for tf_stats_proto_to_gviz."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import io

import gviz_api

# pylint: disable=g-direct-tensorflow-import
from tensorflow.core.profiler.protobuf import tf_stats_pb2
from tensorflow.python.platform import test
from tensorflow.python.profiler import tf_stats_proto_to_gviz
# pylint: enable=g-direct-tensorflow-import


class ProtoToGvizTest(test.TestCase):

  @staticmethod
  def create_empty_stats_table():
    table = tf_stats_pb2.TfStatsTable()

    return table

  @staticmethod
  def create_mock_stats_table():
    table = tf_stats_pb2.TfStatsTable()

    record = table.tf_stats_record.add()
    record.rank = 100
    record.host_or_device = "Device"
    record.op_type = "Compute"
    record.op_name = "Compute0"
    record.occurrences = 1
    record.total_time_in_us = 0.1799
    record.avg_time_in_us = 0.1799
    record.total_self_time_in_us = 0.1799
    record.avg_self_time_in_us = 0.1799
    record.device_total_self_time_as_fraction = 0.2020
    record.device_cumulative_total_self_time_as_fraction = 0.7980
    record.host_total_self_time_as_fraction = 0
    record.host_cumulative_total_self_time_as_fraction = 0
    record.measured_flop_rate = 1.6666
    record.measured_memory_bw = 2.7777
    record.operational_intensity = 0.6000
    record.bound_by = "Memory"

    record = table.tf_stats_record.add()
    record.rank = 200
    record.host_or_device = "Host"
    record.op_type = "Loop"
    record.op_name = "while"
    record.occurrences = 2
    record.total_time_in_us = 0.3
    record.avg_time_in_us = 0.5
    record.total_self_time_in_us = 0.7
    record.avg_self_time_in_us = 0.11
    record.device_total_self_time_as_fraction = 0.13
    record.device_cumulative_total_self_time_as_fraction = 0.17
    record.host_total_self_time_as_fraction = 0.19
    record.host_cumulative_total_self_time_as_fraction = 0.23
    record.measured_flop_rate = 2.9
    record.measured_memory_bw = 3.1
    record.operational_intensity = 0.37
    record.bound_by = "Compute"

    return table

  def test_stats_table_empty(self):
    stats_table = ProtoToGvizTest.create_empty_stats_table()
    data_table = tf_stats_proto_to_gviz.generate_chart_table(stats_table)

    self.assertEqual(0, data_table.NumberOfRows(),
                     "Empty table should have 0 rows.")
    # "Stats table has 17 columns as defined in tf_stats.proto."
    self.assertLen(data_table.columns, 17)

  def test_stats_table_simple(self):
    stats_table = ProtoToGvizTest.create_mock_stats_table()
    (table_description, data, custom_properties
    ) = tf_stats_proto_to_gviz.get_chart_table_args(stats_table)
    data_table = gviz_api.DataTable(table_description, data, custom_properties)

    # Data is a list of 2 rows.
    self.assertLen(data, 2)
    self.assertEqual(2, data_table.NumberOfRows(), "Simple table has 2 rows.")
    # Table descriptor is a list of 17 columns.
    self.assertLen(table_description, 17)
    # Stats table has 17 columns as defined in tf_stats.proto.
    self.assertLen(data_table.columns, 17)

    csv_file = io.StringIO(data_table.ToCsv())
    reader = csv.reader(csv_file)

    for (rr, row_values) in enumerate(reader):
      if rr == 0:
        for (cc, column_header) in enumerate(row_values):
          self.assertEqual(table_description[cc][2], column_header)
      else:
        for (cc, cell_str) in enumerate(row_values):
          raw_value = data[rr - 1][cc]
          value_type = table_description[cc][1]

          # Only number and strings are used in our (tf_stats) proto.
          self.assertIn(value_type, ["number", "string"])

          # Encode in similar fashion as DataTable.ToCsv()
          expected_value = gviz_api.DataTable.CoerceValue(raw_value, value_type)
          self.assertNotIsInstance(expected_value, tuple)
          self.assertEqual(expected_value, raw_value)
          self.assertEqual(str(expected_value), cell_str)


if __name__ == "__main__":
  test.main()
