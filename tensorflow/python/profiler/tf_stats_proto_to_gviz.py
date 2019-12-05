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
"""Contains utilities for conversion of TF proto types to GViz types.

Usage:
    gviz_data_table = generate_chart_table(stats_table)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import google_type_annotations
from __future__ import print_function

import gviz_api


def get_chart_table_args(stats_table):
  """Creates gviz DataTable object from a a TensorFlow stats table.

  Args:
    stats_table: A tf_stats_pb2.TfStatsTable.

  Returns:
    Returns a gviz_api.DataTable
  """

  ## Create schema
  table_description = [
      ("rank", "number", "Rank"),
      ("host_or_device", "string", "Host/device"),
      ("type", "string", "Type"),
      ("operation", "string", "Operation"),
      ("occurrences", "number", "#Occurrences"),
      ("total_time", "number", "Total time (us)"),
      ("avg_time", "number", "Avg. time (us)"),
      ("total_self_time", "number", "Total self-time (us)"),
      ("avg_self_time", "number", "Avg. self-time (us)"),
      ("device_total_self_time_percent", "number",
       "Total self-time on Device (%)"),
      ("device_cumulative_total_self_time_percent", "number",
       "Cumulative total-self time on Device (%)"),
      ("host_total_self_time_percent", "number", "Total self-time on Host (%)"),
      ("Host_cumulative_total_self_time_percent", "number",
       "Cumulative total-self time on Host (%)"),
      ("measured_flop_rate", "number", "Measured GFLOPs/Sec"),
      ("measured_memory_bw", "number", "Measured Memory BW (GBytes/Sec)"),
      ("operational_intensity", "number", "Operational Intensity (FLOPs/Byte)"),
      ("bound_by", "string", "Bound by"),
  ]

  data = []
  for record in stats_table.tf_stats_record:
    row = [
        record.rank,
        record.host_or_device,
        record.op_type,
        record.op_name,
        record.occurrences,
        record.total_time_in_us,
        record.avg_time_in_us,
        record.total_self_time_in_us,
        record.avg_self_time_in_us,
        record.device_total_self_time_as_fraction,
        record.device_cumulative_total_self_time_as_fraction,
        record.host_total_self_time_as_fraction,
        record.host_cumulative_total_self_time_as_fraction,
        record.measured_flop_rate,
        record.measured_memory_bw,
        record.operational_intensity,
        record.bound_by,
    ]

    data.append(row)

  return (table_description, data, [])


def generate_chart_table(stats_table):
  (table_description, data,
   custom_properties) = get_chart_table_args(stats_table)
  return gviz_api.DataTable(table_description, data, custom_properties)
