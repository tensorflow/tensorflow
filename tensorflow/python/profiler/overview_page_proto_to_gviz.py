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
"""For conversion of TF Overview Page protos to GViz DataTables.

Usage:
    gviz_data_table = generate_chart_table(overview_page)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import google_type_annotations
from __future__ import print_function

import datetime
import gviz_api


def get_run_environment_table_args(run_environment):
  """Creates a gviz DataTable object from a RunEnvironment proto.

  Args:
    run_environment: An op_stats_pb2.RunEnvironment.

  Returns:
    Returns a gviz_api.DataTable
  """

  table_description = [
      ("host_id", "string", "host_id"),
      ("command_line", "string", "command_line"),
      ("start_time", "string", "start_time"),
      ("bns_address", "string", "bns_address"),
  ]

  data = []
  for job in run_environment.host_dependent_job_info:
    row = [
        str(job.host_id),
        str(job.command_line),
        str(datetime.datetime.utcfromtimestamp(job.start_time)),
        str(job.bns_address),
    ]
    data.append(row)

  return (table_description, data, [])


def generate_run_environment_table(run_environment):
  (table_description, data,
   custom_properties) = get_run_environment_table_args(run_environment)
  return gviz_api.DataTable(table_description, data, custom_properties)


def get_overview_page_analysis_table_args(overview_page_analysis):
  """Creates a gviz DataTable object from an OverviewPageAnalysis proto.

  Args:
    overview_page_analysis: An overview_page_pb2.OverviewPageAnalysis.

  Returns:
    Returns a gviz_api.DataTable
  """

  table_description = [
      ("selfTimePercent", "number", "Time (%)"),
      ("cumulativeTimePercent", "number", "Cumulative time (%)"),
      ("category", "string", "Category"),
      ("operation", "string", "Operation"),
      ("flopRate", "number", "GFLOPs/Sec"),
  ]

  data = []
  for op in overview_page_analysis.top_device_ops:
    row = [
        op.self_time_fraction,
        op.cumulative_time_fraction,
        str(op.category),
        str(op.name),
        op.flop_rate,
    ]
    data.append(row)

  return (table_description, data, [])


def generate_overview_page_analysis_table(overview_page_analysis):
  (table_description, data, custom_properties) = \
      get_overview_page_analysis_table_args(overview_page_analysis)
  return gviz_api.DataTable(table_description, data, custom_properties)


def get_recommendation_table_args(overview_page_recommendation):
  """Creates a gviz DataTable object from an OverviewPageRecommendation proto.

  Args:
    overview_page_recommendation: An
      overview_page_pb2.OverviewPageRecommendation.

  Returns:
    Returns a gviz_api.DataTable
  """

  table_description = [
      ("tip_type", "string", "tip_type"),
      ("link", "string", "link"),
  ]

  data = []
  for faq_tip in overview_page_recommendation.faq_tips:
    data.append(["faq", faq_tip.link])

  for host_tip in overview_page_recommendation.host_tips:
    data.append(["host", host_tip.link])

  for device_tip in overview_page_recommendation.device_tips:
    data.append(["device", device_tip.link])

  for doc_tip in overview_page_recommendation.documentation_tips:
    data.append(["doc", doc_tip.link])

  for inference_tip in overview_page_recommendation.inference_tips:
    data.append(["inference", inference_tip.link])

  return (table_description, data, [])


def generate_recommendation_table(overview_page_recommendation):
  (table_description, data, custom_properties) = \
      get_recommendation_table_args(overview_page_recommendation)
  return gviz_api.DataTable(table_description, data, custom_properties)
