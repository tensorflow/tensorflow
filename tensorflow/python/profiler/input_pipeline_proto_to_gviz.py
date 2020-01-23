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
"""For conversion of TF Input Pipeline Analyzer protos to GViz DataTables.

Usage:
    gviz_data_table = generate_chart_table(ipa)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import google_type_annotations
from __future__ import print_function

import gviz_api

from tensorflow.core.profiler.protobuf import input_pipeline_pb2


def get_chart_table_args(ipa):
  """Creates a gviz DataTable object from an Input Pipeline Analyzer proto.

  Args:
    ipa: An input_pipeline_pb2.InputPipelineAnalysisResult.

  Returns:
    Returns a gviz_api.DataTable
  """

  table_description = [
      ("stepnum", "string", "Step number"),
      ("deviceComputeTimeMs", "number", "Device compute"),
      ("deviceToDeviceTimeMs", "number", "Device to device"),
      ("hostComputeTimeMs", "number", "Host compute"),
      ("kernelLaunchTimeMs", "number", "Kernel launch"),
      ("infeedTimeMs", "number", "Input"),
      ("outfeedTimeMs", "number", "Output"),
      ("compileTimeMs", "number", "Compilation"),
      ("otherTimeMs", "number", "All others"),
  ]

  data = []
  for step_details in ipa.step_details:
    details = input_pipeline_pb2.PerGenericStepDetails()
    step_details.Unpack(details)
    row = [
        str(details.step_number),
        details.device_compute_ms,
        details.device_to_device_ms,
        details.host_compute_ms,
        details.host_prepare_ms,
        details.host_wait_input_ms + details.host_to_device_ms,
        details.output_ms,
        details.host_compile_ms,
        details.unknown_time_ms,
    ]
    data.append(row)

  return (table_description, data, [])


def generate_chart_table(ipa):
  (table_description, data, custom_properties) = get_chart_table_args(ipa)
  return gviz_api.DataTable(table_description, data, custom_properties)
