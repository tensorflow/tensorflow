# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
# =============================================================================
"""Provides a proper python API for the symbols exported through swig."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.grappler import _pywrap_cost_analyzer as tf_wrap
from tensorflow.python.grappler import cluster as gcluster
from tensorflow.python.grappler import item as gitem


def GenerateCostReport(metagraph,
                       per_node_report=False,
                       verbose=False,
                       cluster=None):
  """Analyze the cost of each TensorFlow op and node in the provided metagraph.

  Args:
    metagraph: A TensorFlow MetaGraphDef.
    per_node_report: by default the report contains stats aggregated on a per op
      type basis, setting per_node_report to True adds results for each
      individual node to the report.
    verbose: Prints out the entire operation proto instead of a summary table.
    cluster: Analyze the costs using the specified cluster, or the local machine
      if no cluster was specified.

  Returns:
    A string of cost report.
  """
  if cluster is None:
    cluster = gcluster.Cluster(disable_detailed_stats=False)

  return tf_wrap.GenerateCostReport(metagraph.SerializeToString(),
                                    per_node_report, verbose,
                                    cluster.tf_cluster)


def GenerateMemoryReport(metagraph, detailed_report=True, cluster=None):
  """Analyze the peak memory usage for the provided metagraph.

  Args:
    metagraph: A TensorFlow MetaGraphDef.
    detailed_report: print the live tensors in addition to the peak memory
      usage.
    cluster: Analyze the memory using the specified cluster, or the local
      machine if no cluster was specified.

  Returns:
    A string with the formatted memory usage.
  """
  if cluster is None:
    cluster = gcluster.Cluster(
        disable_detailed_stats=True, disable_timeline=True)

  item = gitem.Item(metagraph)
  peak_usage = cluster.DeterminePeakMemoryUsage(item)
  report = ""
  for device, snapshot in peak_usage.items():
    peak_usage = snapshot[0]
    report += "Peak usage for device " + device + ": " + str(
        peak_usage) + " bytes\n"
    if detailed_report:
      live_tensors = snapshot[1]
      for tensor in live_tensors:
        op_name = tensor[0]
        output_id = tensor[1]
        mem_used = tensor[2]
        report += "  " + str(op_name) + ":" + str(output_id) + " uses " + str(
            mem_used) + " bytes\n"

  return report
