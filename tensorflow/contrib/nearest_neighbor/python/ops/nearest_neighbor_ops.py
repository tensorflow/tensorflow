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
# ==============================================================================
"""Wrappers for nearest neighbor operations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.util import loader
from tensorflow.python.framework import ops
from tensorflow.python.platform import resource_loader

_nearest_neighbor_ops = loader.load_op_library(
    resource_loader.get_path_to_datafile("_nearest_neighbor_ops.so"))


def hyperplane_lsh_probes(point_hyperplane_product,
                          num_tables,
                          num_hyperplanes_per_table,
                          num_probes,
                          name=None):
  """Computes probes for the hyperplane hash.

  The op supports multiprobing, i.e., the number of requested probes can be
  larger than the number of tables. In that case, the same table can be probed
  multiple times.

  The first `num_tables` probes are always the primary hashes for each table.

  Args:
    point_hyperplane_product: a matrix of inner products between the hyperplanes
      and the points to be hashed. These values should not be quantized so that
      we can correctly compute the probing sequence. The expected shape is
      `batch_size` times `num_tables * num_hyperplanes_per_table`, i.e., each
      element of the batch corresponds to one row of the matrix.
    num_tables: the number of tables to compute probes for.
    num_hyperplanes_per_table: the number of hyperplanes per table.
    num_probes: the requested number of probes per table.
    name: A name prefix for the returned tensors (optional).

  Returns:
    probes: the output matrix of probes. Size `batch_size` times `num_probes`.
    table_ids: the output matrix of tables ids. Size `batch_size` times
      `num_probes`.
  """
  return _nearest_neighbor_ops.hyperplane_lsh_probes(point_hyperplane_product,
                                                     num_tables,
                                                     num_hyperplanes_per_table,
                                                     num_probes,
                                                     name=name)

ops.NotDifferentiable("HyperplaneLSHProbes")
