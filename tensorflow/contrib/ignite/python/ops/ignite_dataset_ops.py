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
"""Ignite Dataset."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.ignite.python.ops import ignite_op_loader  # pylint: disable=unused-import
from tensorflow.contrib.ignite.python.ops import gen_dataset_ops
from tensorflow.python.data.ops.dataset_ops import Dataset
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape


class IgniteDataset(Dataset):
  """An Ignite  Dataset.
  This dataset reads data from specified cache from Apache Ignite.
  For now (as long as connection is made with 'fat' client) only caches
  with integer keys and string values are supported.
  To configure 'fat' client some environment variables should be set:
  LD_LIBRARY_PATH should be set in such a way that it contains libjvm.so, for example
  export LD_LIBRARY_PATH=/usr/lib/jvm/java-8-oracle/jre/lib/amd64/server/
  IGNITE_HOME should point to Apache Ignite installation directory, for example
  export IGNITE_HOME=~/apache-ignite-fabric-2.4.0-bin
  TF_IGNITE_CLIENT_CONFIG should point to client ignite node config, for example
  export TF_IGNITE_CLIENT_CONFIG=../../sample_configs/client.xml
  """

  def __init__(self, cache):
    """Create a IgniteReader.

    Args:
      cache: A `tf.string` tensor containing cache name.
    """
    super(IgniteDataset, self).__init__()
    self._cache = ops.convert_to_tensor(
        cache, dtype=dtypes.string, name="cache")

  def _as_variant_tensor(self):
    return gen_dataset_ops.ignite_dataset(self._cache)

  @property
  def output_classes(self):
    return ops.Tensor

  @property
  def output_shapes(self):
    return tensor_shape.scalar()

  @property
  def output_types(self):
    return dtypes.string
