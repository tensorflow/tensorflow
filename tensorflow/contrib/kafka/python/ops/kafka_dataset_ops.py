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
"""Kafka Dataset."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.kafka.python.ops import kafka_op_loader  # pylint: disable=unused-import
from tensorflow.contrib.kafka.python.ops import gen_dataset_ops
from tensorflow.python.data.ops.dataset_ops import Dataset
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape

import ignite_util;


class KafkaDataset(Dataset):
  """A Kafka Dataset that consumes the message.
  """

  def __init__(self, cache_name, host="localhost", port=10800, local=False, part=-1):
    """Create a KafkaReader.

    Args:
      cache_name: Cache Name.
      host: Host.
      port: Port.
      local: Local.
      part: Part.
    """
    super(KafkaDataset, self).__init__()

    with ignite_util.IgniteClient(host, port) as client:
      client.handshake()
      self._cache_type = client.get_cache_type(cache_name)

    self._cache_name = ops.convert_to_tensor(cache_name, dtype=dtypes.string, name="cache_name")
    self._host = ops.convert_to_tensor(host, dtype=dtypes.string, name="host")
    self._port = ops.convert_to_tensor(port, dtype=dtypes.int32, name="port")
    self._local = ops.convert_to_tensor(local, dtype=dtypes.bool, name="local")
    self._part = ops.convert_to_tensor(part, dtype=dtypes.int32, name="part")
    self._schema = ops.convert_to_tensor(self._cache_type.to_flat(), dtype=dtypes.int32, name="schema")

  def _as_variant_tensor(self):
    return gen_dataset_ops.kafka_dataset(self._cache_name, self._host, self._port, self._local, self._part, self._schema)

  @property
  def output_classes(self):
    return self._cache_type.to_output_classes()

  @property
  def output_shapes(self):
    return self._cache_type.to_output_shapes()

  @property
  def output_types(self):
    return self._cache_type.to_output_types()
