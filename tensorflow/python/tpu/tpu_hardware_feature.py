# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
"""TPU hardware feature info."""
import enum
from tensorflow.core.protobuf.tpu import topology_pb2
from tensorflow.python.util.tf_export import tf_export


@tf_export("tpu.experimental.HardwareFeature")
class HardwareFeature(object):
  """class holds all the feature info about the TPU."""

  def __init__(self, tpu_hardware_feature_proto):
    """Store TPU hardware feature info.

    Args:
      tpu_hardware_feature_proto: protobuf which describe the tpu hardware
        feature.
    """
    self.tpu_hardware_feature_proto = tpu_hardware_feature_proto

  class EmbeddingFeature(enum.Enum):
    """Embedding feature flag strings.

    UNSUPPORTED: No embedding lookup accelerator available on the tpu.
    V1: Embedding lookup accelerator V1. The embedding lookup operation can only
        be placed at the beginning of computation. Only one instance of
        embedding
        lookup layer is allowed.
    V2: Embedding lookup accelerator V2. The embedding lookup operation can be
        placed anywhere of the computation. Multiple instances of embedding
        lookup layer is allowed.
    """
    UNSUPPORTED = "UNSUPPORTED"
    V1 = "V1"
    V2 = "V2"

  @classmethod
  def _embedding_feature_proto_to_string(cls, embedding_feature_proto):
    """Convert the embedding feature proto to enum string."""
    embedding_feature_proto_to_string_map = {
        topology_pb2.TPUHardwareFeature.EmbeddingFeature.UNSUPPORTED:
            HardwareFeature.EmbeddingFeature.UNSUPPORTED,
        topology_pb2.TPUHardwareFeature.EmbeddingFeature.V1:
            HardwareFeature.EmbeddingFeature.V1,
        topology_pb2.TPUHardwareFeature.EmbeddingFeature.V2:
            HardwareFeature.EmbeddingFeature.V2
    }
    return embedding_feature_proto_to_string_map.get(
        embedding_feature_proto, HardwareFeature.EmbeddingFeature.UNSUPPORTED)

  @property
  def embedding_feature(self):
    """TPU embedding feature.

    Returns:
      An EmbeddingFeature enum.
    """
    return HardwareFeature._embedding_feature_proto_to_string(
        self.tpu_hardware_feature_proto.embedding_feature)

  @property
  def num_embedding_devices_per_chip(self):
    """Number of embedding accelerator devices per chip.

    Returns:
      Number of embedding devices per chip.
    """
    return self.tpu_hardware_feature_proto.num_embedding_devices_per_chip
