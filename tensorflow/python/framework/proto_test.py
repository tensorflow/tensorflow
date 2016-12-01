# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""Protobuf related tests."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

class ProtoTest(tf.test.TestCase):

  # TODO(vrv): re-enable this test once we figure out how this can
  # pass the pip install test (where the user is expected to have
  # protobuf installed).
  def _testLargeProto(self):
    # create a constant of size > 64MB.
    a = tf.constant(np.zeros([1024, 1024, 17]))
    # Serialize the resulting graph def.
    gdef = a.op.graph.as_graph_def()
    serialized = gdef.SerializeToString()
    unserialized = tf.Graph().as_graph_def()
    # Deserialize back. Protobuf python library should support
    # protos larger than 64MB.
    unserialized.ParseFromString(serialized)
    self.assertProtoEquals(unserialized, gdef)


if __name__ == "__main__":
  tf.test.main()
