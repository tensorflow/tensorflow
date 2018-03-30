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
"""Tests for boosted_trees resource kernels."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from google.protobuf import text_format

from tensorflow.core.kernels.boosted_trees import boosted_trees_pb2
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import boosted_trees_ops
from tensorflow.python.ops import resources
from tensorflow.python.platform import googletest


class ResourceOpsTest(test_util.TensorFlowTestCase):
  """Tests resource_ops."""

  def testCreate(self):
    with self.test_session():
      ensemble = boosted_trees_ops.TreeEnsemble('ensemble')
      resources.initialize_resources(resources.shared_resources()).run()
      stamp_token = ensemble.get_stamp_token()
      self.assertEqual(0, stamp_token.eval())
      (_, num_trees, num_finalized_trees,
       num_attempted_layers) = ensemble.get_states()
      self.assertEqual(0, num_trees.eval())
      self.assertEqual(0, num_finalized_trees.eval())
      self.assertEqual(0, num_attempted_layers.eval())

  def testCreateWithProto(self):
    with self.test_session():
      ensemble_proto = boosted_trees_pb2.TreeEnsemble()
      text_format.Merge("""
        trees {
          nodes {
            bucketized_split {
              feature_id: 4
              left_id: 1
              right_id: 2
            }
            metadata {
              gain: 7.62
            }
          }
          nodes {
            bucketized_split {
              threshold: 21
              left_id: 3
              right_id: 4
            }
            metadata {
              gain: 1.4
              original_leaf {
                scalar: 7.14
              }
            }
          }
          nodes {
            bucketized_split {
              feature_id: 1
              threshold: 7
              left_id: 5
              right_id: 6
            }
            metadata {
              gain: 2.7
              original_leaf {
                scalar: -4.375
              }
            }
          }
          nodes {
            leaf {
              scalar: 6.54
            }
          }
          nodes {
            leaf {
              scalar: 7.305
            }
          }
          nodes {
            leaf {
              scalar: -4.525
            }
          }
          nodes {
            leaf {
              scalar: -4.145
            }
          }
        }
        trees {
          nodes {
            bucketized_split {
              feature_id: 75
              threshold: 21
              left_id: 1
              right_id: 2
            }
            metadata {
              gain: -1.4
            }
          }
          nodes {
            leaf {
              scalar: -0.6
            }
          }
          nodes {
            leaf {
              scalar: 0.165
            }
          }
        }
        tree_weights: 0.15
        tree_weights: 1.0
        tree_metadata {
          num_layers_grown: 2
          is_finalized: true
        }
        tree_metadata {
          num_layers_grown: 1
          is_finalized: false
        }
        growing_metadata {
          num_trees_attempted: 2
          num_layers_attempted: 6
        }
      """, ensemble_proto)
      ensemble = boosted_trees_ops.TreeEnsemble(
          'ensemble',
          stamp_token=7,
          serialized_proto=ensemble_proto.SerializeToString())
      resources.initialize_resources(resources.shared_resources()).run()
      (stamp_token, num_trees, num_finalized_trees,
       num_attempted_layers) = ensemble.get_states()
      self.assertEqual(7, stamp_token.eval())
      self.assertEqual(2, num_trees.eval())
      self.assertEqual(1, num_finalized_trees.eval())
      self.assertEqual(6, num_attempted_layers.eval())

  def testSerializeDeserialize(self):
    with self.test_session():
      # Initialize.
      ensemble = boosted_trees_ops.TreeEnsemble('ensemble', stamp_token=5)
      resources.initialize_resources(resources.shared_resources()).run()
      (stamp_token, num_trees, num_finalized_trees,
       num_attempted_layers) = ensemble.get_states()
      self.assertEqual(5, stamp_token.eval())
      self.assertEqual(0, num_trees.eval())
      self.assertEqual(0, num_finalized_trees.eval())
      self.assertEqual(0, num_attempted_layers.eval())

      # Deserialize.
      ensemble_proto = boosted_trees_pb2.TreeEnsemble()
      text_format.Merge("""
        trees {
          nodes {
            bucketized_split {
              feature_id: 75
              threshold: 21
              left_id: 1
              right_id: 2
            }
            metadata {
              gain: -1.4
            }
          }
          nodes {
            leaf {
              scalar: -0.6
            }
          }
          nodes {
            leaf {
              scalar: 0.165
            }
          }
        }
        tree_weights: 0.5
        tree_metadata {
          num_layers_grown: 4  # it's fake intentionally.
          is_finalized: false
        }
        growing_metadata {
          num_trees_attempted: 1
          num_layers_attempted: 5
        }
      """, ensemble_proto)
      with ops.control_dependencies([
          ensemble.deserialize(
              stamp_token=3,
              serialized_proto=ensemble_proto.SerializeToString())
      ]):
        (stamp_token, num_trees, num_finalized_trees,
         num_attempted_layers) = ensemble.get_states()
      self.assertEqual(3, stamp_token.eval())
      self.assertEqual(1, num_trees.eval())
      # This reads from metadata, not really counting the layers.
      self.assertEqual(5, num_attempted_layers.eval())
      self.assertEqual(0, num_finalized_trees.eval())

      # Serialize.
      new_ensemble_proto = boosted_trees_pb2.TreeEnsemble()
      new_stamp_token, new_serialized = ensemble.serialize()
      self.assertEqual(3, new_stamp_token.eval())
      new_ensemble_proto.ParseFromString(new_serialized.eval())
      self.assertProtoEquals(ensemble_proto, new_ensemble_proto)


if __name__ == '__main__':
  googletest.main()
