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
"""Integration test for sequence feature columns with SequenceExamples."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from google.protobuf import text_format

from tensorflow.core.example import example_pb2
from tensorflow.core.example import feature_pb2
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.feature_column import feature_column_v2 as fc
from tensorflow.python.feature_column import sequence_feature_column as sfc
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import test_util
from tensorflow.python.keras.feature_column import dense_features
from tensorflow.python.keras.feature_column import sequence_feature_column as ksfc
from tensorflow.python.keras.layers import recurrent
from tensorflow.python.ops import init_ops_v2
from tensorflow.python.ops import parsing_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.util import compat


class SequenceFeatureColumnIntegrationTest(test.TestCase):

  def _make_sequence_example(self):
    example = example_pb2.SequenceExample()
    example.context.feature['int_ctx'].int64_list.value.extend([5])
    example.context.feature['float_ctx'].float_list.value.extend([123.6])
    for val in range(0, 10, 2):
      feat = feature_pb2.Feature()
      feat.int64_list.value.extend([val] * val)
      example.feature_lists.feature_list['int_list'].feature.extend([feat])
    for val in range(1, 11, 2):
      feat = feature_pb2.Feature()
      feat.bytes_list.value.extend([compat.as_bytes(str(val))] * val)
      example.feature_lists.feature_list['str_list'].feature.extend([feat])

    return example

  def _build_feature_columns(self):
    col = fc.categorical_column_with_identity('int_ctx', num_buckets=100)
    ctx_cols = [
        fc.embedding_column(col, dimension=10),
        fc.numeric_column('float_ctx')
    ]

    identity_col = sfc.sequence_categorical_column_with_identity(
        'int_list', num_buckets=10)
    bucket_col = sfc.sequence_categorical_column_with_hash_bucket(
        'bytes_list', hash_bucket_size=100)
    seq_cols = [
        fc.embedding_column(identity_col, dimension=10),
        fc.embedding_column(bucket_col, dimension=20)
    ]

    return ctx_cols, seq_cols

  def test_sequence_example_into_input_layer(self):
    examples = [_make_sequence_example().SerializeToString()] * 100
    ctx_cols, seq_cols = self._build_feature_columns()

    def _parse_example(example):
      ctx, seq = parsing_ops.parse_single_sequence_example(
          example,
          context_features=fc.make_parse_example_spec_v2(ctx_cols),
          sequence_features=fc.make_parse_example_spec_v2(seq_cols))
      ctx.update(seq)
      return ctx

    ds = dataset_ops.Dataset.from_tensor_slices(examples)
    ds = ds.map(_parse_example)
    ds = ds.batch(20)

    # Test on a single batch
    features = dataset_ops.make_one_shot_iterator(ds).get_next()

    # Tile the context features across the sequence features
    sequence_input_layer = ksfc.SequenceFeatures(seq_cols)
    seq_layer, _ = sequence_input_layer(features)
    input_layer = dense_features.DenseFeatures(ctx_cols)
    ctx_layer = input_layer(features)
    input_layer = sfc.concatenate_context_input(ctx_layer, seq_layer)

    rnn_layer = recurrent.RNN(recurrent.SimpleRNNCell(10))
    output = rnn_layer(input_layer)

    with self.cached_session() as sess:
      sess.run(variables.global_variables_initializer())
      features_r = sess.run(features)
      self.assertAllEqual(features_r['int_list'].dense_shape, [20, 3, 6])

      output_r = sess.run(output)
      self.assertAllEqual(output_r.shape, [20, 10])

  @test_util.run_deprecated_v1
  def test_shared_sequence_non_sequence_into_input_layer(self):
    non_seq = fc.categorical_column_with_identity('non_seq',
                                                  num_buckets=10)
    seq = sfc.sequence_categorical_column_with_identity('seq',
                                                        num_buckets=10)
    shared_non_seq, shared_seq = fc.shared_embedding_columns_v2(
        [non_seq, seq],
        dimension=4,
        combiner='sum',
        initializer=init_ops_v2.Ones(),
        shared_embedding_collection_name='shared')

    seq = sparse_tensor.SparseTensor(
        indices=[[0, 0], [0, 1], [1, 0]],
        values=[0, 1, 2],
        dense_shape=[2, 2])
    non_seq = sparse_tensor.SparseTensor(
        indices=[[0, 0], [0, 1], [1, 0]],
        values=[0, 1, 2],
        dense_shape=[2, 2])
    features = {'seq': seq, 'non_seq': non_seq}

    # Tile the context features across the sequence features
    seq_input, seq_length = ksfc.SequenceFeatures([shared_seq])(features)
    non_seq_input = dense_features.DenseFeatures([shared_non_seq])(features)

    with self.cached_session() as sess:
      sess.run(variables.global_variables_initializer())
      output_seq, output_seq_length, output_non_seq = sess.run(
          [seq_input, seq_length, non_seq_input])
      self.assertAllEqual(output_seq, [[[1, 1, 1, 1], [1, 1, 1, 1]],
                                       [[1, 1, 1, 1], [0, 0, 0, 0]]])
      self.assertAllEqual(output_seq_length, [2, 1])
      self.assertAllEqual(output_non_seq, [[2, 2, 2, 2], [1, 1, 1, 1]])


_SEQ_EX_PROTO = """
context {
  feature {
    key: "float_ctx"
    value {
      float_list {
        value: 123.6
      }
    }
  }
  feature {
    key: "int_ctx"
    value {
      int64_list {
        value: 5
      }
    }
  }
}
feature_lists {
  feature_list {
    key: "bytes_list"
    value {
      feature {
        bytes_list {
          value: "a"
        }
      }
      feature {
        bytes_list {
          value: "b"
          value: "c"
        }
      }
      feature {
        bytes_list {
          value: "d"
          value: "e"
          value: "f"
          value: "g"
        }
      }
    }
  }
  feature_list {
    key: "float_list"
    value {
      feature {
        float_list {
          value: 1.0
        }
      }
      feature {
        float_list {
          value: 3.0
          value: 3.0
          value: 3.0
        }
      }
      feature {
        float_list {
          value: 5.0
          value: 5.0
          value: 5.0
          value: 5.0
          value: 5.0
        }
      }
    }
  }
  feature_list {
    key: "int_list"
    value {
      feature {
        int64_list {
          value: 2
          value: 2
        }
      }
      feature {
        int64_list {
          value: 4
          value: 4
          value: 4
          value: 4
        }
      }
      feature {
        int64_list {
          value: 6
          value: 6
          value: 6
          value: 6
          value: 6
          value: 6
        }
      }
    }
  }
}
"""


def _make_sequence_example():
  example = example_pb2.SequenceExample()
  return text_format.Parse(_SEQ_EX_PROTO, example)


if __name__ == '__main__':
  test.main()
