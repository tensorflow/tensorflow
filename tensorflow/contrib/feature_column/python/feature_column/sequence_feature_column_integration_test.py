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

import string
import tempfile

from google.protobuf import text_format

from tensorflow.contrib.feature_column.python.feature_column import sequence_feature_column as sfc
from tensorflow.core.example import example_pb2
from tensorflow.core.example import feature_pb2
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.feature_column import feature_column as fc
from tensorflow.python.keras.layers import recurrent
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
    col = fc._categorical_column_with_identity('int_ctx', num_buckets=100)
    ctx_cols = [
        fc._embedding_column(col, dimension=10),
        fc._numeric_column('float_ctx')
    ]

    identity_col = sfc.sequence_categorical_column_with_identity(
        'int_list', num_buckets=10)
    bucket_col = sfc.sequence_categorical_column_with_hash_bucket(
        'bytes_list', hash_bucket_size=100)
    seq_cols = [
        fc._embedding_column(identity_col, dimension=10),
        fc._embedding_column(bucket_col, dimension=20)
    ]

    return ctx_cols, seq_cols

  def test_sequence_example_into_input_layer(self):
    examples = [_make_sequence_example().SerializeToString()] * 100
    ctx_cols, seq_cols = self._build_feature_columns()

    def _parse_example(example):
      ctx, seq = parsing_ops.parse_single_sequence_example(
          example,
          context_features=fc.make_parse_example_spec(ctx_cols),
          sequence_features=fc.make_parse_example_spec(seq_cols))
      ctx.update(seq)
      return ctx

    ds = dataset_ops.Dataset.from_tensor_slices(examples)
    ds = ds.map(_parse_example)
    ds = ds.batch(20)

    # Test on a single batch
    features = ds.make_one_shot_iterator().get_next()

    # Tile the context features across the sequence features
    seq_layer, _ = sfc.sequence_input_layer(features, seq_cols)
    ctx_layer = fc.input_layer(features, ctx_cols)
    input_layer = sfc.concatenate_context_input(ctx_layer, seq_layer)

    rnn_layer = recurrent.RNN(recurrent.SimpleRNNCell(10))
    output = rnn_layer(input_layer)

    with self.cached_session() as sess:
      sess.run(variables.global_variables_initializer())
      features_r = sess.run(features)
      self.assertAllEqual(features_r['int_list'].dense_shape, [20, 3, 6])

      output_r = sess.run(output)
      self.assertAllEqual(output_r.shape, [20, 10])


class SequenceExampleParsingTest(test.TestCase):

  def test_seq_ex_in_sequence_categorical_column_with_identity(self):
    self._test_parsed_sequence_example(
        'int_list', sfc.sequence_categorical_column_with_identity,
        10, [3, 6], [2, 4, 6])

  def test_seq_ex_in_sequence_categorical_column_with_hash_bucket(self):
    self._test_parsed_sequence_example(
        'bytes_list', sfc.sequence_categorical_column_with_hash_bucket,
        10, [3, 4], [compat.as_bytes(x) for x in 'acg'])

  def test_seq_ex_in_sequence_categorical_column_with_vocabulary_list(self):
    self._test_parsed_sequence_example(
        'bytes_list', sfc.sequence_categorical_column_with_vocabulary_list,
        list(string.ascii_lowercase), [3, 4],
        [compat.as_bytes(x) for x in 'acg'])

  def test_seq_ex_in_sequence_categorical_column_with_vocabulary_file(self):
    _, fname = tempfile.mkstemp()
    with open(fname, 'w') as f:
      f.write(string.ascii_lowercase)
    self._test_parsed_sequence_example(
        'bytes_list', sfc.sequence_categorical_column_with_vocabulary_file,
        fname, [3, 4], [compat.as_bytes(x) for x in 'acg'])

  def _test_parsed_sequence_example(
      self, col_name, col_fn, col_arg, shape, values):
    """Helper function to check that each FeatureColumn parses correctly.

    Args:
      col_name: string, name to give to the feature column. Should match
        the name that the column will parse out of the features dict.
      col_fn: function used to create the feature column. For example,
        sequence_numeric_column.
      col_arg: second arg that the target feature column is expecting.
      shape: the expected dense_shape of the feature after parsing into
        a SparseTensor.
      values: the expected values at index [0, 2, 6] of the feature
        after parsing into a SparseTensor.
    """
    example = _make_sequence_example()
    columns = [
        fc._categorical_column_with_identity('int_ctx', num_buckets=100),
        fc._numeric_column('float_ctx'),
        col_fn(col_name, col_arg)
    ]
    context, seq_features = parsing_ops.parse_single_sequence_example(
        example.SerializeToString(),
        context_features=fc.make_parse_example_spec(columns[:2]),
        sequence_features=fc.make_parse_example_spec(columns[2:]))

    with self.cached_session() as sess:
      ctx_result, seq_result = sess.run([context, seq_features])
      self.assertEqual(list(seq_result[col_name].dense_shape), shape)
      self.assertEqual(
          list(seq_result[col_name].values[[0, 2, 6]]), values)
      self.assertEqual(list(ctx_result['int_ctx'].dense_shape), [1])
      self.assertEqual(ctx_result['int_ctx'].values[0], 5)
      self.assertEqual(list(ctx_result['float_ctx'].shape), [1])
      self.assertAlmostEqual(ctx_result['float_ctx'][0], 123.6, places=1)


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
