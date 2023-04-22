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

from tensorflow.core.example import example_pb2
from tensorflow.python.feature_column import feature_column_v2 as fc
from tensorflow.python.feature_column import sequence_feature_column as sfc
from tensorflow.python.ops import parsing_ops
from tensorflow.python.platform import test
from tensorflow.python.util import compat


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
        fc.categorical_column_with_identity('int_ctx', num_buckets=100),
        fc.numeric_column('float_ctx'),
        col_fn(col_name, col_arg)
    ]
    context, seq_features = parsing_ops.parse_single_sequence_example(
        example.SerializeToString(),
        context_features=fc.make_parse_example_spec_v2(columns[:2]),
        sequence_features=fc.make_parse_example_spec_v2(columns[2:]))

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
