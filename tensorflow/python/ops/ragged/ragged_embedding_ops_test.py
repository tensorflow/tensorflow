# Copyright 2026 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for ragged_embedding_ops."""

from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import test_util
from tensorflow.python.ops import variables
from tensorflow.python.ops.ragged import ragged_embedding_ops
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.platform import googletest


class RaggedEmbeddingOpsTest(test_util.TensorFlowTestCase):

  def testEmbeddingLookupSparseMismatchedTypesErrorMessage(self):
    # Regression test: the error message's `received {{type(sp_ids)...}}`
    # f-string used doubled braces, which f-strings treat as literal `{`
    # and `}` characters rather than substitution syntax. As a result the
    # raised TypeError displayed the literal text
    # "{type(sp_ids).__name__!r}" instead of the actual type name, which
    # is useless for debugging a real type mismatch.
    params = variables.Variable([[1.0, 2.0], [3.0, 4.0]])
    sp_ids = ragged_factory_ops.constant([[1], [0]])
    sp_weights = sparse_tensor.SparseTensor(
        indices=[[0, 0]], values=[1.0], dense_shape=[1, 1])

    with self.assertRaises(TypeError) as cm:
      ragged_embedding_ops.embedding_lookup_sparse(params, sp_ids,
                                                    sp_weights)

    message = str(cm.exception)
    # The substituted type names must appear in the message...
    self.assertIn('RaggedTensor', message)
    self.assertIn('SparseTensor', message)
    # ...and the unsubstituted placeholder text must not.
    self.assertNotIn('{type(sp_ids)', message)
    self.assertNotIn('{type(sp_weights)', message)


if __name__ == '__main__':
  googletest.main()
