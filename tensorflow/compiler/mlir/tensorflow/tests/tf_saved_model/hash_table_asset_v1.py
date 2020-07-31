# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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

# RUN: %p/hash_table_asset_v1| FileCheck %s

# pylint: disable=missing-docstring,line-too-long
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tempfile

import tensorflow.compat.v1 as tf
from tensorflow.compiler.mlir.tensorflow.tests.tf_saved_model import common_v1

# CHECK: "tf_saved_model.session_initializer"() {initializer = [[init:@.*]]} : () -> ()
# CHECK: "tf_saved_model.asset"() {filename = {{.*}}, sym_name = "[[asset:.*]]"}

# CHECK:      func [[init]]
# CHECK-SAME: [[ARG:%.*]]: tensor<!tf.string> {tf_saved_model.bound_input = @[[asset]]}
# CHECK-NEXT: [[R0:%.*]] = "tf.HashTableV2"()
# CHECK-SAME: shared_name = "[[hash_table:.*]]"
# CHECK-NEXT: "tf.InitializeTableFromTextFileV2"([[R0]], [[ARG]])


def write_vocabulary_file(vocabulary):
  """Write temporary vocab file for module construction."""
  tmpdir = tempfile.mkdtemp()
  vocabulary_file = os.path.join(tmpdir, 'tokens.txt')
  with tf.io.gfile.GFile(vocabulary_file, 'w') as f:
    for entry in vocabulary:
      f.write(entry + '\n')
  return vocabulary_file


def test():

  table_initializer = tf.lookup.TextFileInitializer(
      write_vocabulary_file(['cat', 'is', 'on', 'the', 'mat']), tf.string,
      tf.lookup.TextFileIndex.WHOLE_LINE, tf.int64,
      tf.lookup.TextFileIndex.LINE_NUMBER)
  table = tf.lookup.StaticVocabularyTable(table_initializer, num_oov_buckets=10)

  x = tf.placeholder(tf.string, shape=(), name='input')
  r = table.lookup(x)

  tensor_info_x = tf.compat.v1.saved_model.utils.build_tensor_info(x)
  tensor_info_r = tf.compat.v1.saved_model.utils.build_tensor_info(r)

  return {
      'key': (tf.compat.v1.saved_model.signature_def_utils.build_signature_def(
          inputs={'x': tensor_info_x},
          outputs={'r': tensor_info_r},
          method_name='some_function'))
  }, tf.tables_initializer(), tf.get_collection(tf.GraphKeys.ASSET_FILEPATHS)


if __name__ == '__main__':
  common_v1.set_tf_options()
  common_v1.do_test(test)
