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

# RUN: %p/dag_object_graph | FileCheck %s

# pylint: disable=missing-docstring,line-too-long
import tensorflow.compat.v2 as tf
from tensorflow.compiler.mlir.tensorflow.tests.tf_saved_model import common


class Child(tf.Module):

  def __init__(self):
    super(Child, self).__init__()
    self.my_variable = tf.Variable(3.)


# Creates a dag object graph.
# There is only one instance of `Child`, but it is reachable via two names.
# Thus, self.my_variable is reachable via two paths.
class TestModule(tf.Module):

  def __init__(self):
    super(TestModule, self).__init__()
    self.child1 = Child()
    self.child2 = self.child1

  # CHECK: tf_saved_model.global_tensor
  # CHECK-SAME: tf_saved_model.exported_names = ["child1.my_variable", "child2.my_variable"]


if __name__ == '__main__':
  common.do_test(TestModule)
