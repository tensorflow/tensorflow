// Copyright 2026 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ==============================================================================
// RUN: not tf-mlir-translate -split-input-file -mlir-to-graphdef %s -o - 2>&1 | FileCheck %s

// Tests function with multiple blocks.

func.func @main() {
  ^bb:
    cf.br ^bb1
  ^bb1:
    func.return
}

// CHECK: functions must be of a single Graph with single op Islands: only single block functions are supported

// -----

// Tests invalid functions for exporting to Graph/GraphDef.

func.func @main() {
  func.return
}

// CHECK: functions must be of a single Graph with single op Islands: first op in function is not a tf_executor.graph

// -----

func.func @main() {
  tf_executor.graph {
    tf_executor.fetch
  }
  tf_executor.graph {
    tf_executor.fetch
  }
  func.return
}

// CHECK: functions must be of a single Graph with single op Islands: function does not only contain a single tf_executor.graph

// -----

func.func @main() {
  tf_executor.graph {
    %0 = tf_executor.island {
      tf_executor.yield
    }
    tf_executor.fetch
  }
  func.return
}

// CHECK: functions must be of a single Graph with single op Islands: tf_executor.island must perfectly wrap a single op

// -----

func.func @main() {
  tf_executor.graph {
    %0 = tf_executor.island {
      %1 = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
      %2 = "tf.Const"() {value = dense<2> : tensor<i32>} : () -> tensor<i32>
      tf_executor.yield
    }
    tf_executor.fetch
  }
  func.return
}

// CHECK: functions must be of a single Graph with single op Islands: tf_executor.island must perfectly wrap a single op

// -----

func.func @main() {
  tf_executor.graph {
    %0 = tf_executor.island {
      %1 = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
      tf_executor.yield
    }
    tf_executor.fetch
  }
  func.return
}

// CHECK: functions must be of a single Graph with single op Islands: tf_executor.island must perfectly wrap a single op

// -----

func.func @main(%arg0: tensor<i32>, %arg1: tensor<i32>) {
  tf_executor.graph {
    %0:3 = tf_executor.island {
      %1:2 = "tf.IdentityN"(%arg0, %arg1) : (tensor<i32>, tensor<i32>) -> (tensor<i32>, tensor<i32>)
      tf_executor.yield %1#1, %1#0 : tensor<i32>, tensor<i32>
    }
    tf_executor.fetch
  }
  func.return
}

// CHECK: functions must be of a single Graph with single op Islands: tf_executor.island must perfectly wrap a single op
