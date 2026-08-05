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
// RUN: tf-mlir-translate -mlir-to-graphdef %s -o - | FileCheck %s

func.func @main() {
  tf_executor.graph {
    %0:2 = tf_executor.island wraps "tf.RecvTPUEmbeddingActivations"() {config = "test_config_recv_embedding"} : () -> tensor<512x256xf32> loc("RecvTPUEmbedding")
    %1:1 = tf_executor.island wraps "tf.SendTPUEmbeddingGradients"(%0) {N = 1 : i64, NN = 0 : i64, config = "test_config_send_embedding", operandSegmentSizes = array<i32: 1, 0>} : (tensor<512x256xf32>) -> () loc("SendTPUEmbedding")
    tf_executor.fetch
  }
  func.return
}

// CHECK:       name: "RecvTPUEmbedding"
// CHECK-NEXT:  op: "RecvTPUEmbeddingActivations"
// CHECK-NEXT:  attr {
// CHECK-NEXT:    key: "_output_shapes"
// CHECK-NEXT:    value {
// CHECK-NEXT:      list {
// CHECK-NEXT:        shape {
// CHECK-NEXT:          dim {
// CHECK-NEXT:            size: 512
// CHECK-NEXT:          }
// CHECK-NEXT:          dim {
// CHECK-NEXT:            size: 256
// CHECK-NEXT:          }
// CHECK-NEXT:        }
// CHECK-NEXT:      }
// CHECK-NEXT:    }
// CHECK-NEXT:  }
// CHECK-NEXT:  attr {
// CHECK-NEXT:    key: "config"
// CHECK-NEXT:    value {
// CHECK-NEXT:      s: "test_config_recv_embedding"
// CHECK-NEXT:    }
// CHECK-NEXT:  }
// CHECK-NEXT:  attr {
// CHECK-NEXT:    key: "num_outputs"
// CHECK-NEXT:    value {
// CHECK-NEXT:      i: 1
// CHECK-NEXT:    }
// CHECK-NEXT:  }

// CHECK:       name: "SendTPUEmbedding"
// CHECK-NEXT:  op: "SendTPUEmbeddingGradients"
// CHECK-NEXT:  input: "RecvTPUEmbedding"
// CHECK-NEXT:  attr {
// CHECK-NEXT:    key: "N"
// CHECK-NEXT:    value {
// CHECK-NEXT:      i: 1
// CHECK-NEXT:    }
// CHECK-NEXT:  }
// CHECK-NEXT:  attr {
// CHECK-NEXT:    key: "NN"
// CHECK-NEXT:    value {
// CHECK-NEXT:      i: 0
// CHECK-NEXT:    }
// CHECK-NEXT:  }
// CHECK-NEXT:  attr {
// CHECK-NEXT:    key: "config"
// CHECK-NEXT:    value {
// CHECK-NEXT:      s: "test_config_send_embedding"
// CHECK-NEXT:    }
// CHECK-NEXT:  }
