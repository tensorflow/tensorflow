// RUN: tf-mlir-translate -mlir-to-graphdef %s -o - | FileCheck %s

func @main() {
  tf_executor.graph {
    %0:2 = tf_executor.island wraps "tf.RecvTPUEmbeddingActivations"() {config = "test_config_recv_embedding"} : () -> tensor<512x256xf32> loc("RecvTPUEmbedding")
    %1:1 = tf_executor.island wraps "tf.SendTPUEmbeddingGradients"(%0) {N = 1 : i64, NN = 0 : i64, config = "test_config_send_embedding", operand_segment_sizes = dense<[1, 0]> : vector<2xi32>} : (tensor<512x256xf32>) -> () loc("SendTPUEmbedding")
    tf_executor.fetch
  }
  return
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
