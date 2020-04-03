// RUN: tf-mlir-translate -mlir-to-graphdef %s -o - | FileCheck %s

func @main(%arg0: tensor<10xi32>) -> tensor<10xi32>
attributes {tf.entry_function = {inputs = "input0", outputs = "Placeholder"}} {
  %graph = tf_executor.graph {
    %0:2 = tf_executor.island wraps "tf.Placeholder.input"(%arg0) {device = "", dtype = "tfdtype$DT_INT32", shape = "tfshape$dim { size: 10 }"} : (tensor<10xi32>) -> tensor<10xi32>
    tf_executor.fetch %0 : tensor<10xi32>
  }
  return %graph : tensor<10xi32>
}

// CHECK:      node {
// CHECK-NEXT:   name: "Placeholder"
// CHECK-NEXT:   op: "Placeholder"
// CHECK-NEXT:   attr {
// CHECK-NEXT:     key: "_output_shapes"
// CHECK-NEXT:     value {
// CHECK-NEXT:       list {
// CHECK-NEXT:         shape {
// CHECK-NEXT:           dim {
// CHECK-NEXT:             size: 10
// CHECK-NEXT:           }
// CHECK-NEXT:         }
// CHECK-NEXT:       }
// CHECK-NEXT:      }
// CHECK-NEXT:   }
// CHECK-NEXT:   attr {
// CHECK-NEXT:     key: "dtype"
// CHECK-NEXT:     value {
// CHECK-NEXT:       type: DT_INT32
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   attr {
// CHECK-NEXT:     key: "shape"
// CHECK-NEXT:     value {
// CHECK-NEXT:       shape {
// CHECK-NEXT:         dim {
// CHECK-NEXT:           size: 10
// CHECK-NEXT:         }
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   experimental_debug_info {
// CHECK-NEXT:   }
// CHECK-NEXT: }
// CHECK-NEXT: node {
// CHECK-NEXT:   name: "main"
// CHECK-NEXT:   op: "_Retval"
// CHECK-NEXT:   input: "Placeholder"
// CHECK-NEXT:   attr {
// CHECK-NEXT:     key: "T"
// CHECK-NEXT:     value {
// CHECK-NEXT:       type: DT_INT32
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   attr {
// CHECK-NEXT:     key: "index"
// CHECK-NEXT:     value {
// CHECK-NEXT:       i: 0
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT: }
// CHECK-NEXT: library {
// CHECK-NEXT: }
