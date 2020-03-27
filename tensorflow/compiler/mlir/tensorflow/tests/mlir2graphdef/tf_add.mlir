// RUN: tf-mlir-translate -mlir-to-graphdef %s -o - | FileCheck %s

func @main(%arg0: tensor<10xi32>, %arg1: tensor<10xi32>) -> tensor<10xi32>
attributes {tf.entry_function = {inputs = "input0,input1", outputs = "Add"}} {
  %graph = tf_executor.graph {
    %0:2 = tf_executor.island wraps "tf.Placeholder.input"(%arg0) {device = "", dtype = "tfdtype$DT_INT32", shape = "tfshape$dim { size: 10 }"} : (tensor<10xi32>) -> tensor<10xi32>
    %1:2 = tf_executor.island wraps "tf.Placeholder.input"(%arg1) {device = "", dtype = "tfdtype$DT_INT32", shape = "tfshape$dim { size: 10 }"} : (tensor<10xi32>) -> tensor<10xi32>
    %2:2 = tf_executor.island wraps "tf.Add"(%0#0, %1#0) {T = "tfdtype$DT_INT32", device = ""} : (tensor<10xi32>, tensor<10xi32>) -> tensor<10xi32> loc("Add")
    tf_executor.fetch %2 : tensor<10xi32>
  }
  return %graph : tensor<10xi32>
}

// CHECK:      node {
// CHECK-NEXT:   name: "input0"
// CHECK-NEXT:   op: "Placeholder"
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
// CHECK-NEXT:   name: "input1"
// CHECK-NEXT:   op: "Placeholder"
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
// CHECK-NEXT:   name: "Add"
// CHECK-NEXT:   op: "Add"
// CHECK-NEXT:   input: "input0"
// CHECK-NEXT:   input: "input1"
// CHECK-NEXT:   attr {
// CHECK-NEXT:     key: "T"
// CHECK-NEXT:     value {
// CHECK-NEXT:       type: DT_INT32
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   experimental_debug_info {
// CHECK-NEXT:   }
// CHECK-NEXT: }
// CHECK-NEXT: node {
// CHECK-NEXT:   name: "main"
// CHECK-NEXT:   op: "_Retval"
// CHECK-NEXT:   input: "Add"
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
