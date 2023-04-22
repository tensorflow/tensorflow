// RUN: tf-mlir-translate -mlir-to-graphdef %s -o - | FileCheck %s

func @main(%arg0: tensor<10xi32>) -> tensor<10xi32>
attributes {tf.entry_function = {inputs = "input0", outputs = "output0"}} {
  %graph = tf_executor.graph {
    tf_executor.fetch %arg0 : tensor<10xi32>
  }
  return %graph : tensor<10xi32>
}

// CHECK:      node {
// CHECK-NEXT:   name: "input0"
// CHECK-NEXT:   op: "_Arg"
// CHECK:          key: "T"
// CHECK-NEXT:     value {
// CHECK-NEXT:       type: DT_INT32
// CHECK-NEXT:     }
// CHECK:          key: "_output_shapes"
// CHECK-NEXT:     value {
// CHECK-NEXT:      shape {
// CHECK-NEXT:        dim {
// CHECK-NEXT:          size: 10
// CHECK-NEXT:        }
// CHECK-NEXT:      }
// CHECK-NEXT:   }
// CHECK:        name: "output0"
// CHECK-NEXT:   op: "_Retval"
// CHECK-NEXT:   input: "input0"
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
