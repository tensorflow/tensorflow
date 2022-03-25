// RUN: tf-mlir-translate -mlir-to-graphdef %s -o - | FileCheck %s

func.func @main(%arg0: tensor<*x!tf_type.resource<tensor<8x1xf32>>>) -> tensor<8x1xf32> {
  %0 = tf_executor.graph {
     %outputs, %control = tf_executor.island wraps "tf.ReadVariableOp"(%arg0) : (tensor<*x!tf_type.resource<tensor<8x1xf32>>>) -> tensor<8x1xf32>
     tf_executor.fetch %outputs : tensor<8x1xf32>
  }
  func.return %0 : tensor<8x1xf32>
}

// Check that we generate _handle_dtypes and _handle_shapes for the resource
// argument.

// CHECK:      op: "_Arg"

// CHECK:        key: "_handle_dtypes"
// CHECK-NEXT:   value {
// CHECK-NEXT:     list {
// CHECK-NEXT:       type: DT_FLOAT
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT: }

// CHECK:        key: "_handle_shapes"
// CHECK-NEXT:   value {
// CHECK-NEXT:     list {
// CHECK-NEXT:       shape {
// CHECK-NEXT:         dim {
// CHECK-NEXT:           size: 8
// CHECK-NEXT:         }
// CHECK-NEXT:         dim {
// CHECK-NEXT:           size: 1
// CHECK-NEXT:         }
// CHECK-NEXT:       }
// CHECK-NEXT:     }

// CHECK:        key: "index"
// CHECK-NEXT:   value {
// CHECK-NEXT:     i: 0
// CHECK-NEXT:   }
// CHECK-NEXT: }

