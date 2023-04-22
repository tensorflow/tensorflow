// RUN: tf-mlir-translate -mlir-tf-graph-to-hlo-text %s -tf-input-shapes=2:2 -tf-input-data-types=DT_FLOAT,DT_FLOAT -tf-xla-input-types=parameter,resource -emit-return-tuple | FileCheck %s

module attributes {tf.versions = {producer = 511 : i32}} {
  func @main(%arg0: tensor<*xf32>, %arg1: tensor<*x!tf.resource>) {
    tf_executor.graph {
      %control = tf_executor.island wraps "tf.AssignVariableOp"(%arg1, %arg0) : (tensor<*x!tf.resource>, tensor<*xf32>) -> ()
      tf_executor.fetch %control : !tf_executor.control
    }
    return
  }
}

// Tests a conversion from Graph (tf_executor dialect MLIR) to MLIR with
// resource arguments.

// CHECK-LABEL: HloModule main.{{[0-9]+}}, input_output_alias={ {0}: (1, {}, may-alias) }
// CHECK:       ENTRY %main.{{[0-9]+}} ([[ARG0:.*]]: f32[2], [[ARG1:.*]]: f32[2]) -> (f32[2]) {
// CHECK-NEXT:    %[[ARG1]] = f32[2]{0} parameter(1)
// CHECK-NEXT:    %[[ARG0]] = f32[2]{0} parameter(0)
// CHECK-NEXT:    ROOT %tuple.{{[0-9]+}} = (f32[2]{0}) tuple(f32[2]{0} %[[ARG0]])
// CHECK-NEXT:  }

// CHECK:       // InputMapping {0, 1}
// CHECK-NEXT:  // XlaInputShape f32[2]
// CHECK-NEXT:  // XlaInputShape f32[2]
// CHECK-NEXT:  // XlaOutputShape (f32[2])
// CHECK-NEXT:  // ResourceUpdate input_index=1 type=float shape=(2) modified
