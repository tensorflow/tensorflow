// RUN: dtensor-opt %s -split-input-file -dtensor-embedding-checkpoint -mlir-print-ir-after-all | FileCheck %s --dump-input=fail

// Check load embedding function is created.
// CHECK-LABEL: func @load_embedding_fn
// CHECK-SAME:  %[[ARG0:[a-z0-9]*]]: tensor<*x!tf_type.resource<tensor<8x4xf32>>> {tf._tpu_embedding_slot_id = 0 : i64, tf._tpu_embedding_table_id = 0 : i64}
// CHECK-SAME:  %[[ARG1:[a-z0-9]*]]: tensor<*x!tf_type.resource<tensor<8x4xf32>>> {tf._tpu_embedding_slot_id = 1 : i64, tf._tpu_embedding_table_id = 0 : i64}
// CHECK:         %0 = "tf.ReadVariableOp"(%arg0)
// CHECK-NEXT:    %1 = "tf.ReadVariableOp"(%arg1)
// CHECK-NEXT:    %cst = "tf.Const"()
// CHECK-NEXT:    "tf.LoadAllTPUEmbeddingParameters"
// CHECK-NEXT:    return
func.func @main(%arg0: tensor<i32>,
  %arg1: tensor<*x!tf_type.resource<tensor<8x4xf32>>> {tf._tpu_embedding_slot_id = 0 : i64, tf._tpu_embedding_table_id = 0 : i64},
  %arg2: tensor<*x!tf_type.resource<tensor<8x4xf32>>> {tf._tpu_embedding_slot_id = 1 : i64, tf._tpu_embedding_table_id = 0 : i64},
  %arg3: tensor<*x!tf_type.resource<tensor<4xi32>>>
) -> () attributes {tf._tpu_embedding_configuration = "dummy configuration string"} {
  func.return
}
