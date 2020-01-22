// RUN: tf-opt -tfl-prepare-composite-funcs-tf %s | FileCheck %s --dump-input-on-failure

func @foo(%arg0: tensor<?xf32>, %arg1: tensor<?xi32>) -> tensor<?xf32> attributes  {tf._implements = "embedding_matmul", tf._reference = "mlir"} {
  %0 = "tf.Fill" (%arg1, %arg0) : (tensor<? x i32>, tensor<? x f32>) -> tensor<? x f32>
  %1 = "tf.MatMul" (%0, %arg0) : (tensor<? x f32>, tensor<? x f32>) -> tensor<? x f32>
  return %1 : tensor<?xf32>
}

// CHECK:       func @foo([[VAL_0:%.*]]: tensor<?xf32>, [[VAL_1:%.*]]: tensor<?xi32>) -> tensor<?xf32>
// CHECK:        attributes  {tf._implements = "fused_tfl_embedding_lookup", tf._reference = "mlir"}
// CHECK:           [[VAL_2:%.*]] = "tfl.embedding_lookup"([[VAL_1]], [[VAL_0]]) : (tensor<?xi32>, tensor<?xf32>) -> tensor<?xf32>
// CHECK:           return [[VAL_2]] : tensor<?xf32>