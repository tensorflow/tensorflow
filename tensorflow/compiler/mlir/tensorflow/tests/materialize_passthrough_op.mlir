// RUN: tf-opt -allow-unregistered-dialect -tf-materialize-passthrough-op %s  | FileCheck %s


// Check that the MlirPassthroughOp is eliminated and replaced by its attached
// MLIR module.

// CHECK-LABEL: func @main
func.func @main(%arg0 : tensor<10xf32>, %arg1 : tensor<10xf32>) -> tensor<10x10xf32> {
// CHECK-SAME: (%[[ARG0:.*]]: tensor<10xf32>, %[[ARG1:.*]]: tensor<10xf32>)
// CHECK-NEXT:    %[[ADD:.*]] = "tf.Add"(%[[ARG0]], %[[ARG1]]) : (tensor<10xf32>, tensor<10xf32>) -> tensor<10xf32>
// CHECK-NEXT:    %[[MAGIC:.*]] = "magic.op"(%[[ADD]], %[[ADD]]) : (tensor<10xf32>, tensor<10xf32>) -> tensor<10x10xf32>
// CHECK-NEXT:    return %[[MAGIC]]
  %0 = "tf.MlirPassthroughOp"(%arg0, %arg1) {Tinputs = ["tfdtype$DT_FLOAT", "tfdtype$DT_FLOAT"], Toutputs = ["tfdtype$DT_FLOAT"], device = "", mlir_module = "\0Afunc.func @main(%arg0 : tensor<10xf32>, %arg1 : tensor<10xf32>) -> tensor<10x10xf32> {\0A   %add = \22tf.Add\22(%arg0, %arg1) : (tensor<10xf32>, tensor<10xf32>) -> tensor<10xf32>\0A   %ret = \22magic.op\22(%add, %add) : (tensor<10xf32>, tensor<10xf32>) -> tensor<10x10xf32>\0A   func.return %ret : tensor<10x10xf32>\0A}\0A", name = "MlirPassthroughOp"} : (tensor<10xf32>, tensor<10xf32>) -> tensor<10x10xf32>
  func.return %0 : tensor<10x10xf32>
}
