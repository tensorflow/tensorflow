// RUN: tf-opt -tfl-prepare-composite-funcs-tf %s | FileCheck %s --dump-input-on-failure

func @foo(%arg0: tensor<*xf32>, %arg1: tensor<*xi32>) -> tensor<*xf32> attributes  {tf._implements = "embedding_matmul", tf._reference = "mlir"} {
  %0 = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
  %1 = "tf.ExpandDims"(%arg1, %0) : (tensor<*xi32>, tensor<i32>) -> tensor<*xi32>
  %2 = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
  %3 = "tf.Const"() {value = dense<4096> : tensor<i32>} : () -> tensor<i32>
  %4 = "tf.Const"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
  %5 = "tf.Range"(%4, %3, %2) : (tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<4096xi32>
  %6 = "tf.Equal"(%1, %5) : (tensor<*xi32>, tensor<4096xi32>) -> tensor<*xi1>
  %7 = "tf.Cast"(%6) : (tensor<*xi1>) -> tensor<*xf32>
  %8 = "tf.BatchMatMulV2"(%7, %arg0) {adj_x = false, adj_y = false} : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
  return %8 : tensor<*xf32>
}

// CHECK:       func @foo([[VAL_0:%.*]]: tensor<*xf32>, [[VAL_1:%.*]]: tensor<*xi32>) -> tensor<*xf32>
// CHECK:        attributes  {tf._implements = "fused_tfl_embedding_lookup", tf._reference = "mlir"}
// CHECK:           [[VAL_2:%.*]] = "tfl.embedding_lookup"([[VAL_1]], [[VAL_0]]) : (tensor<*xi32>, tensor<*xf32>) -> tensor<*xf32>
// CHECK:           return [[VAL_2]] : tensor<*xf32>
