// RUN: mlir-hlo-opt -hlo-legalize-to-lhlo -canonicalize %s | FileCheck %s

// CHECK-LABEL: func @func_op_unranked_arg_result
func @func_op_unranked_arg_result(%arg0: tensor<*xf32>) -> tensor<*xf32> {
  return %arg0 : tensor<*xf32>
}
// CHECK-SAME: ([[ARG:%.*]]: memref<*xf32>) -> memref<*xf32>
// CHECK-NEXT: return [[ARG]] : memref<*xf32>
