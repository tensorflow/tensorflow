// RUN: fusion_compiler_opt %s -xtile-cpu-tensor-ops-to-bufferizable -split-input-file | FileCheck %s


func.func @bitcast(%arg0 : tensor<8xf32>) -> tensor<8xi32> {
  // CHECK: %[[RESULT:.*]] = arith.bitcast %arg0 : tensor<8xf32> to tensor<8xi32>
  %result = arith.bitcast %arg0 : tensor<8xf32> to tensor<8xi32>
  // CHECK: return %[[RESULT]] : tensor<8xi32>
  return %result : tensor<8xi32>
}
