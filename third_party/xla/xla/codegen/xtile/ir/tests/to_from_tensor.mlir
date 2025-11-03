// RUN: emitters_opt %s  -canonicalize | FileCheck %s

// CHECK-LABEL: @to_scalar_roundtrip
func.func @to_scalar_roundtrip(%arg0: tensor<i32>) -> tensor<i32> {
  %0 = xtile.to_scalar %arg0 : tensor<i32>
  %1 = xtile.to_tensor %0 : i32
  // CHECK: return %arg0 : tensor<i32>
  return %1 : tensor<i32>
}

// CHECK-LABEL: @to_tensor_roundtrip
func.func @to_tensor_roundtrip(%arg0: i32) -> i32 {
  %0 = xtile.to_tensor %arg0 : i32
  %1 = xtile.to_scalar %0 : tensor<i32>
  // CHECK: return %arg0 : i32
  return %1 : i32
}
