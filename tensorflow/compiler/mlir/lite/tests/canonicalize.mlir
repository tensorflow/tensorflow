// RUN: tf-opt -canonicalize %s | FileCheck %s

// Checks that tfl.reshape should be removed if its output's only user is
// another tfl.reshape
func @reshape_removeAdjacent(tensor<4x4x4xf32>) -> tensor<64xf32> {
^bb0(%arg0: tensor<4x4x4xf32>) :
  %0 = "tfl.reshape"(%arg0) : (tensor<4x4x4xf32>) -> tensor<16x4xf32>
  %1 = "tfl.reshape"(%0) : (tensor<16x4xf32>) -> tensor<64xf32>
  return %1 : tensor<64xf32>

// CHECK-LABEL: func @reshape_removeAdjacent
// CHECK:  %0 = "tfl.reshape"(%arg0) : (tensor<4x4x4xf32>) -> tensor<64xf32>
// CHECK:  return
}

// Checks that tfl.reshape should be removed if its output has more than one
// user but all users are tfl.reshape
func @reshape_removeAdjacentWithMultipleUse(tensor<4x4x4xf32>) -> tensor<64xf32> {
^bb0(%arg0: tensor<4x4x4xf32>) :
  %0 = "tfl.reshape"(%arg0) : (tensor<4x4x4xf32>) -> tensor<16x4xf32>
  %1 = "tfl.reshape"(%0) : (tensor<16x4xf32>) -> tensor<64xf32>
  %2 = "tfl.reshape"(%0) : (tensor<16x4xf32>) -> tensor<64xf32>
  %3 = addf %1, %2 : tensor<64xf32>
  return %3 : tensor<64xf32>

// CHECK-LABEL: func @reshape_removeAdjacentWithMultipleUse
// CHECK:  %0 = "tfl.reshape"(%arg0) : (tensor<4x4x4xf32>) -> tensor<64xf32>
// CHECK:  %1 = "tfl.reshape"(%arg0) : (tensor<4x4x4xf32>) -> tensor<64xf32>
// CHECK:  %2 = addf %0, %1
// CHECK:  return %2
}

// Checks that tfl.reshape should be kept if its output has more than one
// user and not all users are tfl.reshape
func @reshape_keepAdjacentWithMultipleUse(tensor<4x4x4xf32>) -> (tensor<16x4xf32>, tensor<64xf32>) {
^bb0(%arg0: tensor<4x4x4xf32>) :
  %0 = "tfl.reshape"(%arg0) : (tensor<4x4x4xf32>) -> tensor<16x4xf32>
  %1 = "tfl.reshape"(%0) : (tensor<16x4xf32>) -> tensor<64xf32>
  return %0, %1 : tensor<16x4xf32>, tensor<64xf32>

// CHECK-LABEL: func @reshape_keepAdjacentWithMultipleUse
// CHECK:  %0 = "tfl.reshape"(%arg0) : (tensor<4x4x4xf32>) -> tensor<16x4xf32>
// CHECK:  %1 = "tfl.reshape"(%arg0) : (tensor<4x4x4xf32>) -> tensor<64xf32>
// CHECK:  return
}

// Checks that tfl.reshape should be removed if its output type is the same
// as its input type
func @reshape_removeIdentity(tensor<4x4x4xf32>) -> tensor<4x4x4xf32> {
^bb0(%arg0: tensor<4x4x4xf32>) :
  %0 = "tfl.reshape"(%arg0) : (tensor<4x4x4xf32>) -> tensor<4x4x4xf32>
  return %0 : tensor<4x4x4xf32>

// CHECK-LABEL: func @reshape_removeIdentity
// CHECK:  return %arg0 : tensor<4x4x4xf32>
}

// Checks that tfl.fake_quant should be removed if all its users have valid
// "minmax" attributes.
func @fakequant_dropfakequant(tensor<i32>, f32, f32) -> tensor<i32> {
^bb0(%arg0: tensor<i32>, %arg1: f32, %arg2: f32):
  %0 = "tfl.fake_quant"(%arg0) {name = 0, minmax = [0.1, 0.2], num_bits = 4 : i32, narrow_range = false} : (tensor<i32>) -> tensor<i32>
  %1 = tfl.pow %arg0, %0 {minmax = [0.4, 0.6]} : tensor<i32>
  %2 = tfl.pow %1, %0 {minmax = [0.5, 0.7]} : tensor<i32>
  return %2 : tensor<i32>

// CHECK-LABEL: fakequant_dropfakequant
// CHECK-NEXT:    %0 = tfl.pow %arg0, %arg0 {minmax = [4.000000e-01, 6.000000e-01]} : tensor<i32>
// CHECK-NEXT:    %1 = tfl.pow %0, %arg0 {minmax = [5.000000e-01, 0.69999999999999996]} : tensor<i32>

// CHECK-NEXT:    return %1 : tensor<i32>
}

// Checks that tfl.fake_quant should not be removed if some of its users or
// itself don't have valid "minmax" attributes.
func @fakequant_notdropfakequant(tensor<i32>, f32, f32) -> tensor<i32> {
^bb0(%arg0: tensor<i32>, %arg1: f32, %arg2: f32):
  %0 = "tfl.fake_quant"(%arg0) {name = 0, minmax = [], num_bits = 4 : i32, narrow_range = false} : (tensor<i32>) -> tensor<i32>
  %1 = tfl.pow %arg0, %0 : tensor<i32>
  %2 = tfl.pow %1, %0 : tensor<i32>

  %5 = "tfl.fake_quant"(%arg0) {name = 1, minmax = [0.1, 0.2], num_bits = 4 : i32, narrow_range = false} : (tensor<i32>) -> tensor<i32>
  %6 = tfl.pow %arg0, %5 : tensor<i32>
  %7 = tfl.pow %6, %5 : tensor<i32>

  %11 = addi %2, %7 : tensor<i32>
  return %11 : tensor<i32>

// CHECK-LABEL: fakequant_notdropfakequant
// CHECK:  %0 = "tfl.fake_quant"(%arg0) {minmax = [], name = 0 : i64, narrow_range = false, num_bits = 4 : i32} : (tensor<i32>) -> tensor<i32>
// CHECK:  %3 = "tfl.fake_quant"(%arg0) {minmax = [1.000000e-01, 2.000000e-01], name = 1 : i64, narrow_range = false, num_bits = 4 : i32} : (tensor<i32>) -> tensor<i32>
}
