// RUN: xla-opt %s -split-input-file \
// RUN: -custom-call-lower-to-triton \
// RUN: | FileCheck %s

func.func private @scaled_dot(%arg0: tensor<128x128xf8E5M2>, %arg1: tensor<128x256xf8E5M2>, %arg2: tensor<128x256xf32>, %arg3: tensor<128x4xf8E8M0FNU>, %arg4: tensor<4x256xf8E8M0FNU>) -> tensor<128x256xf32>
// CHECK: func.func @scaled_dot_custom_call_is_lowered(%[[ARG0:.*]]: tensor<128x128xf8E5M2>, %[[ARG1:.*]]: tensor<128x256xf8E5M2>, %[[ARG2:.*]]: tensor<128x256xf32>, %[[ARG3:.*]]: tensor<128x4xf8E8M0FNU>, %[[ARG4:.*]]: tensor<4x256xf8E8M0FNU>) -> tensor<128x256xf32>
func.func @scaled_dot_custom_call_is_lowered(%arg0: tensor<128x128xf8E5M2>, %arg1: tensor<128x256xf8E5M2>, %arg2: tensor<128x256xf32>, %arg3: tensor<128x4xf8E8M0FNU>, %arg4: tensor<4x256xf8E8M0FNU>) -> tensor<128x256xf32> {
  // CHECK: %[[BITCAST0:.*]] = arith.bitcast %[[ARG3]] : tensor<128x4xf8E8M0FNU> to tensor<128x4xi8>
  // CHECK: %[[BITCAST1:.*]] = arith.bitcast %[[ARG4]] : tensor<4x256xf8E8M0FNU> to tensor<4x256xi8>
  // CHECK: %[[TRANSPOSE:.*]] = tt.trans %[[BITCAST1]] {order = array<i32: 1, 0>} : tensor<4x256xi8> -> tensor<256x4xi8>
  // CHECK: %[[RESULT:.*]] = tt.dot_scaled %[[ARG0]] scale %[[BITCAST0]], %[[ARG1]] scale %[[TRANSPOSE]], %[[ARG2]] lhs = e5m2 rhs = e5m2 {fastMath = true} : tensor<128x128xf8E5M2>, tensor<128x4xi8> * tensor<128x256xf8E5M2>, tensor<256x4xi8> -> tensor<128x256xf32>
  %0 = "func.call"(%arg0, %arg1, %arg2, %arg3, %arg4) <{callee = @scaled_dot}> {fast_math = true} : (tensor<128x128xf8E5M2>, tensor<128x256xf8E5M2>, tensor<128x256xf32>, tensor<128x4xf8E8M0FNU>, tensor<4x256xf8E8M0FNU>) -> tensor<128x256xf32>

  return %0 : tensor<128x256xf32>
}