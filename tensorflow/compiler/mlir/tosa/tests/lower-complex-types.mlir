// RUN: tf-opt --split-input-file --tosa-lower-complex-types --verify-each %s | FileCheck %s

// CHECK-LABEL: test_complex_input
// CHECK-SAME: %[[VAL_0:.*]]: tensor<1x4x4x2xf32>
// CHECK: return %[[VAL_0]] : tensor<1x4x4x2xf32>
func.func @test_complex_input(%arg0: tensor<1x4x4xcomplex<f32>>) -> (tensor<1x4x4x2xf32>) {
  %0 = builtin.unrealized_conversion_cast %arg0 : tensor<1x4x4xcomplex<f32>> to tensor<1x4x4x2xf32>
  return %0 : tensor<1x4x4x2xf32>
}

// -----

// CHECK-LABEL: test_complex_output
// CHECK-SAME: %[[VAL_0:.*]]: tensor<1x4x4x2xf32>
// CHECK: return %[[VAL_0]] : tensor<1x4x4x2xf32>
func.func @test_complex_output(%arg0: tensor<1x4x4x2xf32>) -> (tensor<1x4x4xcomplex<f32>>) {
  %0 = builtin.unrealized_conversion_cast %arg0 : tensor<1x4x4x2xf32> to tensor<1x4x4xcomplex<f32>>
  return %0 : tensor<1x4x4xcomplex<f32>>
}

// -----

// CHECK-LABEL: test_mixed_input
// CHECK-SAME: %[[VAL_0:.*]]: tensor<1x4x4x2xf32>, %[[VAL_1:.*]]: tensor<1x4x4x2xf32>, %[[VAL_2:.*]]: tensor<1x4x4xf32>
// CHECK: return %[[VAL_0]], %[[VAL_1]], %[[VAL_2]]  : tensor<1x4x4x2xf32>, tensor<1x4x4x2xf32>, tensor<1x4x4xf32>
func.func @test_mixed_input(%arg0: tensor<1x4x4xcomplex<f32>>, %arg1: tensor<1x4x4xcomplex<f32>>, %arg2: tensor<1x4x4xf32>)
    -> (tensor<1x4x4x2xf32>, tensor<1x4x4x2xf32>, tensor<1x4x4xf32>) {
  %0 = builtin.unrealized_conversion_cast %arg0 : tensor<1x4x4xcomplex<f32>> to tensor<1x4x4x2xf32>
  %1 = builtin.unrealized_conversion_cast %arg1 : tensor<1x4x4xcomplex<f32>> to tensor<1x4x4x2xf32>
  return %0, %1, %arg2 : tensor<1x4x4x2xf32>, tensor<1x4x4x2xf32>, tensor<1x4x4xf32>
}

// -----

// CHECK-LABEL: test_mixed_output
// CHECK-SAME: %[[VAL_0:.*]]: tensor<1x4x4x2xf32>, %[[VAL_1:.*]]: tensor<1x4x4xf32>
// CHECK: return %[[VAL_0]], %[[VAL_1]] : tensor<1x4x4x2xf32>, tensor<1x4x4xf32>
func.func @test_mixed_output(%arg0: tensor<1x4x4x2xf32>, %arg1: tensor<1x4x4xf32>)
    -> (tensor<1x4x4xcomplex<f32>>, tensor<1x4x4xf32>) {
  %0 = builtin.unrealized_conversion_cast %arg0 : tensor<1x4x4x2xf32> to tensor<1x4x4xcomplex<f32>>
  return %0, %arg1 : tensor<1x4x4xcomplex<f32>>, tensor<1x4x4xf32>
}

// -----

// CHECK-LABEL: test_complex_input_output_op
// CHECK-SAME: %[[VAL_0:.*]]: tensor<1x4x4x2xf32>
// CHECK: %[[VAL_1:.*]] = "tosa.slice"(%[[VAL_0]]) {size = array<i64: 1, 4, 4, 1>, start = array<i64: 0, 0, 0, 0>} : (tensor<1x4x4x2xf32>) -> tensor<1x4x4x1xf32>
// CHECK: %[[VAL_2:.*]] = "tosa.reshape"(%[[VAL_1]]) {new_shape = array<i64: 1, 4, 4>} : (tensor<1x4x4x1xf32>) -> tensor<1x4x4xf32>
// CHECK: %[[VAL_3:.*]], %[[VAL_4:.*]] = "tosa.rfft2d"(%[[VAL_2]]) : (tensor<1x4x4xf32>) -> (tensor<1x4x3xf32>, tensor<1x4x3xf32>)
// CHECK: %[[VAL_5:.*]] = "tosa.reshape"(%[[VAL_3]]) {new_shape = array<i64: 1, 4, 3, 1>} : (tensor<1x4x3xf32>) -> tensor<1x4x3x1xf32>
// CHECK: %[[VAL_6:.*]] = "tosa.reshape"(%[[VAL_4]]) {new_shape = array<i64: 1, 4, 3, 1>} : (tensor<1x4x3xf32>) -> tensor<1x4x3x1xf32>
// CHECK: %[[VAL_7:.*]] = "tosa.concat"(%[[VAL_5]], %[[VAL_6]]) {axis = 3 : i64} : (tensor<1x4x3x1xf32>, tensor<1x4x3x1xf32>) -> tensor<1x4x3x2xf32>
// CHECK: return %[[VAL_7]] : tensor<1x4x3x2xf32>
func.func @test_complex_input_output_op(%arg0: tensor<1x4x4xcomplex<f32>>) -> (tensor<1x4x3xcomplex<f32>>) {
    %0 = builtin.unrealized_conversion_cast %arg0 : tensor<1x4x4xcomplex<f32>> to tensor<1x4x4x2xf32>
    %1 = "tosa.slice"(%0) {size = array<i64: 1, 4, 4, 1>, start = array<i64: 0, 0, 0, 0>} : (tensor<1x4x4x2xf32>) -> tensor<1x4x4x1xf32>
    %2 = "tosa.reshape"(%1) {new_shape = array<i64: 1, 4, 4>} : (tensor<1x4x4x1xf32>) -> tensor<1x4x4xf32>
    %3, %4 = "tosa.rfft2d"(%2) : (tensor<1x4x4xf32>) -> (tensor<1x4x3xf32>, tensor<1x4x3xf32>)
    %5 = "tosa.reshape"(%3) {new_shape = array<i64: 1, 4, 3, 1>} : (tensor<1x4x3xf32>) -> tensor<1x4x3x1xf32>
    %6 = "tosa.reshape"(%4) {new_shape = array<i64: 1, 4, 3, 1>} : (tensor<1x4x3xf32>) -> tensor<1x4x3x1xf32>
    %7 = "tosa.concat"(%5, %6) {axis = 3 : i64} : (tensor<1x4x3x1xf32>, tensor<1x4x3x1xf32>) -> tensor<1x4x3x2xf32>
    %8 = builtin.unrealized_conversion_cast %7 : tensor<1x4x3x2xf32> to tensor<1x4x3xcomplex<f32>>
    return %8 : tensor<1x4x3xcomplex<f32>>
}
