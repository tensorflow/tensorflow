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
