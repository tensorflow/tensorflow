// RUN: mlir-hlo-opt %s --split-input-file --mhlo-legalize-dot-general-to-dot | FileCheck %s

// CHECK-LABEL: @dot_general_is_dot
func.func @dot_general_is_dot(%arg0: tensor<5x6xf32>, %arg1: tensor<6x?xf32>) -> tensor<5x?xf32> {
  // CHECK: %[[DOT:.+]] = "mhlo.dot"(%arg0, %arg1)
  // CHECK-SAME: precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]
  %0 = "mhlo.dot_general"(%arg0, %arg1) <{dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]}> : (tensor<5x6xf32>, tensor<6x?xf32>) -> tensor<5x?xf32>
  // CHECK: %[[DOT]]
  return %0 : tensor<5x?xf32>
}

// -----

// CHECK-LABEL: @dot_general_is_dot_keep_attrs
func.func @dot_general_is_dot_keep_attrs(%arg0: tensor<5x6xf32>, %arg1: tensor<6x?xf32>) -> tensor<5x?xf32> {
  // CHECK: %[[DOT:.+]] = "mhlo.dot"(%arg0, %arg1)
  // CHECK-SAME: precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]
  // CHECK-SAME: mhlo.frontend_attributes = {test_name = "test_value"}
  %0 = "mhlo.dot_general"(%arg0, %arg1) <{dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]}> {mhlo.frontend_attributes = {test_name = "test_value"}} : (tensor<5x6xf32>, tensor<6x?xf32>) -> tensor<5x?xf32>
  // CHECK: %[[DOT]]
  return %0 : tensor<5x?xf32>
}

// -----

