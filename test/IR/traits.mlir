// RUN: mlir-opt %s -split-input-file -verify-diagnostics | FileCheck %s

// CHECK: succeededSameOperandAndResultElementType
func @succeededSameOperandAndResultElementType(%t10x10 : tensor<10x10xf32>, %t1: tensor<1xf32>, %v1: vector<1xf32>, %t1i: tensor<1xi32>) {
  %0 = "test.same_operand_and_result_type"(%t1, %t1) : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
  %1 = "test.same_operand_and_result_type"(%t1, %t10x10) : (tensor<1xf32>, tensor<10x10xf32>) -> tensor<1xf32>
  %2 = "test.same_operand_and_result_type"(%t10x10, %v1) : (tensor<10x10xf32>, vector<1xf32>) -> tensor<1xf32>
  %3 = "test.same_operand_and_result_type"(%v1, %t1) : (vector<1xf32>, tensor<1xf32>) -> tensor<1xf32>
  %4 = "test.same_operand_and_result_type"(%v1, %t1) : (vector<1xf32>, tensor<1xf32>) -> tensor<121xf32>
  return
}

// -----

func @failedSameOperandAndResultElementType(%t10x10 : tensor<10x10xf32>, %t1: tensor<1xf32>, %v1: vector<1xf32>, %t1i: tensor<1xi32>) {
  // expected-error@+1 {{requires the same element type for all operands and results}}
  %0 = "test.same_operand_and_result_type"(%t1, %t1) : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xi32>
}

// -----

func @failedSameOperandAndResultElementType(%t10x10 : tensor<10x10xf32>, %t1: tensor<1xf32>, %v1: vector<1xf32>, %t1i: tensor<1xi32>) {
  // expected-error@+1 {{requires the same element type for all operands and results}}
  %0 = "test.same_operand_and_result_type"(%t1, %t1i) : (tensor<1xf32>, tensor<1xi32>) -> tensor<1xf32>
}

// -----

// CHECK: succeededSameOperandAndResultShape
func @succeededSameOperandAndResultShape(%t10x10 : tensor<10x10xf32>, %t1: tensor<1xf32>, %tr: tensor<*xf32>) {
  %0 = "test.same_operand_and_result_shape"(%t1, %t1) : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
  %1 = "test.same_operand_and_result_shape"(%t10x10, %t10x10) : (tensor<10x10xf32>, tensor<10x10xf32>) -> tensor<10x10xf32>
  %2 = "test.same_operand_and_result_shape"(%t1, %tr) : (tensor<1xf32>, tensor<*xf32>) -> tensor<1xf32>
  return
}

// -----

func @succeededSameOperandAndResultShape(%t10x10 : tensor<10x10xf32>, %t1: tensor<1xf32>, %v1: vector<1xf32>) {
  // expected-error@+1 {{requires the same shape for all operands and results}}
  %0 = "test.same_operand_and_result_shape"(%t1, %t10x10) : (tensor<1xf32>, tensor<10x10xf32>) -> tensor<10x10xf32>
}
