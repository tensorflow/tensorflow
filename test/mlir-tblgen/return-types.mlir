// RUN: mlir-opt %s -test-return-type -split-input-file -verify-diagnostics | FileCheck %s --dump-input-on-failure

// CHECK-LABEL: testReturnTypeOpInterface
func @testReturnTypeOpInterface(%arg0 : tensor<10xf32>) {
  %good = "test.op_with_infer_type_if"(%arg0, %arg0) : (tensor<10xf32>, tensor<10xf32>) -> tensor<10xf32>
  // expected-error@+1 {{incompatible with return type}}
  %bad = "test.op_with_infer_type_if"(%arg0, %arg0) : (tensor<10xf32>, tensor<10xf32>) -> tensor<*xf32>
  return
}

// -----

// CHECK-LABEL: testReturnTypeOpInterface
func @testReturnTypeOpInterfaceMismatch(%arg0 : tensor<10xf32>, %arg1 : tensor<20xf32>) {
  // expected-error@+2 {{incompatible with return type}}
  // expected-error@+1 {{operand type mismatch}}
  %bad = "test.op_with_infer_type_if"(%arg0, %arg1) : (tensor<10xf32>, tensor<20xf32>) -> tensor<*xf32>
  return
}
