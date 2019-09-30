// RUN: mlir-opt %s -test-return-type -split-input-file -verify-diagnostics | FileCheck %s --dump-input-on-failure

// CHECK-LABEL: testReturnTypeOpInterface
func @testReturnTypeOpInterface(%arg0 : tensor<10xf32>) {
  // expected-error@+1 {{expected to fail}}
  %0 = "test.op_with_infer_type_if"(%arg0, %arg0) : (tensor<10xf32>, tensor<10xf32>) -> tensor<*xf32>
  return
}
