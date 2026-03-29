// RUN: sdy_opt %s -xla-sdy-import-func-calls -verify-diagnostics

// expected-error @below {{module contains multiple functions; expected only one.}}
module {
  func.func @public_non_main_function(%arg0: tensor<8x2xi32>) -> tensor<8x2xi32> {
    %0 = call @bar(%arg0) : (tensor<8x2xi32>) -> tensor<8x2xi32>
    return %0 : tensor<8x2xi32>
  }

  func.func @foo(%arg0: tensor<8x2xi32>) -> tensor<8x2xi32> {
    // expected-warning @+1 {{function @bar has multiple call ops, we need to clone the function body for each call}}
    %0 = call @bar(%arg0) : (tensor<8x2xi32>) -> tensor<8x2xi32>
    return %0 : tensor<8x2xi32>
  }

  func.func private @bar(%arg0: tensor<8x2xi32>) -> tensor<8x2xi32> {
    return %arg0 : tensor<8x2xi32>
  }
}

