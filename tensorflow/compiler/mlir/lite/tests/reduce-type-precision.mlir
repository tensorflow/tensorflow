// RUN: tf-opt %s -split-input-file -tfl-reduce-type-precision -verify-diagnostics

func.func @testI8ToI4WithinRange() -> (tensor<4xi8>) {
  %0 = arith.constant dense<[-8, 0, 1, 7]> : tensor<4xi8>
  // expected-error@+1 {{type of return operand 0 ('tensor<4xi4>') doesn't match function result type ('tensor<4xi8>')}}
  func.return %0 : tensor<4xi8>
}

func.func @testI8ToI4NotWithinRange() -> tensor<4xi8> {
  %0 = arith.constant dense<[-10, 2, 3, 8]> : tensor<4xi8>
  func.return %0 : tensor<4xi8>
}
