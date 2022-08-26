// RUN: stablehlo-opt %s -verify-diagnostics -split-input-file

// -----

func.func @unary_eltwise_two_types(%arg0: tensor<?x?xf64>,
                                      %arg1: tensor<?x?xf64>) -> tensor<?x?xf64> {
  // expected-error @+1 {{custom op 'stablehlo.abs' 1 operands present, but expected 2}}
  %0 = stablehlo.abs %arg0 : (tensor<?x?xf64>, tensor<?x?xf32>) -> tensor<?x?xf64>
  func.return %0 : tensor<?x?xf64>
}

// -----

// TODO(ajcbik): error message is a bit too strict, should be "compatible" type?
func.func @binary_eltwise_type_mismatch(%arg0: tensor<?x?xf64>,
                                        %arg1: tensor<?x?xf32>) -> tensor<?x?xf64> {
  // expected-error @+1 {{'stablehlo.add' op requires compatible types for all operands and results}}
  %0 = stablehlo.add %arg0, %arg1 : (tensor<?x?xf64>, tensor<?x?xf32>) -> tensor<?x?xf64>
  func.return %0 : tensor<?x?xf64>
}

// -----

func.func @binary_eltwise_three_types(%arg0: tensor<?x?xf64>,
                                      %arg1: tensor<?x?xf64>) -> tensor<?x?xf64> {
  // expected-error @+1 {{custom op 'stablehlo.add' 2 operands present, but expected 3}}
  %0 = stablehlo.add %arg0, %arg1 : (tensor<?x?xf64>, tensor<?x?xf32>, tensor<?x?xf64>) -> tensor<?x?xf64>
  func.return %0 : tensor<?x?xf64>
}

// -----

func.func @binary_eltwise_multiple_out(%arg0: tensor<?x?xf64>,
                                      %arg1: tensor<?x?xf64>) -> tensor<?x?xf64> {
  // expected-error @+1 {{custom op 'stablehlo.add' expected single output}}
  %0 = stablehlo.add %arg0, %arg1 : (tensor<?x?xf64>, tensor<?x?xf32>) -> (tensor<?x?xf64>, tensor<?x?xf32>)
  func.return %0 : tensor<?x?xf64>
}

// -----

func.func @tuple_type_mismatch(%arg0: tensor<1xf64>) -> tensor<1xf64> {
  // expected-error @+1 {{custom op 'stablehlo.tuple' expected tuple type}}
  %0 = stablehlo.tuple %arg0, %arg1 : tensor<1xf64>, tensor<1xf32>
  func.return %0 : tensor<1xf64>
}

// -----

func.func @tuple_type_mismatch(%arg0: tensor<1xf64>) -> tensor<1xf64> {
  // expected-error @+1 {{expected '->' in function type}}
  %0 = stablehlo.tuple %arg0, %arg0 : (tensor<1xf64>, tensor<1xf64>)
  func.return %0 : tensor<1xf64>
}

// -----

func.func @tuple_count_mismatch(%arg0: tensor<1xf64>) -> tensor<1xf64> {
  // expected-error @+1 {{custom op 'stablehlo.tuple' 2 operands present, but expected 1}}
  %0 = stablehlo.tuple %arg0, %arg0 : tuple<tensor<1xf64>>
  func.return %0 : tensor<1xf64>
}

// -----

func.func @pairwise_count_mismatch(%arg0: tensor<1xf64>) -> tensor<1xf64> {
  // expected-error @+1 {{custom op 'stablehlo.optimization_barrier' 2 operands present, but expected 1}}
  %0 = stablehlo.optimization_barrier %arg0, %arg0 : tensor<1xf64>
  func.return %0 : tensor<1xf64>
}

// -----

func.func @pairwise_type_not_list(%arg0: tensor<1xf64>) -> tensor<1xf64> {
  // expected-error @+2 {{xpected non-function type}}
  // expected-error @+1 {{custom op 'stablehlo.optimization_barrier' expected type list}}
  %0 = stablehlo.optimization_barrier %arg0, %arg0 : %arg0
  func.return %0 : tensor<1xf64>
}

// -----

func.func @one_result_type(%arg0: tensor<1xf64>) -> tensor<1xf64> {
  // expected-error @+1 {{xpected non-function type}}
  %0 = stablehlo.abs %arg0 : %arg0
  func.return %0 : tensor<1xf64>
}
