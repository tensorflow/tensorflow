// RUN: mlir-hlo-opt %s -verify-diagnostics -split-input-file

// -----

func.func @unary_eltwise_two_types(%arg0: tensor<?x?xf64>,
                                      %arg1: tensor<?x?xf64>) -> tensor<?x?xf64> {
  // expected-error @+1 {{custom op 'mhlo.abs' 1 operands present, but expected 2}}
  %0 = mhlo.abs %arg0 : (tensor<?x?xf64>, tensor<?x?xf32>) -> tensor<?x?xf64>
  func.return %0 : tensor<?x?xf64>
}

// -----

// TODO(ajcbik): error message is a bit too strict, should be "compatible" type?
func.func @binary_eltwise_type_mismatch(%arg0: tensor<?x?xf64>,
                                        %arg1: tensor<?x?xf32>) -> tensor<?x?xf64> {
  // expected-error @+1 {{'mhlo.add' op requires compatible types for all operands and results}}
  %0 = mhlo.add %arg0, %arg1 : (tensor<?x?xf64>, tensor<?x?xf32>) -> tensor<?x?xf64>
  func.return %0 : tensor<?x?xf64>
}

// -----

func.func @binary_eltwise_three_types(%arg0: tensor<?x?xf64>,
                                      %arg1: tensor<?x?xf64>) -> tensor<?x?xf64> {
  // expected-error @+1 {{custom op 'mhlo.add' 2 operands present, but expected 3}}
  %0 = mhlo.add %arg0, %arg1 : (tensor<?x?xf64>, tensor<?x?xf32>, tensor<?x?xf64>) -> tensor<?x?xf64>
  func.return %0 : tensor<?x?xf64>
}

// -----

func.func @binary_eltwise_multiple_out(%arg0: tensor<?x?xf64>,
                                      %arg1: tensor<?x?xf64>) -> tensor<?x?xf64> {
  // expected-error @+1 {{custom op 'mhlo.add' expected single output}}
  %0 = mhlo.add %arg0, %arg1 : (tensor<?x?xf64>, tensor<?x?xf32>) -> (tensor<?x?xf64>, tensor<?x?xf32>)
  func.return %0 : tensor<?x?xf64>
}

// -----

func.func @tuple_type_mismatch(%arg0: tensor<1xf64>) -> tensor<1xf64> {
  // expected-error @+1 {{custom op 'mhlo.tuple' expected tuple type}}
  %0 = mhlo.tuple %arg0, %arg1 : tensor<1xf64>, tensor<1xf32>
  func.return %0 : tensor<1xf64>
}

// -----

func.func @tuple_type_mismatch(%arg0: tensor<1xf64>) -> tensor<1xf64> {
  // expected-error @+1 {{expected '->' in function type}}
  %0 = mhlo.tuple %arg0, %arg0 : (tensor<1xf64>, tensor<1xf64>)
  func.return %0 : tensor<1xf64>
}

// -----

func.func @tuple_count_mismatch(%arg0: tensor<1xf64>) -> tensor<1xf64> {
  // expected-error @+1 {{custom op 'mhlo.tuple' number of operands and types do not match: got 2 operands and 1 types}}
  %0 = mhlo.tuple %arg0, %arg0 : tuple<tensor<1xf64>>
  func.return %0 : tensor<1xf64>
}

// -----

func.func @pairwise_count_mismatch(%arg0: tensor<1xf64>) -> tensor<1xf64> {
  // expected-error @+1 {{custom op 'mhlo.optimization_barrier' number of operands and types do not match: got 2 operands and 1 types}}
  %0 = mhlo.optimization_barrier %arg0, %arg0 : tensor<1xf64>
  func.return %0 : tensor<1xf64>
}

// -----

func.func @pairwise_type_not_list(%arg0: tensor<1xf64>) -> tensor<1xf64> {
  // expected-error @+2 {{expected non-function type}}
  // expected-error @+1 {{custom op 'mhlo.optimization_barrier' expected type list}}
  %0 = mhlo.optimization_barrier %arg0, %arg0 : %arg0
  func.return %0 : tensor<1xf64>
}

// -----

func.func @one_result_type(%arg0: tensor<1xf64>) -> tensor<1xf64> {
  // expected-error @+1 {{expected non-function type}}
  %0 = mhlo.abs %arg0 : %arg0
  func.return %0 : tensor<1xf64>
}

// -----
func.func @complex_type_not_type(%arg0: tensor<1xf64>) -> tensor<1xf64> {
  // expected-error @+1 {{expected non-function type}}
  %0 = mhlo.complex %arg0, %arg0 : %arg0
  func.return %0 : tensor<1xf64>
}

// -----

func.func @complex_type_not_tensor(%arg0: tensor<1xf64>) -> () {
  // expected-error @+1 {{custom op 'mhlo.complex' expected tensor with complex element type}}
  %0 = mhlo.complex %arg0, %arg0 : complex<f64>
  func.return
}

// -----

func.func @complex_type_not_complex(%arg0: tensor<1xf64>) -> () {
  // expected-error @+1 {{custom op 'mhlo.complex' expected tensor with complex element type}}
  %0 = mhlo.complex %arg0, %arg0 : tensor<1xf64>
  func.return
}

// -----

func.func @dense_array_nested(%arg0: tensor<1x2xf64>) -> () {
  // expected-error @+2 {{custom op 'stablehlo.transpose' expected integer value}}
  // expected-error @+1 {{expected ']'}}
  %0 = stablehlo.transpose %arg0, dims = [1, [0]] : tensor<1xf64>
  func.return
}

// -----

func.func @select_type_wrong_type(%arg0: tensor<2x3xi1>, %arg1: tensor<2x3xi32>) -> () {
  // expected-error @+1 {{custom op 'mhlo.select' expected functional type or list of two types}}
  %0 = mhlo.select %arg0, %arg1, %arg1 : tensor<2x3xi1>
  func.return %0
}

// -----

func.func @select_type_too_many(%arg0: tensor<2x3xi1>, %arg1: tensor<2x3xi32>) -> () {
  // expected-error @+1 {{custom op 'mhlo.select' expected functional type or list of two types}}
  %0 = mhlo.select %arg0, %arg1, %arg1 : tensor<2x3xi1>, tensor<2x3xi32>, tensor<2x3xi32>
  func.return
}

// -----

func.func @pairwise_type_not_type(%arg0: tensor<1xf64>) -> tensor<1xf64> {
  // expected-error @+1 {{expected non-function type}}
  %0 = mhlo.select %arg0, %arg1, %arg1 : %arg0
  func.return %0 : tensor<1xf64>
}

// -----

func.func @reduce_precision_no_e_num(%arg0: tensor<3x4xf32>) -> (tensor<3x4xf32>) {
  // expected-error @+1 {{custom op 'mhlo.reduce_precision' expected exponent mantissa in format e#m#, saw em2}}
  %0 = mhlo.reduce_precision %arg0, format = em2 : tensor<3x4xf32>
  func.return %0 : tensor<?x?xf64>
}

// -----

func.func @reduce_precision_not_literal(%arg0: tensor<3x4xf32>) -> (tensor<3x4xf32>) {
  // expected-error @+1 {{custom op 'mhlo.reduce_precision' expected valid keyword}}
  %0 = mhlo.reduce_precision %arg0, format = "e2m2" : tensor<3x4xf32>
  func.return %0 : tensor<?x?xf64>
}

// -----

func.func @reduce_precision_no_em(%arg0: tensor<3x4xf32>) -> (tensor<3x4xf32>) {
  // expected-error @+1 {{custom op 'mhlo.reduce_precision' expected exponent mantissa in format e#m#, saw z4f2}}
  %0 = mhlo.reduce_precision %arg0, format = z4f2 : tensor<3x4xf32>
  func.return %0 : tensor<?x?xf64>
}

// -----

func.func @reduce_precision_em_order(%arg0: tensor<3x4xf32>) -> (tensor<3x4xf32>) {
  // expected-error @+1 {{custom op 'mhlo.reduce_precision' expected exponent mantissa in format e#m#, saw m2e2}}
  %0 = mhlo.reduce_precision %arg0, format = m2e2 : tensor<3x4xf32>
  func.return %0 : tensor<?x?xf64>
}

// -----

func.func @reduce_precision_no_e(%arg0: tensor<3x4xf32>) -> (tensor<3x4xf32>) {
  // expected-error @+1 {{custom op 'mhlo.reduce_precision' expected exponent mantissa in format e#m#, saw m2}}
  %0 = mhlo.reduce_precision %arg0, format = m2 : tensor<3x4xf32>
  func.return %0 : tensor<?x?xf64>
}

// -----

func.func @reduce_precision_no_m(%arg0: tensor<3x4xf32>) -> (tensor<3x4xf32>) {
  // expected-error @+1 {{custom op 'mhlo.reduce_precision' expected exponent mantissa in format e#m#, saw e2}}
  %0 = mhlo.reduce_precision %arg0, format = e2 : tensor<3x4xf32>
  func.return %0 : tensor<?x?xf64>
}

// -----

func.func @reduce_precision_no_m_num(%arg0: tensor<3x4xf32>) -> (tensor<3x4xf32>) {
  // expected-error @+1 {{custom op 'mhlo.reduce_precision' expected exponent mantissa in format e#m#, saw e2m}}
  %0 = mhlo.reduce_precision %arg0, format = e2m : tensor<3x4xf32>
  func.return %0 : tensor<?x?xf64>
}

// -----

func.func @scan_body_returns_too_few_results(%input: tensor<10xf32>, %init: tensor<f32>) -> tensor<10xf32> {
  // expected-error @+2 {{'mhlo.scan' op failed to infer returned types}}
  // expected-error @+1 {{ScanOp body must return at least 1 values (carries)}}
  %0:2 = mhlo.scan (%input) inits (%init) dimension=0 {
  ^bb0(%input0: tensor<f32>, %carry0: tensor<f32>):
    "mhlo.return"() : () -> ()
  } : (tensor<10xf32>, tensor<f32>) -> (tensor<10xf32>, tensor<f32>)
  func.return %0#0 : tensor<10xf32>
}

// -----

func.func @scan_dim_out_of_bounds(%input: tensor<10xf32>, %init: tensor<f32>) -> tensor<10xf32> {
  // expected-error @+1 {{scan dimension of operand 0 is out of bounds}}
  %0:2 = mhlo.scan (%input) inits (%init) dimension=1 {
  ^bb0(%carry0: tensor<f32>, %input0: tensor<f32>):
    %1 = mhlo.add %carry0, %input0 : tensor<f32>
    mhlo.return %1, %1 : tensor<f32>, tensor<f32>
  } : (tensor<10xf32>, tensor<f32>) -> (tensor<10xf32>, tensor<f32>)
  func.return %0#0 : tensor<10xf32>
}

// -----

func.func @scan_dim_out_of_bounds_output(%input: tensor<10x10xf32>, %init: tensor<f32>) -> tensor<10x10xf32> {
  // expected-error @+1 {{operand and body argument 0 are incompatible}}
  %0:2 = mhlo.scan (%input) inits (%init) dimension=1 {
  ^bb0(%carry0: tensor<f32>, %input0: tensor<10xf32>):
    %1 = mhlo.add %carry0, %carry0 : tensor<f32>
    mhlo.return %1, %1 : tensor<f32>, tensor<f32>
  } : (tensor<10x10xf32>, tensor<f32>) -> (tensor<10x10xf32>, tensor<f32>)
  func.return %0#0 : tensor<10x10xf32>
}
