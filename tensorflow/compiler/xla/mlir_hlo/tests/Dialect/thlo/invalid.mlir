// RUN: mlir-hlo-opt %s -verify-diagnostics -split-input-file

func.func @transpose_invalid_permutation(%input: tensor<16x32x64xf32>,
    %init: tensor<32x64x16xf32>) -> tensor<32x64x16xf32> {
  // expected-error @+1 {{'thlo.transpose' op permutation is not valid}}
  %transpose = thlo.transpose
      ins(%input:tensor<16x32x64xf32>)
      outs(%init:tensor<32x64x16xf32>)
      permutation = [1, 1, 2]
  func.return %transpose : tensor<32x64x16xf32>
}

// -----

func.func @transpose_permutated_dims_mismatch(%input: tensor<16x32x64xf32>,
    %init: tensor<32x64x16xf32>) -> tensor<32x64x16xf32> {
  // expected-error @+1 {{'thlo.transpose' op dim(result, 0) = 32 doesn't match dim(input, permutation[0]) = 16}}
  %transpose = thlo.transpose
      ins(%input:tensor<16x32x64xf32>)
      outs(%init:tensor<32x64x16xf32>)
      permutation = [0, 1, 2]
  func.return %transpose : tensor<32x64x16xf32>
}

// -----

func.func @transpose_rank_permutation_size_mismatch(
    %input: tensor<16x32x64xf32>,
    %init: tensor<32x64x16xf32>) -> tensor<32x64x16xf32> {
  // expected-error @+1 {{'thlo.transpose' op size of permutation 2 does not match the argument rank 3}}
  %transpose = thlo.transpose
      ins(%input:tensor<16x32x64xf32>)
      outs(%init:tensor<32x64x16xf32>)
      permutation = [1, 0]
  func.return %transpose : tensor<32x64x16xf32>
}

// -----

func.func @transpose_input_init_rank_mismatch(%input: tensor<16x32xf32>,
    %init: tensor<32x64x16xf32>) -> tensor<32x64x16xf32> {
  // expected-error @+1 {{'thlo.transpose' op input rank 2 does not match init rank 3}}
  %transpose = thlo.transpose
      ins(%input:tensor<16x32xf32>)
      outs(%init:tensor<32x64x16xf32>)
      permutation = [1, 0, 2]
  func.return %transpose : tensor<32x64x16xf32>
}

