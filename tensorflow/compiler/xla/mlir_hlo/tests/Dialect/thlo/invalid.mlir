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

// -----

func.func @reduction_input_vs_init_dimension_mismatch(
    %input: tensor<16x32x64xf32>,
    %init: tensor<16x64xf32>)  -> tensor<16x64xf32> {
  // expected-error @+1 {{'thlo.reduction' op init dimensions [16, 64] doesn't match input dimensions after reduction [16, 32]}}
  %reduction = thlo.reduction
      ins(%input:tensor<16x32x64xf32>)
      outs(%init:tensor<16x64xf32>)
      dimensions = [2]
      (%in: f32, %out: f32) {
        %0 = arith.addf %in, %out: f32
        thlo.yield %0: f32
      }
  func.return %reduction : tensor<16x64xf32>
}

// -----

func.func @reduction_dimensions_out_of_range(%input: tensor<16x32x64xf32>,
    %init: tensor<16x64xf32>)  -> tensor<16x64xf32> {
  // expected-error @+1 {{'thlo.reduction' op dimensions for reduction should be in the range [0, 2].}}
  %reduction = thlo.reduction
      ins(%input:tensor<16x32x64xf32>)
      outs(%init:tensor<16x64xf32>)
      dimensions = [3]
      (%in: f32, %out: f32) {
        %0 = arith.addf %in, %out: f32
        thlo.yield %0: f32
      }
  func.return %reduction : tensor<16x64xf32>
}

// -----

func.func @reduction_duplicate_dimensions(%input: tensor<16x32x64xf32>,
    %init: tensor<16xf32>)  -> tensor<16xf32> {
  // expected-error @+1 {{'thlo.reduction' op reduction dimensions are not in increasing order: 1, 1}}
  %reduction = thlo.reduction
      ins(%input:tensor<16x32x64xf32>)
      outs(%init:tensor<16xf32>)
      dimensions = [1, 1]
      (%in: f32, %out: f32) {
        %0 = arith.addf %in, %out: f32
        thlo.yield %0: f32
      }
  func.return %reduction : tensor<16xf32>
}

// -----

func.func @reduction_non_increasing_dimensions(%input: tensor<16x32x64xf32>,
    %init: tensor<16xf32>)  -> tensor<16xf32> {
  // expected-error @+1 {{'thlo.reduction' op reduction dimensions are not in increasing order: 2, 1}}
  %reduction = thlo.reduction
      ins(%input:tensor<16x32x64xf32>)
      outs(%init:tensor<16xf32>)
      dimensions = [2, 1]
      (%in: f32, %out: f32) {
        %0 = arith.addf %in, %out: f32
        thlo.yield %0: f32
      }
  func.return %reduction : tensor<16xf32>
}

// -----

func.func @reduction_reduced_input_init_rank_mismatch(%input: tensor<16x32x64xf32>,
    %init: tensor<16x64xf32>)  -> tensor<16x64xf32> {
  // expected-error @+1 {{'thlo.reduction' op number of dimensions after reduction 1 doesn't match the init rank 2}}
  %reduction = thlo.reduction
      ins(%input:tensor<16x32x64xf32>)
      outs(%init:tensor<16x64xf32>)
      dimensions = [1, 2]
      (%in: f32, %out: f32) {
        %0 = arith.addf %in, %out: f32
        thlo.yield %0: f32
      }
  func.return %reduction : tensor<16x64xf32>
}

// -----

func.func @reduction_wrong_number_of_block_arguments(
    %input1: tensor<16x32x64xf32>,
    %init1: tensor<16x64xf32>, %input2: tensor<16x32x64xf32>,
    %init2: tensor<16x64xf32>)  -> (tensor<16x64xf32>, tensor<16x64xf32>) {
  // expected-error @+1{{'thlo.reduction' op number of block arguments 2 doesn't match the number of inputs plus the number of outputs 4}}
  %reduction, %reduction2 = thlo.reduction
      ins(%input1:tensor<16x32x64xf32>, %input2:tensor<16x32x64xf32>)
      outs(%init1:tensor<16x64xf32>, %init2:tensor<16x64xf32>)
      dimensions = [1]
      (%in: f32, %out: f32) {
        %0 = arith.addf %in, %out: f32
        thlo.yield %0: f32
      }
  func.return %reduction, %reduction2 : tensor<16x64xf32>, tensor<16x64xf32>
}

// -----

func.func @reduction_wrong_block_argument_input_type(
    %input1: tensor<16x32x64xf32>,
    %init1: tensor<16x64xf32>, %input2: tensor<16x32x64xf32>,
    %init2: tensor<16x64xf32>)  -> (tensor<16x64xf32>, tensor<16x64xf32>) {
  // expected-error @+1{{'thlo.reduction' op input element types 'f32', 'f32' do not match block argument types 'f32', 'f64'}}
  %reduction, %reduction2 = thlo.reduction
      ins(%input1:tensor<16x32x64xf32>, %input2:tensor<16x32x64xf32>)
      outs(%init1:tensor<16x64xf32>, %init2:tensor<16x64xf32>)
      dimensions = [1]
      (%in1: f32, %in2: f64, %out1: f32, %out2: f64) {
        %0 = arith.addf %in1, %out1: f32
        %1 = arith.addf %in2, %out2: f64
        thlo.yield %0, %1: f32, f64
      }
  func.return %reduction, %reduction2 : tensor<16x64xf32>, tensor<16x64xf32>
}

// -----

func.func @reduction_wrong_block_argument_output_type(
    %input1: tensor<16x32x64xf32>,
    %init1: tensor<16x64xf32>, %input2: tensor<16x32x64xf32>,
    %init2: tensor<16x64xf32>)  -> (tensor<16x64xf32>, tensor<16x64xf32>) {
  // expected-error @+1{{'thlo.reduction' op output element types 'f32', 'f32' do not match block argument types 'f32', 'f64'}}
  %reduction, %reduction2 = thlo.reduction
      ins(%input1:tensor<16x32x64xf32>, %input2:tensor<16x32x64xf32>)
      outs(%init1:tensor<16x64xf32>, %init2:tensor<16x64xf32>)
      dimensions = [1]
      (%in1: f32, %in2: f32, %out1: f32, %out2: f64) {
        %0 = arith.addf %in1, %out1: f32
        thlo.yield %0, %out2: f32, f64
      }
  func.return %reduction, %reduction2 : tensor<16x64xf32>, tensor<16x64xf32>
}

// -----

func.func @reduction_incompatible_input_shapes(%input1: tensor<16x32x64xf32>,
    %init1: tensor<16x64xf32>, %input2: tensor<17x32x64xf32>,
    %init2: tensor<17x64xf32>)  -> (tensor<16x64xf32>, tensor<17x64xf32>) {
  // expected-error @+1{{'thlo.reduction' op expects all inputs to have compatible shapes. Shape at input-index 1 is not compatible with shape at input-index 0.}}
  %reduction, %reduction2 = thlo.reduction
      ins(%input1:tensor<16x32x64xf32>, %input2:tensor<17x32x64xf32>)
      outs(%init1:tensor<16x64xf32>, %init2:tensor<17x64xf32>)
      dimensions = [1]
      (%in1: f32, %in2: f32, %out1: f32, %out2: f32) {
        %0 = arith.addf %in1, %out1: f32
        %1 = arith.addf %in2, %out2: f32
        thlo.yield %0, %1: f32, f32
      }
  func.return %reduction, %reduction2 : tensor<16x64xf32>, tensor<17x64xf32>
}

// -----

func.func @reduction_incompatible_output_shapes(%input1: tensor<16x32x64xf32>,
    %init1: tensor<16x64xf32>, %input2: tensor<16x32x64xf32>,
    %init2: tensor<17x64xf32>)  -> (tensor<16x64xf32>, tensor<17x64xf32>) {
  // expected-error @+1{{'thlo.reduction' op expects all outputs to have compatible shapes. Shape at output-index 1 is not compatible with shape at output-index 0.}}
  %reduction, %reduction2 = thlo.reduction
      ins(%input1:tensor<16x32x64xf32>, %input2:tensor<16x32x64xf32>)
      outs(%init1:tensor<16x64xf32>, %init2:tensor<17x64xf32>)
      dimensions = [1]
      (%in1: f32, %in2: f32, %out1: f32, %out2: f32) {
        %0 = arith.addf %in1, %out1: f32
        %1 = arith.addf %in2, %out2: f32
        thlo.yield %0, %1: f32, f32
      }
  func.return %reduction, %reduction2 : tensor<16x64xf32>, tensor<17x64xf32>
}

// -----

func.func @yield_op_inside_mhlo_reduce(
    %arg0: tensor<5x4xf32>, %arg1: tensor<f32>) -> tensor<5xf32> {
  %0 = "mhlo.reduce"(%arg0, %arg1) ({
  ^bb0(%init: tensor<f32>, %arg3: tensor<f32>):
    %1 = mhlo.add %init, %arg3 : tensor<f32>
    // expected-error @+1{{'thlo.yield' op expects parent op to be one of 'thlo.map, thlo.reduction'}}
    thlo.yield %1: tensor<f32>
  }) {dimensions = dense<1> : tensor<1xi64>} :
    (tensor<5x4xf32>, tensor<f32>) -> tensor<5xf32>
  func.return %0 : tensor<5xf32>
}

// -----

func.func @map_binary_wrong_yield_operands(
    %lhs: tensor<64xf32>, %rhs: tensor<64xf32>, %init: tensor<64xf32>)
    -> tensor<64xf32> {
   %add = thlo.map
          ins(%lhs:tensor<64xf32>, %rhs:tensor<64xf32>)
          outs(%init:tensor<64xf32>)
          (%lhs_elem: f32, %rhs_elem: f32) {
            %0 = arith.addf %lhs_elem, %rhs_elem: f32
            // expected-error @+1{{'thlo.yield' op expects number of tensor output args = 1 to match the number of yield operands = 2}}
            thlo.yield %0, %0: f32, f32
          }
  func.return %add : tensor<64xf32>
}

// -----

func.func @variadic_reduction_wrong_yield_operand_types(
    %input1: tensor<16x32x64xf32>, %init1: tensor<16x64xf32>,
    %input2: tensor<16x32x64xi64>, %init2: tensor<16x64xi64>)
    -> (tensor<16x64xf32>, tensor<16x64xi64>) {
  %reduction, %reduction2 = thlo.reduction
      ins(%input1:tensor<16x32x64xf32>, %input2:tensor<16x32x64xi64>)
      outs(%init1:tensor<16x64xf32>, %init2:tensor<16x64xi64>)
      dimensions = [1]
      (%in1: f32, %in2: i64, %out1: f32, %out2: i64) {
        %0 = arith.addf %in1, %out1: f32
        %1 = arith.addi %in2, %out2: i64
        // expected-error @+1{{'thlo.yield' op expects yield operand 1 with type = 'f32' to match output arg element type = 'i64'}}
        thlo.yield %0, %0: f32, f32
      }
  func.return %reduction, %reduction2 : tensor<16x64xf32>, tensor<16x64xi64>
}
