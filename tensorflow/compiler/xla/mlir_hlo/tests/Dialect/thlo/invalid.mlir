// RUN: mlir-hlo-opt %s -verify-diagnostics -split-input-file

func.func @concatenate(%arg1: tensor<?x?xf32>,
                       %arg2: tensor<?x?xi32>,
                       %dst: tensor<?x?xf32>) -> tensor<?x?xf32> {
  // expected-error @+1 {{thlo.concatenate' op expected element type of input 'i32' to match output element type 'f32'}}
  %cat = thlo.concatenate
      ins(%arg1: tensor<?x?xf32>, %arg2: tensor<?x?xi32>)
      outs(%dst: tensor<?x?xf32>)
      { dimension = 0 : i64 }
  func.return %cat : tensor<?x?xf32>
}

// -----

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
    // expected-error @+1{{'thlo.yield' op expects parent op to be one of}}
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

func.func @map_buffer_semantics_with_tensor_result(
    %lhs: memref<64xf32>, %rhs: memref<64xf32>, %init: tensor<64xf32>)
    -> tensor<64xf32> {
  // expected-error@+1{{'thlo.map' op expected either buffer or tensor semantics}}
  %add = "thlo.map"(%lhs, %rhs, %init) ({
     ^bb0(%lhs_elem: f32, %rhs_elem: f32):
       %0 = arith.addf %lhs_elem, %rhs_elem: f32
       thlo.yield %0: f32
  }) : (memref<64xf32>, memref<64xf32>, tensor<64xf32>) -> tensor<64xf32>
  func.return %add : tensor<64xf32>
}

// -----

func.func @map_input_mapper_arity_mismatch(
    %lhs: tensor<64xf32>, %rhs: tensor<64xf32>, %init: tensor<64xf32>)
    -> tensor<64xf32> {
  // expected-error@+1{{'thlo.map' op expects number of operands to match the arity of mapper, but got: 2 and 3}}
  %add = thlo.map
      ins(%lhs:tensor<64xf32>, %rhs:tensor<64xf32>)
      outs(%init:tensor<64xf32>)
      (%lhs_elem: f32, %rhs_elem: f32, %extra_elem: f32) {
        %0 = arith.addf %lhs_elem, %rhs_elem: f32
        thlo.yield %0: f32
      }
  func.return %add : tensor<64xf32>
}

// -----

func.func @map_input_mapper_type_mismatch(
    %lhs: tensor<64xf32>, %rhs: tensor<64xf32>, %init: tensor<64xf32>)
    -> tensor<64xf32> {
    // expected-error@+1{{'thlo.map' op expected element type of input 'f32' to match bbArg type 'f64'}}
  %add = thlo.map
      ins(%lhs:tensor<64xf32>, %rhs:tensor<64xf32>)
      outs(%init:tensor<64xf32>)
      (%lhs_elem: f64, %rhs_elem: f64) {
        %0 = arith.addf %lhs_elem, %rhs_elem: f64
        thlo.yield %0: f64
      }
  func.return %add : tensor<64xf32>
}

// -----

func.func @map_input_output_shape_mismatch(
    %lhs: tensor<64x64xf32>, %rhs: tensor<64x64xf32>, %init: tensor<32xf32>)
    -> tensor<32xf32> {
    // expected-error@+1{{'thlo.map' op expected shape of input (64, 64) to match shape of output (32)}}
  %add = thlo.map
      ins(%lhs:tensor<64x64xf32>, %rhs:tensor<64x64xf32>)
      outs(%init:tensor<32xf32>)
      (%lhs_elem: f32, %rhs_elem: f32) {
        %0 = arith.addf %lhs_elem, %rhs_elem: f32
        thlo.yield %0: f32
      }
  func.return %add : tensor<32xf32>
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

// -----

func.func @scatter_indices_wrong_rank(%indices: tensor<2x2x2xi32>,
    %updates: tensor<2x1x3xf32>, %init: tensor<3x3xf32>) -> tensor<3x3xf32> {
  // expected-error@+1{{expected `indices` to be a 2D tensor}}
  %0 = thlo.scatter ins(%indices : tensor<2x2x2xi32>,
                        %updates : tensor<2x1x3xf32>)
                    outs(%init : tensor<3x3xf32>)
                    (%in: f32, %out: f32) {
    %sum = arith.addf %in, %out : f32
    thlo.yield %sum : f32
  }
  return %0 : tensor<3x3xf32>
}

// -----

func.func @scatter_updates_indices_major_dim_mismatch(%indices: tensor<2x2xi32>,
    %updates: tensor<3x1x3xf32>, %init: tensor<3x3xf32>) -> tensor<3x3xf32> {
  // expected-error@+1{{expected major dimension of `indices` to match major dimension of `updates`}}
  %0 = thlo.scatter ins(%indices : tensor<2x2xi32>, %updates : tensor<3x1x3xf32>)
                    outs(%init : tensor<3x3xf32>)
                    (%in: f32, %out: f32) {
    %sum = arith.addf %in, %out : f32
    thlo.yield %sum : f32
  }
  return %0 : tensor<3x3xf32>
}

// -----

func.func @scatter_indices_dynamic_index_vector_dim(%indices: tensor<2x?xi32>,
    %updates: tensor<2x1x3xf32>, %init: tensor<3x3xf32>) -> tensor<3x3xf32> {
  // expected-error@+1{{expected index vector dimension size to be static}}
  %0 = thlo.scatter ins(%indices : tensor<2x?xi32>, %updates : tensor<2x1x3xf32>)
                    outs(%init : tensor<3x3xf32>)
                    (%in: f32, %out: f32) {
    %sum = arith.addf %in, %out : f32
    thlo.yield %sum : f32
  }
  return %0 : tensor<3x3xf32>
}

// -----

func.func @scatter_indices_index_vector_dim_too_big(%indices: tensor<2x9xi32>,
    %updates: tensor<2x1x3xf32>, %init: tensor<3x3xf32>) -> tensor<3x3xf32> {
  // expected-error@+1{{expected index vector dimension size = 9 to be smaller or equal than `init` rank = 2}}
  %0 = thlo.scatter ins(%indices : tensor<2x9xi32>, %updates : tensor<2x1x3xf32>)
                    outs(%init : tensor<3x3xf32>)
                    (%in: f32, %out: f32) {
    %sum = arith.addf %in, %out : f32
    thlo.yield %sum : f32
  }
  return %0 : tensor<3x3xf32>
}

// -----

func.func @scatter_updates_init_rank_mismatch(%indices: tensor<2x2xi32>,
    %updates: tensor<2x3xf32>, %init: tensor<3x3xf32>) -> tensor<3x3xf32> {
  // expected-error@+1{{expected `updates` rank + 1 to match `init` rank}}
  %0 = thlo.scatter ins(%indices : tensor<2x2xi32>, %updates : tensor<2x3xf32>)
                    outs(%init : tensor<3x3xf32>)
                    (%in: f32, %out: f32) {
    %sum = arith.addf %in, %out : f32
    thlo.yield %sum : f32
  }
  return %0 : tensor<3x3xf32>
}

// -----

func.func @scatter_updates_init_element_type_mismatch(%indices: tensor<2x2xi32>,
    %updates: tensor<2x1x3xf32>, %init: tensor<3x3xi32>) -> tensor<3x3xi32> {
  // expected-error@+1{{expected `updates` element type to match `init` element type}}
  %0 = thlo.scatter ins(%indices : tensor<2x2xi32>, %updates : tensor<2x1x3xf32>)
                    outs(%init : tensor<3x3xi32>)
                    (%in: f32, %out: f32) {
    %sum = arith.addf %in, %out : f32
    thlo.yield %sum : f32
  }
  return %0 : tensor<3x3xi32>
}

// -----

func.func @gather_output_result_mismatch(
    %arg: tensor<100xf32>, %indices: tensor<42x1xi64>, %dst: tensor<42xf32>)
    -> tensor<42xf64> {
  // expected-error@+1{{'thlo.gather' op expected type of operand #2 ('tensor<42xf32>') to match type of corresponding result ('tensor<42xf64>')}}
  %gather = "thlo.gather"(%arg, %indices, %dst) :
      (tensor<100xf32>, tensor<42x1xi64>, tensor<42xf32>) -> (tensor<42xf64>)
  func.return %gather : tensor<42xf64>
}

// -----

func.func @sort_mismatched_number_of_inputs_and_outputs(
      %input1: tensor<?x?xf32>, %input2: tensor<?x?xi32>,
      %init1: tensor<?x?xf32>)
    -> tensor<?x?xf32> {
  // expected-error@+1{{'thlo.sort' op expected 2 operands, but found 3}}
  %sorted = thlo.sort
      ins(%input1: tensor<?x?xf32>, %input2: tensor<?x?xi32>)
      outs(%init1: tensor<?x?xf32>)
      { dimension = 0 : i64, is_stable = true }
      (%e11: f32, %e12: f32) {
        %gt = arith.cmpf ogt, %e11, %e12: f32
        thlo.yield %gt : i1
      }
  func.return %sorted : tensor<?x?xf32>
}

// -----

func.func @sort_mismatched_number_of_inputs_and_comparator_arguments(
      %input1: tensor<?x?xf32>, %input2: tensor<?x?xi32>,
      %init1: tensor<?x?xf32>, %init2: tensor<?x?xi32>)
    -> (tensor<?x?xf32>, tensor<?x?xi32>) {
  // expected-error@+1{{'thlo.sort' op expected the number of block arguments 3 to be twice the number of inputs (2*2)}}
  %sorted1, %sorted2 = thlo.sort
      ins(%input1: tensor<?x?xf32>, %input2: tensor<?x?xi32>)
      outs(%init1: tensor<?x?xf32>, %init2: tensor<?x?xi32>)
      { dimension = 0 : i64, is_stable = true }
      (%e11: f32, %e12: f32, %e21: i32) {
        %gt = arith.cmpf ogt, %e11, %e12: f32
        thlo.yield %gt : i1
      }
  func.return %sorted1, %sorted2 : tensor<?x?xf32>, tensor<?x?xi32>
}

// -----

func.func @sort_mismatched_input_and_comparator_type(
      %input1: tensor<?x?xf32>, %input2: tensor<?x?xi32>,
      %init1: tensor<?x?xf32>, %init2: tensor<?x?xi32>)
    -> (tensor<?x?xf32>, tensor<?x?xi32>) {
  // expected-error@+1{{'thlo.sort' op expected element type of input 1 to match type of the corresponding arguments to the comparison function but got 'i32' and ('i32', 'f32')}}
  %sorted1, %sorted2 = thlo.sort
      ins(%input1: tensor<?x?xf32>, %input2: tensor<?x?xi32>)
      outs(%init1: tensor<?x?xf32>, %init2: tensor<?x?xi32>)
      { dimension = 0 : i64, is_stable = true }
      (%e11: f32, %e12: f32, %e21: i32, %e22: f32) {
        %gt = arith.cmpf ogt, %e11, %e12: f32
        thlo.yield %gt : i1
      }
  func.return %sorted1, %sorted2 : tensor<?x?xf32>, tensor<?x?xi32>
}

// -----

func.func @sort_comparator_yields_different_than_one_output(
      %input1: tensor<?x?xf32>, %input2: tensor<?x?xi32>,
      %init1: tensor<?x?xf32>, %init2: tensor<?x?xi32>)
    -> (tensor<?x?xf32>, tensor<?x?xi32>) {
  %sorted1, %sorted2 = thlo.sort
      ins(%input1: tensor<?x?xf32>, %input2: tensor<?x?xi32>)
      outs(%init1: tensor<?x?xf32>, %init2: tensor<?x?xi32>)
      { dimension = 0 : i64, is_stable = true }
      (%e11: f32, %e12: f32, %e21: i32, %e22: i32) {
        %gt = arith.cmpf ogt, %e11, %e12: f32
        // expected-error@+1{{'thlo.yield' op expects number of tensor output args = 1 to match the number of yield operands = 2}}
        thlo.yield %gt, %gt : i1, i1
      }
  func.return %sorted1, %sorted2 : tensor<?x?xf32>, tensor<?x?xi32>
}

// -----

func.func @sort_comparator_yields_non_boolean(
      %input1: tensor<?x?xf32>, %input2: tensor<?x?xi32>,
      %init1: tensor<?x?xf32>, %init2: tensor<?x?xi32>)
    -> (tensor<?x?xf32>, tensor<?x?xi32>) {
  %sorted1, %sorted2 = thlo.sort
      ins(%input1: tensor<?x?xf32>, %input2: tensor<?x?xi32>)
      outs(%init1: tensor<?x?xf32>, %init2: tensor<?x?xi32>)
      { dimension = 0 : i64, is_stable = true }
      (%e11: f32, %e12: f32, %e21: i32, %e22: i32) {
        // expected-error@+1{{'thlo.yield' op expects yield operand 0 with type = 'f32' to match output arg element type = 'i1'}}
        thlo.yield %e11 : f32
      }
  func.return %sorted1, %sorted2 : tensor<?x?xf32>, tensor<?x?xi32>
}

// -----

func.func @sort_inputs_have_different_shapes(
      %input1: tensor<64x32xf32>, %input2: tensor<32x32xi32>,
      %init1: tensor<?x?xf32>, %init2: tensor<?x?xi32>)
    -> (tensor<?x?xf32>, tensor<?x?xi32>) {
  // expected-error@+1{{'thlo.sort' op expected all inputs to have the same shape (64, 32) but input 1 has shape (32, 32)}}
  %sorted1, %sorted2 = thlo.sort
      ins(%input1: tensor<64x32xf32>, %input2: tensor<32x32xi32>)
      outs(%init1: tensor<?x?xf32>, %init2: tensor<?x?xi32>)
      { dimension = 0 : i64, is_stable = true }
      (%e11: f32, %e12: f32, %e21: i32, %e22: i32) {
        %gt = arith.cmpf ogt, %e11, %e12: f32
        thlo.yield %gt : i1
      }
  func.return %sorted1, %sorted2 : tensor<?x?xf32>, tensor<?x?xi32>
}

// -----

func.func @sort_output_has_different_shape_from_inputs(
      %input1: tensor<64x32xf32>, %input2: tensor<64x32xi32>,
      %init1: tensor<32x64xf32>, %init2: tensor<?x?xi32>)
    -> (tensor<32x64xf32>, tensor<?x?xi32>) {
  // expected-error@+1{{'thlo.sort' op expected outputs to have shape (64, 32) but output 0 has shape (32, 64)}}
  %sorted1, %sorted2 = thlo.sort
      ins(%input1: tensor<64x32xf32>, %input2: tensor<64x32xi32>)
      outs(%init1: tensor<32x64xf32>, %init2: tensor<?x?xi32>)
      { dimension = 0 : i64, is_stable = true }
      (%e11: f32, %e12: f32, %e21: i32, %e22: i32) {
        %gt = arith.cmpf ogt, %e11, %e12: f32
        thlo.yield %gt : i1
      }
  func.return %sorted1, %sorted2 : tensor<32x64xf32>, tensor<?x?xi32>
}

// -----

func.func @sort_dimension_is_incompatible_with_rank_of_inputs(
      %input1: tensor<?x?xf32>, %input2: tensor<?x?xi32>,
      %init1: tensor<?x?xf32>, %init2: tensor<?x?xi32>)
    -> (tensor<?x?xf32>, tensor<?x?xi32>) {
  // expected-error@+1{{'thlo.sort' op sorting dimension must be in range [0, 2) but got 2}}
  %sorted1, %sorted2 = thlo.sort
      ins(%input1: tensor<?x?xf32>, %input2: tensor<?x?xi32>)
      outs(%init1: tensor<?x?xf32>, %init2: tensor<?x?xi32>)
      { dimension = 2 : i64, is_stable = true }
      (%e11: f32, %e12: f32, %e21: i32, %e22: i32) {
        %gt = arith.cmpf ogt, %e11, %e12: f32
        thlo.yield %gt : i1
      }
  func.return %sorted1, %sorted2 : tensor<?x?xf32>, tensor<?x?xi32>
}