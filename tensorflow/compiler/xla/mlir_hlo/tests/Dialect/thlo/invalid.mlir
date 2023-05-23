// RUN: mlir-hlo-opt %s -verify-diagnostics -split-input-file

func.func @concatenate(%arg1: tensor<?x?xf32>,
                       %arg2: tensor<?x?xi32>,
                       %dst: tensor<?x?xf32>) -> tensor<?x?xf32> {
  // expected-error @+1 {{thlo.concatenate' op expected element type of input 'i32' to match output element type 'f32'}}
  %cat = thlo.concatenate
      ins(%arg1: tensor<?x?xf32>, %arg2: tensor<?x?xi32>)
      outs(%dst: tensor<?x?xf32>)
      dimension = 0
  func.return %cat : tensor<?x?xf32>
}

// -----

func.func @concatenate_mismatch_rank(%arg1: tensor<?x?xf32>,
                       %arg2: tensor<?x?x?xf32>,
                       %dst: tensor<?x?xf32>) -> tensor<?x?xf32> {
  // expected-error @+1 {{thlo.concatenate' op expected all args to be rank 2, got 3 in arg 1}}
  %cat = thlo.concatenate
      ins(%arg1: tensor<?x?xf32>, %arg2: tensor<?x?x?xf32>)
      outs(%dst: tensor<?x?xf32>)
      dimension = 0
  func.return %cat : tensor<?x?xf32>
}

// -----

func.func @concatenate_mismatch_shape(%arg1: tensor<?x8xf32>,
                       %arg2: tensor<?x?xf32>,
                       %dst: tensor<?x?xf32>) -> tensor<?x?xf32> {
  // expected-error @+1 {{thlo.concatenate' op shape of input arg 1: 'tensor<?x?xf32>' doesn't match expected shape 'tensor<?x8xf32>'}}
  %cat = thlo.concatenate
      ins(%arg1: tensor<?x8xf32>, %arg2: tensor<?x?xf32>)
      outs(%dst: tensor<?x?xf32>)
      dimension = 0
  func.return %cat : tensor<?x?xf32>
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

func.func @scatter_indices_wrong_rank(%indices: tensor<2x2x2xindex>,
    %updates: tensor<2x1x3xf32>, %init: tensor<3x3xf32>) -> tensor<3x3xf32> {
  // expected-error@+1{{expected `indices` to be a 2D tensor}}
  %0 = thlo.scatter ins(%indices : tensor<2x2x2xindex>,
                        %updates : tensor<2x1x3xf32>)
                    outs(%init : tensor<3x3xf32>)
                    (%in: f32, %out: f32) {
    %sum = arith.addf %in, %out : f32
    thlo.yield %sum : f32
  }
  return %0 : tensor<3x3xf32>
}

// -----

func.func @scatter_updates_indices_major_dim_mismatch(
    %indices: tensor<2x2xindex>, %updates: tensor<3x1x3xf32>,
    %init: tensor<3x3xf32>) -> tensor<3x3xf32> {
  // expected-error@+1{{expected major dimension of `indices` to match major dimension of `updates`}}
  %0 = thlo.scatter ins(%indices : tensor<2x2xindex>,
                        %updates : tensor<3x1x3xf32>)
                    outs(%init : tensor<3x3xf32>)
                    (%in: f32, %out: f32) {
    %sum = arith.addf %in, %out : f32
    thlo.yield %sum : f32
  }
  return %0 : tensor<3x3xf32>
}

// -----

func.func @scatter_indices_dynamic_index_vector_dim(
    %indices: tensor<2x?xindex>, %updates: tensor<2x1x3xf32>,
    %init: tensor<3x3xf32>) -> tensor<3x3xf32> {
  // expected-error@+1{{expected index vector dimension size to be static}}
  %0 = thlo.scatter ins(%indices : tensor<2x?xindex>,
                        %updates : tensor<2x1x3xf32>)
                    outs(%init : tensor<3x3xf32>)
                    (%in: f32, %out: f32) {
    %sum = arith.addf %in, %out : f32
    thlo.yield %sum : f32
  }
  return %0 : tensor<3x3xf32>
}

// -----

func.func @scatter_indices_index_vector_dim_too_big(
    %indices: tensor<2x9xindex>, %updates: tensor<2x1x3xf32>,
    %init: tensor<3x3xf32>) -> tensor<3x3xf32> {
  // expected-error@+1{{expected index vector dimension size = 9 to be smaller or equal than `init` rank = 2}}
  %0 = thlo.scatter ins(%indices : tensor<2x9xindex>,
                        %updates : tensor<2x1x3xf32>)
                    outs(%init : tensor<3x3xf32>)
                    (%in: f32, %out: f32) {
    %sum = arith.addf %in, %out : f32
    thlo.yield %sum : f32
  }
  return %0 : tensor<3x3xf32>
}

// -----

func.func @scatter_updates_init_rank_mismatch(%indices: tensor<2x2xindex>,
    %updates: tensor<2x3xf32>, %init: tensor<3x3xf32>) -> tensor<3x3xf32> {
  // expected-error@+1{{expected `updates` rank + 1 to match `init` rank}}
  %0 = thlo.scatter ins(%indices : tensor<2x2xindex>,
                        %updates : tensor<2x3xf32>)
                    outs(%init : tensor<3x3xf32>)
                    (%in: f32, %out: f32) {
    %sum = arith.addf %in, %out : f32
    thlo.yield %sum : f32
  }
  return %0 : tensor<3x3xf32>
}

// -----

func.func @scatter_updates_init_element_type_mismatch(
    %indices: tensor<2x2xindex>, %updates: tensor<2x1x3xf32>,
    %init: tensor<3x3xi32>) -> tensor<3x3xi32> {
  // expected-error@+1{{expected `updates` element type to match `init` element type}}
  %0 = thlo.scatter ins(%indices : tensor<2x2xindex>,
                        %updates : tensor<2x1x3xf32>)
                    outs(%init : tensor<3x3xi32>)
                    (%in: f32, %out: f32) {
    %sum = arith.addf %in, %out : f32
    thlo.yield %sum : f32
  }
  return %0 : tensor<3x3xi32>
}

// -----

func.func @gather_output_result_mismatch(
    %arg: tensor<100xf32>, %indices: tensor<42x1xindex>, %dst: tensor<42xf32>)
    -> tensor<42xf64> {
  // expected-error@+1{{'thlo.gather' op expected type of operand #2 ('tensor<42xf32>') to match type of corresponding result ('tensor<42xf64>')}}
  %gather = "thlo.gather"(%arg, %indices, %dst) :
      (tensor<100xf32>, tensor<42x1xindex>, tensor<42xf32>) -> (tensor<42xf64>)
  func.return %gather : tensor<42xf64>
}

// -----

func.func @gather_invalid_dynamic_indices(
    %arg: tensor<100xf32>, %indices: tensor<42x?xindex>, %dst: tensor<42xf32>)
    -> tensor<42xf64> {
  // expected-error@+1{{'thlo.gather' op expected type of operand #2 ('tensor<42xf32>') to match type of corresponding result ('tensor<42xf64>')}}
  %gather = "thlo.gather"(%arg, %indices, %dst) :
      (tensor<100xf32>, tensor<42x?xindex>, tensor<42xf32>) -> (tensor<42xf64>)
  func.return %gather : tensor<42xf64>
}

// -----

func.func @gather_invalid_indices_shape(
    %arg: tensor<100xf32>, %indices: tensor<42xindex>, %dst: tensor<42xf32>)
    -> tensor<42xf64> {
  // expected-error@+1{{'thlo.gather' op expected `indices` to be a 2D tensor}}
  %gather = "thlo.gather"(%arg, %indices, %dst) :
      (tensor<100xf32>, tensor<42xindex>, tensor<42xf32>) -> (tensor<42xf64>)
  func.return %gather : tensor<42xf64>
}

// -----

func.func @gather_indices_dst_mismatch(
    %arg: tensor<100xf32>, %indices: tensor<42x1xindex>, %dst: tensor<43xf32>)
    -> tensor<43xf64> {
  // expected-error@+1{{'thlo.gather' op expected major dimension of `startIndices` to match major dimension of `init`}}
  %gather = "thlo.gather"(%arg, %indices, %dst) :
      (tensor<100xf32>, tensor<42x1xindex>, tensor<43xf32>) -> (tensor<43xf64>)
  func.return %gather : tensor<43xf64>
}

// -----

func.func @gather_invalid_dst_shape(
    %arg: tensor<100xf32>, %indices: tensor<42x1xindex>, %dst: tensor<42x?xf32>)
    -> tensor<42x?xf64> {
  // expected-error@+1{{'thlo.gather' op only the major dimenion of `init` may be dynamic}}
  %gather = "thlo.gather"(%arg, %indices, %dst) :
      (tensor<100xf32>, tensor<42x1xindex>, tensor<42x?xf32>) -> (tensor<42x?xf64>)
  func.return %gather : tensor<42x?xf64>
}

// -----

func.func @sort_mismatched_number_of_inputs_and_outputs(
      %input1: tensor<?x?xf32>, %input2: tensor<?x?xi32>,
      %init1: tensor<?x?xf32>)
    -> tensor<?x?xf32> {
  // expected-error@+1{{'thlo.sort' op expected the number of inputs 2 to match the number of outputs 1}}
  %sorted = thlo.sort
      ins(%input1: tensor<?x?xf32>, %input2: tensor<?x?xi32>)
      outs(%init1: tensor<?x?xf32>)
      dimension = 0
      is_stable = true
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
      dimension = 0
      is_stable = true
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
      dimension = 0
      is_stable = true
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
      dimension = 0
      is_stable = true
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
      dimension = 0
      is_stable = true
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
      dimension = 0
      is_stable = true
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
      dimension = 0
      is_stable = true
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
      dimension = 2
      is_stable = true
      (%e11: f32, %e12: f32, %e21: i32, %e22: i32) {
        %gt = arith.cmpf ogt, %e11, %e12: f32
        thlo.yield %gt : i1
      }
  func.return %sorted1, %sorted2 : tensor<?x?xf32>, tensor<?x?xi32>
}
