// RUN: mlir-hlo-opt %s -verify-diagnostics -split-input-file  | FileCheck %s

// A few valid test-cases.

// -----

// CHECK-LABEL: func @reduce_valid
func.func @reduce_valid(%arg0: tensor<4x4xf32>, %arg1 : tensor<4xf32>)
    -> (tensor<4xf32>) {
  %0 = "mhlo.reduce"(%arg0, %arg1) ({

  ^bb0(%arg2: tensor<4xf32>, %arg3: tensor<4xf32> ):
    %1 = "mhlo.add"(%arg2, %arg3) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
    "mhlo.return"(%1) : (tensor<4xf32>) -> ()

  }) {dimensions = dense<[0]> : tensor<1xi64>} : (tensor<4x4xf32>, tensor<4xf32>) -> tensor<4xf32>

  func.return %0: tensor<4xf32>
}

// -----

// CHECK-LABEL:    func @reduce_complex_type
func.func @reduce_complex_type(%arg0: tensor<1x2xcomplex<f32>>, %arg1 : tensor<complex<f32>>)
    -> (tensor<1xcomplex<f32>>) {
  %0 = "mhlo.reduce"(%arg0, %arg1) ({

  ^bb0(%arg2: tensor<complex<f32>> loc("foo"), %arg3: tensor<complex<f32>> loc("foo")):
    %1 = "mhlo.add"(%arg2, %arg3) : (tensor<complex<f32>>, tensor<complex<f32>>) -> tensor<complex<f32>> loc("foo")
    "mhlo.return"(%1) : (tensor<complex<f32>>) -> () loc("foo")
  }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<1x2xcomplex<f32>>, tensor<complex<f32>>) -> tensor<1xcomplex<f32>> loc("foo")

  func.return %0: tensor<1xcomplex<f32>>
}

// -----

// CHECK-LABEL:    func @reduce_single_operand_unranked
func.func @reduce_single_operand_unranked(%arg0: tensor<*xf32>, %arg1 : tensor<*xf32>)
    -> (tensor<*xf32>) {
  %0 = "mhlo.reduce"(%arg0, %arg1) ({

  ^bb0(%arg2: tensor<*xf32>, %arg3: tensor<*xf32> ):
    %1 = "mhlo.add"(%arg2, %arg3) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
    "mhlo.return"(%1) : (tensor<*xf32>) -> ()

  }) {dimensions = dense<[0]> : tensor<1xi64>} : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>

  func.return %0: tensor<*xf32>
}

// -----

// CHECK-LABEL:    func @reduce_mixed_dynamism
func.func @reduce_mixed_dynamism(%arg0: tensor<4x4xf32>, %arg1 : tensor<f32>)
    -> (tensor<?xf32>) {
  %0 = mhlo.reduce(%arg0 init: %arg1)
    applies mhlo.multiply across dimensions = [1]
    : (tensor<4x4xf32>, tensor<f32>) -> tensor<?xf32>
  func.return %0: tensor<?xf32>
}

// -----

// CHECK-LABEL:    func @reduce_unranked
func.func @reduce_unranked(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>,
    %arg2: tensor<*xf32>, %arg3: tensor<*xf32>) -> (tensor<*xf32>, tensor<*xf32>) {
  %0:2 = "mhlo.reduce"(%arg0, %arg1, %arg2, %arg3) ({

  ^bb0(%arg4: tensor<*xf32>, %arg5: tensor<*xf32>, %arg6: tensor<*xf32>, %arg7: tensor<*xf32>):
    %1 = "mhlo.add"(%arg4, %arg6) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
    %2 = "mhlo.add"(%arg5, %arg7) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
    "mhlo.return"(%1, %2) : (tensor<*xf32>, tensor<*xf32>) -> ()

  }) {dimensions = dense<[1]> : tensor<1xi64>} : (tensor<4x4xf32>, tensor<4x4xf32>, tensor<*xf32>, tensor<*xf32>) -> (tensor<*xf32>, tensor<*xf32>)

  func.return %0#0, %0#1 : tensor<*xf32>, tensor<*xf32>
}

// -----

// CHECK-LABEL:    func @reduce_mix_rank_and_unranked
func.func @reduce_mix_rank_and_unranked(%arg0: tensor<4x4xf32>, %arg1: tensor<*xf32>,
    %arg2: tensor<4xf32>, %arg3: tensor<*xf32>) -> (tensor<4xf32>, tensor<*xf32>) {
  %0:2 = "mhlo.reduce"(%arg0, %arg1, %arg2, %arg3) ({

  ^bb0(%arg4: tensor<4xf32>, %arg5: tensor<*xf32>, %arg6: tensor<4xf32>, %arg7: tensor<*xf32>):
    %1 = "mhlo.add"(%arg4, %arg6) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
    %2 = "mhlo.add"(%arg5, %arg7) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
    "mhlo.return"(%1, %2) : (tensor<4xf32>, tensor<*xf32>) -> ()

  }) {dimensions = dense<[1]> : tensor<1xi64>} : (tensor<4x4xf32>, tensor<*xf32>, tensor<4xf32>, tensor<*xf32>) -> (tensor<4xf32>, tensor<*xf32>)

  func.return %0#0, %0#1 : tensor<4xf32>, tensor<*xf32>
}

// Next, we have the invalid testcases.

// -----

func.func @reduce_odd_num_args(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>,
    %arg2: tensor<f32>, %arg3: tensor<f32>) -> (tensor<?xf32>, tensor<?xf32>) {

  // As there are only 3 parameters of op, ODS takes in 1 for inputs and 1 for init_values
  // by SameVariadicOperandSize, so the Reduction-region expect 1*2=2 parameters.
  // expected-error@+1 {{Reduction-region must take 2 parameters, but takes 4 parameter(s)}}
  %0:2 = "mhlo.reduce"(%arg0, %arg1, %arg2) ({

  ^bb0(%arg4: tensor<f32>, %arg5: tensor<f32>, %arg6: tensor<f32>, %arg7: tensor<f32>):
    %1 = "mhlo.add"(%arg4, %arg6) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %2 = "mhlo.add"(%arg5, %arg7) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    "mhlo.return"(%arg5, %arg7) : (tensor<f32>, tensor<f32>) -> ()

  }) {dimensions = dense<[1]> : tensor<1xi64>} : (tensor<?x?xf32>, tensor<?x?xf32>, tensor<f32>) -> (tensor<?xf32>, tensor<?xf32>)

  func.return %0#0, %0#1 : tensor<?xf32>, tensor<?xf32>
}

// -----

func.func @reduce_diferent_input_shapes(%arg0: tensor<?x?xf32>, %arg1: tensor<?xf32>,
    %arg2: tensor<f32>, %arg3: tensor<f32>) -> (tensor<?xf32>, tensor<?xf32>) {

  // expected-error@+1 {{expects all inputs to have compatible shapes. Shape at input-index 1 is not compatible with shape at input-index 0}}
  %0:2 = "mhlo.reduce"(%arg0, %arg1, %arg2, %arg3) ({

  ^bb0(%arg4: tensor<f32>, %arg5: tensor<f32>, %arg6: tensor<f32>, %arg7: tensor<f32>):
    %1 = "mhlo.add"(%arg4, %arg6) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %2 = "mhlo.add"(%arg5, %arg7) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    "mhlo.return"(%1, %2) : (tensor<f32>, tensor<f32>) -> ()

  }) {dimensions = dense<[1]> : tensor<1xi64>} : (tensor<?x?xf32>, tensor<?xf32>, tensor<f32>, tensor<f32>) -> (tensor<?xf32>, tensor<?xf32>)

  func.return %0#0, %0#1 : tensor<?xf32>, tensor<?xf32>
}

// -----

func.func @reduce_diferent_input_shapes(%arg0: tensor<2x3xf32>, %arg1: tensor<3x2xf32>,
    %arg2: tensor<f32>, %arg3: tensor<f32>) -> (tensor<2xf32>, tensor<2xf32>) {

  // expected-error@+1 {{expects all inputs to have compatible shapes. Shape at input-index 1 is not compatible with shape at input-index 0}}
  %0:2 = "mhlo.reduce"(%arg0, %arg1, %arg2, %arg3) ({

  ^bb0(%arg4: tensor<f32>, %arg5: tensor<f32>, %arg6: tensor<f32>, %arg7: tensor<f32>):
    %1 = "mhlo.add"(%arg4, %arg6) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %2 = "mhlo.add"(%arg5, %arg7) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    "mhlo.return"(%1, %2) : (tensor<f32>, tensor<f32>) -> ()

  }) {dimensions = dense<[1]> : tensor<1xi64>} : (tensor<2x3xf32>, tensor<3x2xf32>, tensor<f32>, tensor<f32>) -> (tensor<2xf32>, tensor<2xf32>)

  func.return %0#0, %0#1 : tensor<2xf32>, tensor<2xf32>
}

// -----

func.func @reduce_oob_dims(%arg0: tensor<?x?xf32>, %arg1 : tensor<f32>)
    -> (tensor<?xf32>) {

  // expected-error@+1 {{Out-of-bounds dimension 2 for input-tensor rank: 2}}
  %0 = "mhlo.reduce"(%arg0, %arg1) ({

  ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32> ):
    %1 = "mhlo.add"(%arg2, %arg3) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    "mhlo.return"(%1) : (tensor<f32>) -> ()
  }) {dimensions = dense<[2]> : tensor<1xi64>} : (tensor<?x?xf32>, tensor<f32>) -> tensor<?xf32>

  func.return %0: tensor<?xf32>
}

// -----

func.func @reduce_duplicate_dims(%arg0: tensor<?x?xf32>, %arg1 : tensor<f32>)
    -> (tensor<?xf32>) {

  // expected-error@+1 {{Duplicate reduction dimension: 1}}
  %0 = "mhlo.reduce"(%arg0, %arg1) ({

  ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32> ):
    %1 = "mhlo.add"(%arg2, %arg3) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    "mhlo.return"(%1) : (tensor<f32>) -> ()

  }) {dimensions = dense<[1,1]> : tensor<2xi64>} : (tensor<?x?xf32>, tensor<f32>) -> tensor<?xf32>

  func.return %0: tensor<?xf32>
}

// -----

func.func @verify_reducer_function(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>,
    %arg2: tensor<f32>, %arg3: tensor<f32>) -> (tensor<?xf32>) {

  // expected-error@+1 {{Reduction-region must take 4 parameters, but takes 2 parameter(s)}}
  %0 = "mhlo.reduce"(%arg0, %arg1, %arg2, %arg3) ({

  ^bb0(%arg4: tensor<f32>, %arg5: tensor<f32>):
    %1 = "mhlo.add"(%arg4, %arg5) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    "mhlo.return"(%1) : (tensor<f32>) -> ()

  }) {dimensions = dense<[1]> : tensor<1xi64>} : (tensor<?x?xf32>, tensor<?x?xf32>, tensor<f32>, tensor<f32>) -> tensor<?xf32>

  func.return %0 : tensor<?xf32>
}

// -----

func.func @verify_reducer_function(%arg0: tensor<?x?xf32>, %arg1 : tensor<f32>)
    -> (tensor<f32>) {

  // expected-error@+1 {{The reduction-region expected to return some value(s)}}
  %0 = "mhlo.reduce"(%arg0, %arg1) ({

  ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32> ):
    %1 = "mhlo.add"(%arg2, %arg3) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    "mhlo.return"() : () -> ()

  }) {dimensions = dense<[1]> : tensor<1xi64>} : (tensor<?x?xf32>, tensor<f32>) -> tensor<f32>

    func.return %0: tensor<f32>
}

// -----

func.func @verify_reducer_function(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>,
    %arg2: tensor<f32>, %arg3: tensor<f32>) -> (tensor<?xf32>, tensor<?xf32>) {

  // expected-error@+1 {{Reduction-region here must produce 2 tensors, but produces 1 instead}}
  %0:2 = "mhlo.reduce"(%arg0, %arg1, %arg2, %arg3) ({

  ^bb0(%arg4: tensor<f32>, %arg5: tensor<f32>, %arg6: tensor<f32>, %arg7: tensor<f32>):
    %1 = "mhlo.add"(%arg4, %arg6) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    "mhlo.return"(%1) : (tensor<f32>) -> ()

  }) {dimensions = dense<[1]> : tensor<1xi64>} : (tensor<?x?xf32>, tensor<?x?xf32>, tensor<f32>, tensor<f32>) -> (tensor<?xf32>, tensor<?xf32>)

  func.return %0#0, %0#1 : tensor<?xf32>, tensor<?xf32>
}

// -----

func.func @verify_reducer_function(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xi32>,
    %arg2: tensor<f32>, %arg3: tensor<i32>) -> (tensor<?xf32>, tensor<?xi32>) {

  // expected-error@+1 {{Reduction-region here must produce tensor-typed result(s), but produces 'tuple<tensor<f32>, tensor<i32>>' instead}}
  %0:2 = "mhlo.reduce"(%arg0, %arg1, %arg2, %arg3) ({

  ^bb0(%arg4: tensor<f32>, %arg5: tensor<i32>, %arg6: tensor<f32>, %arg7: tensor<i32>):
    %1 = "mhlo.add"(%arg4, %arg6) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %2 = "mhlo.add"(%arg5, %arg7) : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %3 = "mhlo.tuple"(%1, %2) : (tensor<f32>, tensor<i32>) -> tuple<tensor<f32>, tensor<i32>>
    "mhlo.return"(%3, %1) : (tuple<tensor<f32>, tensor<i32>>, tensor<f32>) -> ()

  }) {dimensions = dense<[1]> : tensor<1xi64>} : (tensor<?x?xf32>, tensor<?x?xi32>, tensor<f32>, tensor<i32>) -> (tensor<?xf32>, tensor<?xi32>)

  func.return %0#0, %0#1 : tensor<?xf32>, tensor<?xi32>
}

// -----

func.func @verify_reducer_function(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xi32>,
    %arg2: tensor<f32>, %arg3: tensor<i32>) -> (tensor<?xf32>, tensor<?xi32>) {

  // expected-error@+1 {{The type of reduction-region's parameter at index 0 is different than the corresponding result type: 'tensor<f32>' vs 'tensor<i32>'}}
  %0:2 = "mhlo.reduce"(%arg0, %arg1, %arg2, %arg3) ({

  ^bb0(%arg4: tensor<f32>, %arg5: tensor<i32>, %arg6: tensor<f32>, %arg7: tensor<i32>):
    %1 = "mhlo.add"(%arg4, %arg6) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %2 = "mhlo.add"(%arg5, %arg7) : (tensor<i32>, tensor<i32>) -> tensor<i32>
    "mhlo.return"(%2, %1) : (tensor<i32>, tensor<f32>) -> ()

  }) {dimensions = dense<[1]> : tensor<1xi64>} : (tensor<?x?xf32>, tensor<?x?xi32>, tensor<f32>, tensor<i32>) -> (tensor<?xf32>, tensor<?xi32>)

  func.return %0#0, %0#1 : tensor<?xf32>, tensor<?xi32>
}

// -----

func.func @verify_reducer_function(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xi32>,
    %arg2: tensor<f32>, %arg3: tensor<i32>) -> (tensor<?xf32>, tensor<?xi32>) {

  // expected-error@+1 {{The type of reduction-region's parameter at index 1 is different than the corresponding result type: 'tensor<i32>' vs 'tensor<f32>'}}
  %0:2 = "mhlo.reduce"(%arg0, %arg1, %arg2, %arg3) ({

  ^bb0(%arg4: tensor<f32>, %arg5: tensor<i32>, %arg6: tensor<f32>, %arg7: tensor<i32>):
    %1 = "mhlo.add"(%arg4, %arg6) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %2 = "mhlo.add"(%arg5, %arg7) : (tensor<i32>, tensor<i32>) -> tensor<i32>
    "mhlo.return"(%1, %1) : (tensor<f32>, tensor<f32>) -> ()

  }) {dimensions = dense<[1]> : tensor<1xi64>} : (tensor<?x?xf32>, tensor<?x?xi32>, tensor<f32>, tensor<i32>) -> (tensor<?xf32>, tensor<?xi32>)

  func.return %0#0, %0#1 : tensor<?xf32>, tensor<?xi32>
}

// -----

func.func @verify_reducer_function(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xi32>,
    %arg2: tensor<f32>, %arg3: tensor<f32>) -> (tensor<?xf32>, tensor<?xf32>) {

  // expected-error@+1 {{The type of reduction-region's parameter at index 3 is different than the corresponding result type: 'tensor<i32>' vs 'tensor<f32>'}}
  %0:2 = "mhlo.reduce"(%arg0, %arg1, %arg2, %arg3) ({

  ^bb0(%arg4: tensor<f32>, %arg5: tensor<f32>, %arg6: tensor<f32>, %arg7: tensor<i32>):
    %1 = "mhlo.add"(%arg4, %arg6) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %2 = "mhlo.maximum"(%arg4, %arg6) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    "mhlo.return"(%1, %2) : (tensor<f32>, tensor<f32>) -> ()

  }) {dimensions = dense<[1]> : tensor<1xi64>} : (tensor<?x?xf32>, tensor<?x?xi32>, tensor<f32>, tensor<f32>) -> (tensor<?xf32>, tensor<?xi32>)

  func.return %0#0, %0#1 : tensor<?xf32>, tensor<?xi32>
}

// -----

func.func @verify_reducer_function(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xi32>,
    %arg2: tensor<f32>, %arg3: tensor<i32>) -> (tensor<?xf32>, tensor<?xi32>) {

  // expected-error@+1 {{The type of reduction-region's result type at index 1 differs from the op's corresponding init-value type: 'tensor<f32>' vs 'tensor<i32>'}}
  %0:2 = "mhlo.reduce"(%arg0, %arg1, %arg2, %arg3) ({

  ^bb0(%arg4: tensor<f32>, %arg5: tensor<f32>, %arg6: tensor<f32>, %arg7: tensor<f32>):
    %1 = "mhlo.add"(%arg4, %arg6) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %2 = "mhlo.add"(%arg5, %arg7) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    "mhlo.return"(%1, %2) : (tensor<f32>, tensor<f32>) -> ()

  }) {dimensions = dense<[1]> : tensor<1xi64>} : (tensor<?x?xf32>, tensor<?x?xi32>, tensor<f32>, tensor<i32>) -> (tensor<?xf32>, tensor<?xi32>)

  func.return %0#0, %0#1 : tensor<?xf32>, tensor<?xi32>
}

// -----

func.func @verify_reducer_function(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xi32>,
    %arg2: tensor<f32>, %arg3: tensor<f32>) -> (tensor<?xf32>, tensor<?xf32>) {

  // expected-error@+1 {{The element-type of reduction-region's argument at index 3 is expected to be 'i32', but got 'tensor<f32>'}}
  %0:2 = "mhlo.reduce"(%arg0, %arg1, %arg2, %arg3) ({

  ^bb0(%arg4: tensor<f32>, %arg5: tensor<f32>, %arg6: tensor<f32>, %arg7: tensor<f32>):
    %1 = "mhlo.add"(%arg4, %arg6) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %2 = "mhlo.maximum"(%arg4, %arg6) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    "mhlo.return"(%1, %2) : (tensor<f32>, tensor<f32>) -> ()

  }) {dimensions = dense<[1]> : tensor<1xi64>} : (tensor<?x?xf32>, tensor<?x?xi32>, tensor<f32>, tensor<f32>) -> (tensor<?xf32>, tensor<?xf32>)

  func.return %0#0, %0#1 : tensor<?xf32>, tensor<?xf32>
}

// -----

func.func @verify_reducer_function(%arg0: tensor<?xf32>, %arg1 : tensor<?xf32>)
    -> (tensor<f32>) {

  // expected-error@+1 {{The rank of reduction-region's argument at index 1 is expected to be <= 0, got 1}}
  %0 = "mhlo.reduce"(%arg0, %arg1) ({

  ^bb0(%arg2: tensor<?xf32>, %arg3: tensor<?xf32> ):
    %1 = "mhlo.add"(%arg2, %arg3) : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
    "mhlo.return"(%1) : (tensor<?xf32>) -> ()

  }) {dimensions = dense<[0]> : tensor<1xi64>} : (tensor<?xf32>, tensor<?xf32>) -> tensor<f32>

  func.return %0: tensor<f32>
}

// -----

func.func @verify_reducer_function(%arg0: tensor<8x5xf32>, %arg1 : tensor<4xf32>)
    -> (tensor<5xf32>) {

  // expected-error@+1 {{The shape of reduction-region's argument at index 1 is not compatible with that of reduce-op's input-parameter at index 0}}
  %0 = "mhlo.reduce"(%arg0, %arg1) ({

  ^bb0(%arg2: tensor<4xf32>, %arg3: tensor<4xf32> ):
    %1 = "mhlo.add"(%arg2, %arg3) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
    "mhlo.return"(%1) : (tensor<4xf32>) -> ()

  }) {dimensions = dense<[0]> : tensor<1xi64>} : (tensor<8x5xf32>, tensor<4xf32>) -> tensor<5xf32>

  func.return %0: tensor<5xf32>
}

// -----

// Verifies that dynamic input type is allowed with reducer function with static shapes.
func.func @verify_dynamic_operand(%arg0: tensor<8x?xf32>, %arg1 : tensor<4xf32>)
    -> (tensor<?xf32>) {

  %0 = "mhlo.reduce"(%arg0, %arg1) ({

  ^bb0(%arg2: tensor<4xf32>, %arg3: tensor<4xf32> ):
    %1 = "mhlo.add"(%arg2, %arg3) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
    "mhlo.return"(%1) : (tensor<4xf32>) -> ()

  }) {dimensions = dense<[0]> : tensor<1xi64>} : (tensor<8x?xf32>, tensor<4xf32>) -> tensor<?xf32>

  func.return %0: tensor<?xf32>
}

// -----

func.func @reduce_verify_rettype(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xi32>,
    %arg2: tensor<f32>, %arg3: tensor<i32>) -> (tensor<?xf32>) {

  // expected-error@+1 {{inferred type(s) 'tensor<?xf32>', 'tensor<?xi32>' are incompatible with return type(s) of operation 'tensor<?xf32>', 'tensor<?xi32>', 'tensor<?xi32>'}}
  %0:3 = "mhlo.reduce"(%arg0, %arg1, %arg2, %arg3) ({

  ^bb0(%arg4: tensor<f32>, %arg5: tensor<i32>, %arg6: tensor<f32>, %arg7: tensor<i32>):
    %1 = "mhlo.add"(%arg4, %arg6) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %2 = "mhlo.add"(%arg5, %arg7) : (tensor<i32>, tensor<i32>) -> tensor<i32>
    "mhlo.return"(%1, %2) : (tensor<f32>, tensor<i32>) -> ()

  }) {dimensions = dense<[1]> : tensor<1xi64>} : (tensor<?x?xf32>, tensor<?x?xi32>, tensor<f32>, tensor<i32>) -> (tensor<?xf32>, tensor<?xi32>, tensor<?xi32>)

  func.return %0#0: tensor<?xf32>
}

// -----

func.func @reduce_verify_rettype(%arg0: tensor<?x?xf32>, %arg1 : tensor<f32>)
    -> (tensor<?x?xi32>) {

  // expected-error@+1 {{'mhlo.reduce' op inferred type(s) 'tensor<?xf32>' are incompatible with return type(s) of operation 'tensor<?xf32>', 'tensor<?xf32>'}}
  %0:2 = "mhlo.reduce"(%arg0, %arg1) ({

  ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32> ):
    %1 = "mhlo.add"(%arg2, %arg3) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    "mhlo.return"(%1) : (tensor<f32>) -> ()

  }) {dimensions = dense<[1]> : tensor<1xi64>} : (tensor<?x?xf32>, tensor<f32>) -> (tensor<?xf32>, tensor<?xf32>)

  func.return %0#0: tensor<?xf32>
}

// -----

func.func @reduce_verify_rettype(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xi32>,
    %arg2: tensor<f32>, %arg3: tensor<i32>) -> (tensor<?xf32>) {

  // expected-error@+1 {{'mhlo.reduce' op inferred type(s) 'tensor<?xf32>', 'tensor<?xi32>' are incompatible with return type(s) of operation 'tensor<?xf32>', 'tensor<?x?xf32>'}}
  %0:2 = "mhlo.reduce"(%arg0, %arg1, %arg2, %arg3) ({

  ^bb0(%arg4: tensor<f32>, %arg5: tensor<i32>, %arg6: tensor<f32>, %arg7: tensor<i32>):
    %1 = "mhlo.add"(%arg4, %arg6) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %2 = "mhlo.add"(%arg5, %arg7) : (tensor<i32>, tensor<i32>) -> tensor<i32>
    "mhlo.return"(%1, %2) : (tensor<f32>, tensor<i32>) -> ()

  }) {dimensions = dense<[1]> : tensor<1xi64>} : (tensor<?x?xf32>, tensor<?x?xi32>, tensor<f32>, tensor<i32>) -> (tensor<?xf32>, tensor<?x?xf32>)

  func.return %0#0: tensor<?xf32>
}

// -----

func.func @reduce_verify_rettype(%arg0: tensor<?x?xf32>, %arg1 : tensor<f32>)
    -> (tensor<?xi32>) {

  // expected-error@+1 {{'mhlo.reduce' op inferred type(s) 'tensor<?xf32>' are incompatible with return type(s) of operation 'tensor<?xi32>'}}
  %0 = "mhlo.reduce"(%arg0, %arg1) ({

  ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32> ):
    %1 = "mhlo.add"(%arg2, %arg3) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    "mhlo.return"(%1) : (tensor<f32>) -> ()

  }) {dimensions = dense<[1]> : tensor<1xi64>} : (tensor<?x?xf32>, tensor<f32>) -> tensor<?xi32>

  func.return %0: tensor<?xi32>
}

// -----

func.func @reduce_verify_rettype(%arg0: tensor<?x?xf32>, %arg1 : tensor<f32>)
    -> (tensor<?x?xi32>) {

  // expected-error@+1 {{'mhlo.reduce' op inferred type(s) 'tensor<?xf32>' are incompatible with return type(s) of operation 'tensor<?x?xf32>'}}
  %0 = "mhlo.reduce"(%arg0, %arg1) ({

  ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32> ):
    %1 = "mhlo.add"(%arg2, %arg3) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    "mhlo.return"(%1) : (tensor<f32>) -> ()

  }) {dimensions = dense<[1]> : tensor<1xi64>} : (tensor<?x?xf32>, tensor<f32>) -> tensor<?x?xf32>

  func.return %0: tensor<?x?xf32>
}

// The following invalid cases arises while parsing a pretty-printed version of reduce-op will "non-eligible" inner-op.
// -----

func.func @reduce_parsing_pretty_reduce_non_commutative(%arg0: tensor<?x?xf32> , %arg1: tensor<f32> ) -> tensor<?xf32> {
  // expected-error@+1 {{expected the inner-op to be a commutative binary-op from mhlo dialect, zero region, producing single result}}
 %0 = mhlo.reduce(%arg0 init: %arg1) applies mhlo.divide across dimensions = [1] : (tensor<?x?xf32>, tensor<f32>) -> tensor<?xf32> loc("foo")
 func.return %0 : tensor<?xf32>
}

// -----

func.func @reduce_parsing_pretty_reduce_wrong_dialect(%arg0: tensor<?x?xf32> , %arg1: tensor<f32> ) -> tensor<?xf32> {
  // expected-error@+1 {{expected the inner-op to be a commutative binary-op from mhlo dialect, zero region, producing single result}}
 %0 = mhlo.reduce(%arg0 init: %arg1) applies std.add across dimensions = [1] : (tensor<?x?xf32>, tensor<f32>) -> tensor<?xf32> loc("foo")
 func.return %0 : tensor<?xf32>
}

// -----

func.func @reduce_parsing_pretty_reduce_non_binary(%arg0: tensor<?x?xf32> , %arg1: tensor<f32> ) -> tensor<?xf32> {
  // expected-error@+1 {{expected the inner-op to be a commutative binary-op from mhlo dialect, zero region, producing single result}}
 %0 = mhlo.reduce(%arg0 init: %arg1) applies mhlo.reshape across dimensions = [1] : (tensor<?x?xf32>, tensor<f32>) -> tensor<?xf32> loc("foo")
 func.return %0 : tensor<?xf32>
}
