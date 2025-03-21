// RUN: mlir-hlo-opt %s -verify-diagnostics -split-input-file -allow-unregistered-dialect | FileCheck %s

// CHECK: func @while
func.func @while(%arg0: tensor<4xf32>, %arg1: tensor<f32>, %arg2: tensor<f32>, %arg3: tensor<4xf32>, %arg4: tensor<f32>, %arg5: tensor<f32>, %arg6: tensor<f32>, %arg7: tensor<f32>, %arg8: tensor<i32>) -> (tensor<i32>, tensor<i32>, tensor<i32>) {
  %cst = arith.constant dense<-1> : tensor<i32>
  %cst_0 = arith.constant dense<1> : tensor<i32>
  %cst_1 = arith.constant dense<0> : tensor<i32>
  %cst_2 = arith.constant dense<1000> : tensor<i32>
  %1:3 = "mhlo.while"(%cst_1, %cst, %cst_2) ({
  ^bb0(%arg9: tensor<i32>, %arg10: tensor<i32>, %arg11: tensor<i32>):
    %4 = "mhlo.compare"(%arg9, %arg11) {comparison_direction = #mhlo<comparison_direction LT>} : (tensor<i32>, tensor<i32>) -> tensor<i1>
    "mhlo.return"(%4) : (tensor<i1>) -> ()
  },  {
  ^bb0(%arg9: tensor<i32>, %arg10: tensor<i32>, %arg11: tensor<i32>):
    %3 = mhlo.add %arg9, %cst_0 : tensor<i32>
    "mhlo.return"(%3, %arg10, %arg11) : (tensor<i32>, tensor<i32>, tensor<i32>) -> ()
  }) : (tensor<i32>, tensor<i32>, tensor<i32>) -> (tensor<i32>, tensor<i32>, tensor<i32>)
  func.return %1#0, %1#2, %1#2: tensor<i32>, tensor<i32>, tensor<i32>
}

// -----

// CHECK-LABEL: while_with_different_types
func.func @while_with_different_types(%arg0: tensor<3xf32>) -> tensor<3xf32> {
  %cst_0 = arith.constant dense<0> : tensor<1xi32>
  %cst_1 = arith.constant dense<[100, 100]> : tensor<2xi32>
  %cst_2 = arith.constant dense<1.00> : tensor<1xf32>
  %1:4 = "mhlo.while"(%cst_0, %cst_1, %cst_2, %arg0) ({
  ^bb0(%arg1: tensor<1xi32>, %arg2: tensor<2xi32>, %arg3: tensor<1xf32>, %arg4: tensor<3xf32>):
    %2 = arith.constant dense<0> : tensor<i32>
    %3 = "mhlo.slice"(%arg2) <{limit_indices = dense<[1]> : tensor<1xi64>, start_indices = dense<[0]> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>}> : (tensor<2xi32>) -> tensor<1xi32>
    %4 = "mhlo.compare"(%arg1, %3) {comparison_direction = #mhlo<comparison_direction LT>} : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    %5 = "mhlo.reshape"(%4) : (tensor<1xi1>) -> tensor<i1>
    "mhlo.return"(%5) : (tensor<i1>) -> ()
  },  {
  ^bb0(%arg1: tensor<1xi32>, %arg2: tensor<2xi32>, %arg3: tensor<1xf32>, %arg4: tensor<3xf32>):
    %3 = "mhlo.broadcast_in_dim"(%arg3) <{broadcast_dimensions = dense<0> : tensor<1xi64>}> : (tensor<1xf32>) -> tensor<3xf32>
    %4 = mhlo.add %3, %arg4 : tensor<3xf32>
    "mhlo.return"(%arg1, %arg2, %arg3, %4) : (tensor<1xi32>, tensor<2xi32>, tensor<1xf32>, tensor<3xf32>) -> ()
  }) : (tensor<1xi32>, tensor<2xi32>, tensor<1xf32>, tensor<3xf32>) -> (tensor<1xi32>, tensor<2xi32>, tensor<1xf32>, tensor<3xf32>)
  func.return %1#3: tensor<3xf32>
}

// -----

// CHECK-LABEL: while_dynamic
func.func @while_dynamic(%arg0: tensor<3xf32>) -> tensor<?xf32> {
  %cst_0 = arith.constant dense<0> : tensor<1xi32>
  %cst_1 = arith.constant dense<[100, 100]> : tensor<2xi32>
  %cst_2 = arith.constant dense<1.00> : tensor<1xf32>
  %1:4 = "mhlo.while"(%cst_0, %cst_1, %cst_2, %arg0) ({
  ^bb0(%arg1: tensor<1xi32>, %arg2: tensor<2xi32>, %arg3: tensor<1xf32>, %arg4: tensor<?xf32>):
    %2 = arith.constant dense<0> : tensor<i32>
    %3 = "mhlo.slice"(%arg2) <{limit_indices = dense<[1]> : tensor<1xi64>, start_indices = dense<[0]> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>}> : (tensor<2xi32>) -> tensor<1xi32>
    %4 = "mhlo.compare"(%arg1, %3) {comparison_direction = #mhlo<comparison_direction LT>} : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    %5 = "mhlo.reshape"(%4) : (tensor<1xi1>) -> tensor<i1>
    "mhlo.return"(%5) : (tensor<i1>) -> ()
  },  {
  ^bb0(%arg1: tensor<1xi32>, %arg2: tensor<?xi32>, %arg3: tensor<1xf32>, %arg4: tensor<3xf32>):
    %3 = "mhlo.broadcast_in_dim"(%arg3) <{broadcast_dimensions = dense<0> : tensor<1xi64>}> : (tensor<1xf32>) -> tensor<3xf32>
    %4 = mhlo.add %3, %arg4 : tensor<3xf32>
    "mhlo.return"(%arg1, %arg2, %arg3, %4) : (tensor<1xi32>, tensor<?xi32>, tensor<1xf32>, tensor<3xf32>) -> ()
  }) : (tensor<1xi32>, tensor<2xi32>, tensor<1xf32>, tensor<3xf32>) -> (tensor<1xi32>, tensor<2xi32>, tensor<1xf32>, tensor<?xf32>)
  func.return %1#3: tensor<?xf32>
}

// Negative tests below

// -----

func.func @while_with_invalid_types(%arg0: tensor<3xf32>) -> tensor<3xf32> {
  %cst_0 = arith.constant dense<0> : tensor<1xi32>
  %cst_1 = arith.constant dense<[100, 100]> : tensor<2xi32>
  %cst_2 = arith.constant dense<1.00> : tensor<1xf32>
  // expected-error @+2 {{'mhlo.while' op failed to infer returned types}}
  // expected-error @+1 {{inferred type(s) 'tensor<1xi32>', 'tensor<2xi32>', 'tensor<1xf32>', 'tensor<3xf32>' are incompatible with return type(s) of operation 'tensor<1xi32>', 'tensor<2xi32>', 'tensor<3xf32>', 'tensor<1xf32>'}}
  %1:4 = "mhlo.while"(%cst_0, %cst_1, %cst_2, %arg0) ({
  ^bb0(%arg1: tensor<1xi32>, %arg2: tensor<2xi32>, %arg3: tensor<1xf32>, %arg4: tensor<3xf32>):
    %2 = arith.constant dense<0> : tensor<i32>
    %3 = "mhlo.slice"(%arg2) <{limit_indices = dense<[1]> : tensor<1xi64>, start_indices = dense<[0]> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>}> : (tensor<2xi32>) -> tensor<1xi32>
    %4 = "mhlo.compare"(%arg1, %3) {comparison_direction = #mhlo<comparison_direction LT>} : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    %5 = "mhlo.reshape"(%4) : (tensor<1xi1>) -> tensor<i1>
    "mhlo.return"(%5) : (tensor<i1>) -> ()
  },  {
  ^bb0(%arg1: tensor<1xi32>, %arg2: tensor<2xi32>, %arg3: tensor<1xf32>, %arg4: tensor<3xf32>):
    %3 = "mhlo.broadcast_in_dim"(%arg3) <{broadcast_dimensions = dense<0> : tensor<1xi64>}> : (tensor<1xf32>) -> tensor<3xf32>
    %4 = mhlo.add %3, %arg4 : tensor<3xf32>
    "mhlo.return"(%arg1, %arg2, %arg3, %4) : (tensor<1xi32>, tensor<2xi32>, tensor<1xf32>, tensor<3xf32>) -> ()
  }) : (tensor<1xi32>, tensor<2xi32>, tensor<1xf32>, tensor<3xf32>) -> (tensor<1xi32>, tensor<2xi32>, tensor<3xf32>, tensor<1xf32>)
  func.return %1#2: tensor<3xf32>
}

// -----

func.func @while_with_invalid_tuples(%arg0: tensor<3xf32>) -> tensor<3xf32> {
  %cst_0 = arith.constant dense<0> : tensor<1xi32>
  %cst_1 = arith.constant dense<[100, 100]> : tensor<2xi32>
  %cst_2 = arith.constant dense<1.00> : tensor<1xf32>
  %0 = "mhlo.tuple"(%arg0, %cst_2) : (tensor<3xf32>, tensor<1xf32>) -> tuple<tensor<3xf32>, tensor<1xf32>>
  %1 = "mhlo.tuple"(%cst_1, %0) : (tensor<2xi32>, tuple<tensor<3xf32>, tensor<1xf32>>) -> tuple<tensor<2xi32>, tuple<tensor<3xf32>, tensor<1xf32>>>
  // expected-error-re@+1 {{operand #1 must be variadic of ranked tensor of {{.*}}, but got 'tuple<tensor<2xi32>, tuple<tensor<3xf32>, tensor<1xf32>>>'}}
  %2:2 = "mhlo.while"(%cst_0, %1) ({
  ^bb0(%arg1: tensor<1xi32>, %arg2: tuple<tensor<2xi32>, tuple<tensor<1xf32>, tensor<3xf32>>>):
    %t0 = "mhlo.get_tuple_element"(%arg2) {index = 0 : i32} : (tuple<tensor<2xi32>, tuple<tensor<1xf32>, tensor<3xf32>>>) -> tensor<2xi32>
    %3 = arith.constant dense<0> : tensor<i32>
    %4 = "mhlo.slice"(%t0) <{limit_indices = dense<[1]> : tensor<1xi64>, start_indices = dense<[0]> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>}> : (tensor<2xi32>) -> tensor<1xi32>
    %5 = "mhlo.compare"(%arg1, %4) {comparison_direction = #mhlo<comparison_direction LT>} : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    "mhlo.return"(%5) : (tensor<1xi1>) -> ()
  },  {
  ^bb0(%arg1: tensor<1xi32>, %arg2: tuple<tensor<2xi32>, tuple<tensor<1xf32>, tensor<3xf32>>>):
    %t0 = "mhlo.get_tuple_element"(%arg2) {index = 0 : i32} : (tuple<tensor<2xi32>, tuple<tensor<1xf32>, tensor<3xf32>>>) -> tensor<2xi32>
    %t1_2 = "mhlo.get_tuple_element"(%arg2) {index = 1 : i32} : (tuple<tensor<2xi32>, tuple<tensor<1xf32>, tensor<3xf32>>>) -> tuple<tensor<1xf32>, tensor<3xf32>>
    %t1 = "mhlo.get_tuple_element"(%t1_2) {index = 0 : i32} : (tuple<tensor<1xf32>, tensor<3xf32>>) -> tensor<1xf32>
    %t2 = "mhlo.get_tuple_element"(%t1_2) {index = 1 : i32} : (tuple<tensor<1xf32>, tensor<3xf32>>) -> tensor<3xf32>
    %3 = "mhlo.broadcast_in_dim"(%t1) <{broadcast_dimensions = dense<0> : tensor<1xi64>}> : (tensor<1xf32>) -> tensor<3xf32>
    %4 = mhlo.add %3, %t2 : tensor<3xf32>
    %5 = "mhlo.tuple"(%t1, %4) : (tensor<1xf32>, tensor<3xf32>) -> tuple<tensor<1xf32>, tensor<3xf32>>
    %6 = "mhlo.tuple"(%t0, %5) : (tensor<2xi32>, tuple<tensor<1xf32>, tensor<3xf32>>) -> tuple<tensor<2xi32>, tuple<tensor<1xf32>, tensor<3xf32>>>
    "mhlo.return"(%arg1, %6) : (tensor<1xi32>, tuple<tensor<2xi32>, tuple<tensor<1xf32>, tensor<3xf32>>>) -> ()
  }) : (tensor<1xi32>, tuple<tensor<2xi32>, tuple<tensor<3xf32>, tensor<1xf32>>>) -> (tensor<1xi32>, tuple<tensor<2xi32>, tuple<tensor<1xf32>, tensor<3xf32>>>)
  %3 = "mhlo.get_tuple_element"(%2#1) {index = 1 : i32} : (tuple<tensor<2xi32>, tuple<tensor<1xf32>, tensor<3xf32>>>) -> tuple<tensor<1xf32>, tensor<3xf32>>
  %4 = "mhlo.get_tuple_element"(%3) {index = 1 : i32} : (tuple<tensor<1xf32>, tensor<3xf32>>) -> tensor<3xf32>
  func.return %4: tensor<3xf32>
}
// -----

func.func @while_with_different_types(%arg0: tensor<3xf32>) -> tensor<3xf32> {
  %cst_0 = arith.constant dense<0> : tensor<1xi32>
  %cst_1 = arith.constant dense<[100, 100]> : tensor<2xi32>
  %cst_2 = arith.constant dense<1.00> : tensor<1xf32>
  // expected-error @+1 {{expect operands to be compatible with condition block arguments but got 'tensor<1xi32>', 'tensor<2xi32>', 'tensor<1xf32>', 'tensor<3xf32>' vs 'tensor<1xi32>', 'tensor<2xi32>', 'tensor<3xf32>', 'tensor<3xf32>'}}
  %1:4 = "mhlo.while"(%cst_0, %cst_1, %cst_2, %arg0) ({
  ^bb0(%arg1: tensor<1xi32>, %arg2: tensor<2xi32>, %arg3: tensor<3xf32>, %arg4: tensor<3xf32>):
    %2 = arith.constant dense<0> : tensor<i32>
    %3 = "mhlo.slice"(%arg2) <{limit_indices = dense<[1]> : tensor<1xi64>, start_indices = dense<[0]> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>}> : (tensor<2xi32>) -> tensor<1xi32>
    %4 = "mhlo.compare"(%arg1, %3) {comparison_direction = #mhlo<comparison_direction LT>} : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    %5 = "mhlo.reshape"(%4) : (tensor<1xi1>) -> tensor<i1>
    "mhlo.return"(%5) : (tensor<i1>) -> ()
  },  {
  ^bb0(%arg1: tensor<1xi32>, %arg2: tensor<2xi32>, %arg3: tensor<1xf32>, %arg4: tensor<3xf32>):
    %3 = "mhlo.broadcast_in_dim"(%arg3) <{broadcast_dimensions = dense<0> : tensor<1xi64>}> : (tensor<1xf32>) -> tensor<3xf32>
    %4 = mhlo.add %3, %arg4 : tensor<3xf32>
    "mhlo.return"(%arg1, %arg2, %arg3, %4) : (tensor<1xi32>, tensor<2xi32>, tensor<1xf32>, tensor<3xf32>) -> ()
  }) : (tensor<1xi32>, tensor<2xi32>, tensor<1xf32>, tensor<3xf32>) -> (tensor<1xi32>, tensor<2xi32>, tensor<1xf32>, tensor<3xf32>)
  func.return %1#3: tensor<3xf32>
}

// -----

func.func @while_with_different_types(%arg0: tensor<3xf32>) -> tensor<3xf32> {
  %cst_0 = arith.constant dense<0> : tensor<1xi32>
  %cst_1 = arith.constant dense<[100, 100]> : tensor<2xi32>
  %cst_2 = arith.constant dense<1.00> : tensor<1xf32>
  // expected-error @+1 {{expect operands to be compatible with body block arguments but got 'tensor<1xi32>', 'tensor<2xi32>', 'tensor<1xf32>', 'tensor<3xf32>' vs 'tensor<1xi32>', 'tensor<3xi32>', 'tensor<1xf32>', 'tensor<3xf32>'}}
  %1:4 = "mhlo.while"(%cst_0, %cst_1, %cst_2, %arg0) ({
  ^bb0(%arg1: tensor<1xi32>, %arg2: tensor<2xi32>, %arg3: tensor<1xf32>, %arg4: tensor<3xf32>):
    %2 = arith.constant dense<0> : tensor<i32>
    %3 = "mhlo.slice"(%arg2) <{limit_indices = dense<[1]> : tensor<1xi64>, start_indices = dense<[0]> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>}> : (tensor<2xi32>) -> tensor<1xi32>
    %4 = "mhlo.compare"(%arg1, %3) {comparison_direction = #mhlo<comparison_direction LT>} : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    %5 = "mhlo.reshape"(%4) : (tensor<1xi1>) -> tensor<i1>
    "mhlo.return"(%5) : (tensor<i1>) -> ()
  },  {
  ^bb0(%arg1: tensor<1xi32>, %arg2: tensor<3xi32>, %arg3: tensor<1xf32>, %arg4: tensor<3xf32>):
    %3 = "mhlo.broadcast_in_dim"(%arg3) <{broadcast_dimensions = dense<0> : tensor<1xi64>}> : (tensor<1xf32>) -> tensor<3xf32>
    %4 = mhlo.add %3, %arg4 : tensor<3xf32>
    "mhlo.return"(%arg1, %arg2, %arg3, %4) : (tensor<1xi32>, tensor<3xi32>, tensor<1xf32>, tensor<3xf32>) -> ()
  }) : (tensor<1xi32>, tensor<2xi32>, tensor<1xf32>, tensor<3xf32>) -> (tensor<1xi32>, tensor<2xi32>, tensor<1xf32>, tensor<3xf32>)
  func.return %1#3: tensor<3xf32>
}

// -----

func.func @while_with_block_count_mismatch(%arg0: tensor<3xf32>) -> tensor<3xf32> {
  %cst_0 = arith.constant dense<0> : tensor<1xi32>
  %cst_1 = arith.constant dense<[100, 100]> : tensor<2xi32>
  %cst_2 = arith.constant dense<1.00> : tensor<1xf32>
  // expected-error @+1 {{expect operands to be compatible with condition block arguments but got 'tensor<1xi32>', 'tensor<2xi32>', 'tensor<1xf32>', 'tensor<3xf32>' vs 'tensor<1xi32>', 'tensor<2xi32>', 'tensor<1xf32>'}}
  %1:4 = "mhlo.while"(%cst_0, %cst_1, %cst_2, %arg0) ({
  ^bb0(%arg1: tensor<1xi32>, %arg2: tensor<2xi32>, %arg3: tensor<1xf32>):
    %2 = arith.constant dense<0> : tensor<i32>
    %3 = "mhlo.slice"(%arg2) <{limit_indices = dense<[1]> : tensor<1xi64>, start_indices = dense<[0]> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>}> : (tensor<2xi32>) -> tensor<1xi32>
    %4 = "mhlo.compare"(%arg1, %3) {comparison_direction = #mhlo<comparison_direction LT>} : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    %5 = "mhlo.reshape"(%4) : (tensor<1xi1>) -> tensor<i1>
    "mhlo.return"(%5) : (tensor<i1>) -> ()
  },  {
  ^bb0(%arg1: tensor<1xi32>, %arg2: tensor<3xi32>, %arg3: tensor<1xf32>, %arg4: tensor<3xf32>):
    %3 = "mhlo.broadcast_in_dim"(%arg3) <{broadcast_dimensions = dense<0> : tensor<1xi64>}> : (tensor<1xf32>) -> tensor<3xf32>
    %4 = mhlo.add %3, %arg4 : tensor<3xf32>
    "mhlo.return"(%arg1, %arg2, %arg3, %4) : (tensor<1xi32>, tensor<3xi32>, tensor<1xf32>, tensor<3xf32>) -> ()
  }) : (tensor<1xi32>, tensor<2xi32>, tensor<1xf32>, tensor<3xf32>) -> (tensor<1xi32>, tensor<2xi32>, tensor<1xf32>, tensor<3xf32>)
  func.return %1#3: tensor<3xf32>
}

// -----

func.func @while_with_block_count_mismatch(%arg0: tensor<3xf32>) -> tensor<3xf32> {
  %cst_0 = arith.constant dense<0> : tensor<1xi32>
  %cst_1 = arith.constant dense<[100, 100]> : tensor<2xi32>
  %cst_2 = arith.constant dense<1.00> : tensor<1xf32>
  // expected-error @+1 {{expect operands to be compatible with body block arguments but got 'tensor<1xi32>', 'tensor<2xi32>', 'tensor<1xf32>', 'tensor<3xf32>' vs 'tensor<1xi32>', 'tensor<3xi32>', 'tensor<1xf32>'}}
  %1:4 = "mhlo.while"(%cst_0, %cst_1, %cst_2, %arg0) ({
  ^bb0(%arg1: tensor<1xi32>, %arg2: tensor<2xi32>, %arg3: tensor<1xf32>, %arg4: tensor<3xf32>):
    %2 = arith.constant dense<0> : tensor<i32>
    %3 = "mhlo.slice"(%arg2) <{limit_indices = dense<[1]> : tensor<1xi64>, start_indices = dense<[0]> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>}> : (tensor<2xi32>) -> tensor<1xi32>
    %4 = "mhlo.compare"(%arg1, %3) {comparison_direction = #mhlo<comparison_direction LT>} : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    %5 = "mhlo.reshape"(%4) : (tensor<1xi1>) -> tensor<i1>
    "mhlo.return"(%5) : (tensor<i1>) -> ()
  },  {
  ^bb0(%arg1: tensor<1xi32>, %arg2: tensor<3xi32>, %arg3: tensor<1xf32>):
    %3 = "mhlo.broadcast_in_dim"(%arg3) <{broadcast_dimensions = dense<0> : tensor<1xi64>}> : (tensor<1xf32>) -> tensor<3xf32>
    "mhlo.return"(%arg1, %arg2, %arg3, %3) : (tensor<1xi32>, tensor<3xi32>, tensor<1xf32>, tensor<3xf32>) -> ()
  }) : (tensor<1xi32>, tensor<2xi32>, tensor<1xf32>, tensor<3xf32>) -> (tensor<1xi32>, tensor<2xi32>, tensor<1xf32>, tensor<3xf32>)
  func.return %1#3: tensor<3xf32>
}

// -----

func.func @while_with_cond_return_width_mismatch(%arg0: tensor<3xf32>) -> tensor<3xf32> {
  %cst_0 = arith.constant dense<0> : tensor<1xi32>
  %cst_1 = arith.constant dense<[100, 100]> : tensor<2xi32>
  %cst_2 = arith.constant dense<1.00> : tensor<1xf32>
  // expected-error @+1 {{expect condition block return a zero-ranked tensor of i1 but got 'tensor<i32>'}}
  %1:4 = "mhlo.while"(%cst_0, %cst_1, %cst_2, %arg0) ({
  ^bb0(%arg1: tensor<1xi32>, %arg2: tensor<2xi32>, %arg3: tensor<1xf32>, %arg4: tensor<3xf32>):
    %2 = "mhlo.reshape"(%arg1) : (tensor<1xi32>) -> tensor<i32>
    "mhlo.return"(%2) : (tensor<i32>) -> ()
  },  {
  ^bb0(%arg1: tensor<1xi32>, %arg2: tensor<2xi32>, %arg3: tensor<1xf32>, %arg4: tensor<3xf32>):
    %3 = "mhlo.broadcast_in_dim"(%arg3) <{broadcast_dimensions = dense<0> : tensor<1xi64>}> : (tensor<1xf32>) -> tensor<3xf32>
    %4 = mhlo.add %3, %arg4 : tensor<3xf32>
    "mhlo.return"(%arg1, %arg2, %arg3, %4) : (tensor<1xi32>, tensor<2xi32>, tensor<1xf32>, tensor<3xf32>) -> ()
  }) : (tensor<1xi32>, tensor<2xi32>, tensor<1xf32>, tensor<3xf32>) -> (tensor<1xi32>, tensor<2xi32>, tensor<1xf32>, tensor<3xf32>)
  func.return %1#3: tensor<3xf32>
}

// -----

func.func @while_with_cond_return_rank_mismatch(%arg0: tensor<3xf32>) -> tensor<3xf32> {
  %cst_0 = arith.constant dense<0> : tensor<1xi32>
  %cst_1 = arith.constant dense<[100, 100]> : tensor<2xi32>
  %cst_2 = arith.constant dense<1.00> : tensor<1xf32>
  // expected-error @+1 {{expect condition block return a zero-ranked tensor of i1 but got 'tensor<1xi1>'}}
  %1:4 = "mhlo.while"(%cst_0, %cst_1, %cst_2, %arg0) ({
  ^bb0(%arg1: tensor<1xi32>, %arg2: tensor<2xi32>, %arg3: tensor<1xf32>, %arg4: tensor<3xf32>):
    %3 = "mhlo.slice"(%arg2) <{limit_indices = dense<[1]> : tensor<1xi64>, start_indices = dense<[0]> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>}> : (tensor<2xi32>) -> tensor<1xi32>
    %4 = "mhlo.compare"(%arg1, %3) {comparison_direction = #mhlo<comparison_direction LT>} : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    "mhlo.return"(%4) : (tensor<1xi1>) -> ()
  },  {
  ^bb0(%arg1: tensor<1xi32>, %arg2: tensor<2xi32>, %arg3: tensor<1xf32>, %arg4: tensor<3xf32>):
    %3 = "mhlo.broadcast_in_dim"(%arg3) <{broadcast_dimensions = dense<0> : tensor<1xi64>}> : (tensor<1xf32>) -> tensor<3xf32>
    %4 = mhlo.add %3, %arg4 : tensor<3xf32>
    "mhlo.return"(%arg1, %arg2, %arg3, %4) : (tensor<1xi32>, tensor<2xi32>, tensor<1xf32>, tensor<3xf32>) -> ()
  }) : (tensor<1xi32>, tensor<2xi32>, tensor<1xf32>, tensor<3xf32>) -> (tensor<1xi32>, tensor<2xi32>, tensor<1xf32>, tensor<3xf32>)
  func.return %1#3: tensor<3xf32>
}

// -----

func.func @while_with_cond_return_type_mismatch(%arg0: tensor<3xf32>) -> tensor<3xf32> {
  %cst_0 = arith.constant dense<0> : tensor<1xi32>
  %cst_1 = arith.constant dense<[100, 100]> : tensor<2xi32>
  %cst_2 = arith.constant dense<1.00> : tensor<1xf32>
  // expected-error @+1 {{expect condition block return a zero-ranked tensor of i1 but got 'tensor<f32>'}}
  %1:4 = "mhlo.while"(%cst_0, %cst_1, %cst_2, %arg0) ({
  ^bb0(%arg1: tensor<1xi32>, %arg2: tensor<2xi32>, %arg3: tensor<1xf32>, %arg4: tensor<3xf32>):
    %2 = "mhlo.reshape"(%arg3) : (tensor<1xf32>) -> tensor<f32>
    "mhlo.return"(%2) : (tensor<f32>) -> ()
  },  {
  ^bb0(%arg1: tensor<1xi32>, %arg2: tensor<2xi32>, %arg3: tensor<1xf32>, %arg4: tensor<3xf32>):
    %3 = "mhlo.broadcast_in_dim"(%arg3) <{broadcast_dimensions = dense<0> : tensor<1xi64>}> : (tensor<1xf32>) -> tensor<3xf32>
    %4 = mhlo.add %3, %arg4 : tensor<3xf32>
    "mhlo.return"(%arg1, %arg2, %arg3, %4) : (tensor<1xi32>, tensor<2xi32>, tensor<1xf32>, tensor<3xf32>) -> ()
  }) : (tensor<1xi32>, tensor<2xi32>, tensor<1xf32>, tensor<3xf32>) -> (tensor<1xi32>, tensor<2xi32>, tensor<1xf32>, tensor<3xf32>)
  func.return %1#3: tensor<3xf32>
}

// -----

func.func @while_with_body_return_mismatch(%arg0: tensor<3xf32>) -> tensor<3xf32> {
  %cst_0 = arith.constant dense<0> : tensor<1xi32>
  %cst_1 = arith.constant dense<[100, 100]> : tensor<2xi32>
  %cst_2 = arith.constant dense<1.00> : tensor<1xf32>
  // expected-error @+1 {{expect operands to be compatible with body block return types but got 'tensor<1xi32>', 'tensor<2xi32>', 'tensor<1xf32>', 'tensor<3xf32>' vs 'tensor<1xi32>', 'tensor<2xi32>', 'tensor<1xf32>', 'tensor<1xf32>'}}
  %1:4 = "mhlo.while"(%cst_0, %cst_1, %cst_2, %arg0) ({
  ^bb0(%arg1: tensor<1xi32>, %arg2: tensor<2xi32>, %arg3: tensor<1xf32>, %arg4: tensor<3xf32>):
    %2 = arith.constant dense<0> : tensor<i32>
    %3 = "mhlo.slice"(%arg2) <{limit_indices = dense<[1]> : tensor<1xi64>, start_indices = dense<[0]> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>}> : (tensor<2xi32>) -> tensor<1xi32>
    %4 = "mhlo.compare"(%arg1, %3) {comparison_direction = #mhlo<comparison_direction LT>} : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    %5 = "mhlo.reshape"(%4) : (tensor<1xi1>) -> tensor<i1>
    "mhlo.return"(%5) : (tensor<i1>) -> ()
  },  {
  ^bb0(%arg1: tensor<1xi32>, %arg2: tensor<2xi32>, %arg3: tensor<1xf32>, %arg4: tensor<3xf32>):
    "mhlo.return"(%arg1, %arg2, %arg3, %arg3) : (tensor<1xi32>, tensor<2xi32>, tensor<1xf32>, tensor<1xf32>) -> ()
  }) : (tensor<1xi32>, tensor<2xi32>, tensor<1xf32>, tensor<3xf32>) -> (tensor<1xi32>, tensor<2xi32>, tensor<1xf32>, tensor<3xf32>)
  func.return %1#3: tensor<3xf32>
}

// -----

func.func @while_with_multiple_operand_in_cond_return(%arg0: tensor<3xf32>) -> tensor<3xf32> {
  %cst_0 = arith.constant dense<0> : tensor<1xi32>
  %cst_1 = arith.constant dense<[100, 100]> : tensor<2xi32>
  %cst_2 = arith.constant dense<1.00> : tensor<1xf32>
  // expected-error @+1 {{expect condition body returns a single value but got 2}}
  %1:4 = "mhlo.while"(%cst_0, %cst_1, %cst_2, %arg0) ({
  ^bb0(%arg1: tensor<1xi32>, %arg2: tensor<2xi32>, %arg3: tensor<1xf32>, %arg4: tensor<3xf32>):
    %2 = arith.constant dense<0> : tensor<i32>
    %3 = "mhlo.slice"(%arg2) <{limit_indices = dense<[1]> : tensor<1xi64>, start_indices = dense<[0]> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>}> : (tensor<2xi32>) -> tensor<1xi32>
    %4 = "mhlo.compare"(%arg1, %3) {comparison_direction = #mhlo<comparison_direction LT>} : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    %5 = "mhlo.reshape"(%4) : (tensor<1xi1>) -> tensor<i1>
    "mhlo.return"(%5, %5) : (tensor<i1>, tensor<i1>) -> ()
  },  {
  ^bb0(%arg1: tensor<1xi32>, %arg2: tensor<2xi32>, %arg3: tensor<1xf32>, %arg4: tensor<3xf32>):
    %3 = "mhlo.broadcast_in_dim"(%arg3) <{broadcast_dimensions = dense<0> : tensor<1xi64>}> : (tensor<1xf32>) -> tensor<3xf32>
    %4 = mhlo.add %3, %arg4 : tensor<3xf32>
    "mhlo.return"(%arg1, %arg2, %arg3, %4) : (tensor<1xi32>, tensor<2xi32>, tensor<1xf32>, tensor<3xf32>) -> ()
  }) : (tensor<1xi32>, tensor<2xi32>, tensor<1xf32>, tensor<3xf32>) -> (tensor<1xi32>, tensor<2xi32>, tensor<1xf32>, tensor<3xf32>)
  func.return %1#3: tensor<3xf32>
}

// -----

func.func @while_mismatch_operand_count_with_body_return(%arg0: tensor<3xf32>) -> tensor<3xf32> {
  %cst_0 = arith.constant dense<0> : tensor<1xi32>
  %cst_1 = arith.constant dense<[100, 100]> : tensor<2xi32>
  %cst_2 = arith.constant dense<1.00> : tensor<1xf32>
  // expected-error @+1 {{expect operands to be compatible with body block return types but got 'tensor<1xi32>', 'tensor<2xi32>', 'tensor<1xf32>', 'tensor<3xf32>' vs 'tensor<1xi32>', 'tensor<2xi32>', 'tensor<1xf32>'}}
  %1:4 = "mhlo.while"(%cst_0, %cst_1, %cst_2, %arg0) ({
  ^bb0(%arg1: tensor<1xi32>, %arg2: tensor<2xi32>, %arg3: tensor<1xf32>, %arg4: tensor<3xf32>):
    %2 = arith.constant dense<0> : tensor<i32>
    %3 = "mhlo.slice"(%arg2) <{limit_indices = dense<[1]> : tensor<1xi64>, start_indices = dense<[0]> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>}> : (tensor<2xi32>) -> tensor<1xi32>
    %4 = "mhlo.compare"(%arg1, %3) {comparison_direction = #mhlo<comparison_direction LT>} : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    %5 = "mhlo.reshape"(%4) : (tensor<1xi1>) -> tensor<i1>
    "mhlo.return"(%5) : (tensor<i1>) -> ()
  },  {
  ^bb0(%arg1: tensor<1xi32>, %arg2: tensor<2xi32>, %arg3: tensor<1xf32>, %arg4: tensor<3xf32>):
    %3 = "mhlo.broadcast_in_dim"(%arg3) <{broadcast_dimensions = dense<0> : tensor<1xi64>}> : (tensor<1xf32>) -> tensor<3xf32>
    %4 = mhlo.add %3, %arg4 : tensor<3xf32>
    "mhlo.return"(%arg1, %arg2, %arg3) : (tensor<1xi32>, tensor<2xi32>, tensor<1xf32>) -> ()
  }) : (tensor<1xi32>, tensor<2xi32>, tensor<1xf32>, tensor<3xf32>) -> (tensor<1xi32>, tensor<2xi32>, tensor<1xf32>, tensor<3xf32>)
  func.return %1#3: tensor<3xf32>
}
