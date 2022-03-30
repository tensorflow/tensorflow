// RUN: mlir-hlo-opt %s -verify-diagnostics -split-input-file

// -----

func.func @while_with_different_types(%arg0: tensor<3xf32>) -> tensor<3xf32> {
  %cst_0 = arith.constant dense<0> : tensor<1xi32>
  %cst_1 = arith.constant dense<[100, 100]> : tensor<2xi32>
  %cst_2 = arith.constant dense<1.00> : tensor<1xf32>
  // expected-error @+1 {{'mhlo.while' op type mismatch between operand #2 and the matching condition block argument: 'tensor<1xf32>' vs 'tensor<3xf32>'}}
  %1:4 = "mhlo.while"(%cst_0, %cst_1, %cst_2, %arg0) ({
  ^bb0(%arg1: tensor<1xi32>, %arg2: tensor<2xi32>, %arg3: tensor<3xf32>, %arg4: tensor<3xf32>):
    %2 = arith.constant dense<0> : tensor<i32>
    %3 = "mhlo.slice"(%arg2) {limit_indices = dense<[1]> : tensor<1xi64>, start_indices = dense<[0]> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<2xi32>) -> tensor<1xi32>
    %4 = "mhlo.compare"(%arg1, %3) {comparison_direction = "LT"} : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    %5 = "mhlo.reshape"(%4) : (tensor<1xi1>) -> tensor<i1>
    "mhlo.return"(%5) : (tensor<i1>) -> ()
  },  {
  ^bb0(%arg1: tensor<1xi32>, %arg2: tensor<2xi32>, %arg3: tensor<1xf32>, %arg4: tensor<3xf32>):
    %3 = "mhlo.broadcast_in_dim"(%arg3) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<1xf32>) -> tensor<3xf32>
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
  // expected-error @+1 {{'mhlo.while' op type mismatch between operand #1 and the matching body block argument: 'tensor<2xi32>' vs 'tensor<3xi32>'}}
  %1:4 = "mhlo.while"(%cst_0, %cst_1, %cst_2, %arg0) ({
  ^bb0(%arg1: tensor<1xi32>, %arg2: tensor<2xi32>, %arg3: tensor<1xf32>, %arg4: tensor<3xf32>):
    %2 = arith.constant dense<0> : tensor<i32>
    %3 = "mhlo.slice"(%arg2) {limit_indices = dense<[1]> : tensor<1xi64>, start_indices = dense<[0]> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<2xi32>) -> tensor<1xi32>
    %4 = "mhlo.compare"(%arg1, %3) {comparison_direction = "LT"} : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    %5 = "mhlo.reshape"(%4) : (tensor<1xi1>) -> tensor<i1>
    "mhlo.return"(%5) : (tensor<i1>) -> ()
  },  {
  ^bb0(%arg1: tensor<1xi32>, %arg2: tensor<3xi32>, %arg3: tensor<1xf32>, %arg4: tensor<3xf32>):
    %3 = "mhlo.broadcast_in_dim"(%arg3) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<1xf32>) -> tensor<3xf32>
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
  // expected-error @+1 {{'mhlo.while' op mismatch in operand count (4) vs the condition block argument count (3)}}
  %1:4 = "mhlo.while"(%cst_0, %cst_1, %cst_2, %arg0) ({
  ^bb0(%arg1: tensor<1xi32>, %arg2: tensor<2xi32>, %arg3: tensor<1xf32>):
    %2 = arith.constant dense<0> : tensor<i32>
    %3 = "mhlo.slice"(%arg2) {limit_indices = dense<[1]> : tensor<1xi64>, start_indices = dense<[0]> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<2xi32>) -> tensor<1xi32>
    %4 = "mhlo.compare"(%arg1, %3) {comparison_direction = "LT"} : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    %5 = "mhlo.reshape"(%4) : (tensor<1xi1>) -> tensor<i1>
    "mhlo.return"(%5) : (tensor<i1>) -> ()
  },  {
  ^bb0(%arg1: tensor<1xi32>, %arg2: tensor<3xi32>, %arg3: tensor<1xf32>, %arg4: tensor<3xf32>):
    %3 = "mhlo.broadcast_in_dim"(%arg3) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<1xf32>) -> tensor<3xf32>
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
  // expected-error @+1 {{'mhlo.while' op mismatch in operand count (4) vs the body block argument count (3)}}
  %1:4 = "mhlo.while"(%cst_0, %cst_1, %cst_2, %arg0) ({
  ^bb0(%arg1: tensor<1xi32>, %arg2: tensor<2xi32>, %arg3: tensor<1xf32>, %arg4: tensor<3xf32>):
    %2 = arith.constant dense<0> : tensor<i32>
    %3 = "mhlo.slice"(%arg2) {limit_indices = dense<[1]> : tensor<1xi64>, start_indices = dense<[0]> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<2xi32>) -> tensor<1xi32>
    %4 = "mhlo.compare"(%arg1, %3) {comparison_direction = "LT"} : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    %5 = "mhlo.reshape"(%4) : (tensor<1xi1>) -> tensor<i1>
    "mhlo.return"(%5) : (tensor<i1>) -> ()
  },  {
  ^bb0(%arg1: tensor<1xi32>, %arg2: tensor<3xi32>, %arg3: tensor<1xf32>):
    %3 = "mhlo.broadcast_in_dim"(%arg3) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<1xf32>) -> tensor<3xf32>
    "mhlo.return"(%arg1, %arg2, %arg3, %3) : (tensor<1xi32>, tensor<3xi32>, tensor<1xf32>, tensor<3xf32>) -> ()
  }) : (tensor<1xi32>, tensor<2xi32>, tensor<1xf32>, tensor<3xf32>) -> (tensor<1xi32>, tensor<2xi32>, tensor<1xf32>, tensor<3xf32>)
  func.return %1#3: tensor<3xf32>
}

// -----

func.func @while_with_cond_return_width_mismatch(%arg0: tensor<3xf32>) -> tensor<3xf32> {
  %cst_0 = arith.constant dense<0> : tensor<1xi32>
  %cst_1 = arith.constant dense<[100, 100]> : tensor<2xi32>
  %cst_2 = arith.constant dense<1.00> : tensor<1xf32>
  %1:4 = "mhlo.while"(%cst_0, %cst_1, %cst_2, %arg0) ({
  ^bb0(%arg1: tensor<1xi32>, %arg2: tensor<2xi32>, %arg3: tensor<1xf32>, %arg4: tensor<3xf32>):
    %2 = "mhlo.reshape"(%arg1) : (tensor<1xi32>) -> tensor<i32>
    // expected-error @+1 {{'mhlo.return' op expects a zero-ranked tensor of i1, got 'tensor<i32>'}}
    "mhlo.return"(%2) : (tensor<i32>) -> ()
  },  {
  ^bb0(%arg1: tensor<1xi32>, %arg2: tensor<2xi32>, %arg3: tensor<1xf32>, %arg4: tensor<3xf32>):
    %3 = "mhlo.broadcast_in_dim"(%arg3) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<1xf32>) -> tensor<3xf32>
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
  %1:4 = "mhlo.while"(%cst_0, %cst_1, %cst_2, %arg0) ({
  ^bb0(%arg1: tensor<1xi32>, %arg2: tensor<2xi32>, %arg3: tensor<1xf32>, %arg4: tensor<3xf32>):
    %3 = "mhlo.slice"(%arg2) {limit_indices = dense<[1]> : tensor<1xi64>, start_indices = dense<[0]> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<2xi32>) -> tensor<1xi32>
    %4 = "mhlo.compare"(%arg1, %3) {comparison_direction = "LT"} : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    // expected-error @+1 {{'mhlo.return' op expects a zero-ranked tensor of i1, got 'tensor<1xi1>'}}
    "mhlo.return"(%4) : (tensor<1xi1>) -> ()
  },  {
  ^bb0(%arg1: tensor<1xi32>, %arg2: tensor<2xi32>, %arg3: tensor<1xf32>, %arg4: tensor<3xf32>):
    %3 = "mhlo.broadcast_in_dim"(%arg3) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<1xf32>) -> tensor<3xf32>
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
  %1:4 = "mhlo.while"(%cst_0, %cst_1, %cst_2, %arg0) ({
  ^bb0(%arg1: tensor<1xi32>, %arg2: tensor<2xi32>, %arg3: tensor<1xf32>, %arg4: tensor<3xf32>):
    %2 = "mhlo.reshape"(%arg3) : (tensor<1xf32>) -> tensor<f32>
    // expected-error @+1 {{'mhlo.return' op expects a zero-ranked tensor of i1, got 'tensor<f32>'}}
    "mhlo.return"(%2) : (tensor<f32>) -> ()
  },  {
  ^bb0(%arg1: tensor<1xi32>, %arg2: tensor<2xi32>, %arg3: tensor<1xf32>, %arg4: tensor<3xf32>):
    %3 = "mhlo.broadcast_in_dim"(%arg3) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<1xf32>) -> tensor<3xf32>
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
  %1:4 = "mhlo.while"(%cst_0, %cst_1, %cst_2, %arg0) ({
  ^bb0(%arg1: tensor<1xi32>, %arg2: tensor<2xi32>, %arg3: tensor<1xf32>, %arg4: tensor<3xf32>):
    %2 = arith.constant dense<0> : tensor<i32>
    %3 = "mhlo.slice"(%arg2) {limit_indices = dense<[1]> : tensor<1xi64>, start_indices = dense<[0]> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<2xi32>) -> tensor<1xi32>
    %4 = "mhlo.compare"(%arg1, %3) {comparison_direction = "LT"} : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    %5 = "mhlo.reshape"(%4) : (tensor<1xi1>) -> tensor<i1>
    "mhlo.return"(%5) : (tensor<i1>) -> ()
  },  {
  ^bb0(%arg1: tensor<1xi32>, %arg2: tensor<2xi32>, %arg3: tensor<1xf32>, %arg4: tensor<3xf32>):
    // expected-error @+1 {{'mhlo.return' op type mismatch between operand #3 and the enclosing WhileOp returned value: 'tensor<1xf32>' vs 'tensor<3xf32>'}}
    "mhlo.return"(%arg1, %arg2, %arg3, %arg3) : (tensor<1xi32>, tensor<2xi32>, tensor<1xf32>, tensor<1xf32>) -> ()
  }) : (tensor<1xi32>, tensor<2xi32>, tensor<1xf32>, tensor<3xf32>) -> (tensor<1xi32>, tensor<2xi32>, tensor<1xf32>, tensor<3xf32>)
  func.return %1#3: tensor<3xf32>
}

// -----

func.func @while_with_multiple_operand_in_cond_return(%arg0: tensor<3xf32>) -> tensor<3xf32> {
  %cst_0 = arith.constant dense<0> : tensor<1xi32>
  %cst_1 = arith.constant dense<[100, 100]> : tensor<2xi32>
  %cst_2 = arith.constant dense<1.00> : tensor<1xf32>
  %1:4 = "mhlo.while"(%cst_0, %cst_1, %cst_2, %arg0) ({
  ^bb0(%arg1: tensor<1xi32>, %arg2: tensor<2xi32>, %arg3: tensor<1xf32>, %arg4: tensor<3xf32>):
    %2 = arith.constant dense<0> : tensor<i32>
    %3 = "mhlo.slice"(%arg2) {limit_indices = dense<[1]> : tensor<1xi64>, start_indices = dense<[0]> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<2xi32>) -> tensor<1xi32>
    %4 = "mhlo.compare"(%arg1, %3) {comparison_direction = "LT"} : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    %5 = "mhlo.reshape"(%4) : (tensor<1xi1>) -> tensor<i1>
  // expected-error @+1 {{'mhlo.return' op expects a single operand for while condition body return, got 2}}
    "mhlo.return"(%5, %5) : (tensor<i1>, tensor<i1>) -> ()
  },  {
  ^bb0(%arg1: tensor<1xi32>, %arg2: tensor<2xi32>, %arg3: tensor<1xf32>, %arg4: tensor<3xf32>):
    %3 = "mhlo.broadcast_in_dim"(%arg3) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<1xf32>) -> tensor<3xf32>
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
  %1:4 = "mhlo.while"(%cst_0, %cst_1, %cst_2, %arg0) ({
  ^bb0(%arg1: tensor<1xi32>, %arg2: tensor<2xi32>, %arg3: tensor<1xf32>, %arg4: tensor<3xf32>):
    %2 = arith.constant dense<0> : tensor<i32>
    %3 = "mhlo.slice"(%arg2) {limit_indices = dense<[1]> : tensor<1xi64>, start_indices = dense<[0]> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<2xi32>) -> tensor<1xi32>
    %4 = "mhlo.compare"(%arg1, %3) {comparison_direction = "LT"} : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
    %5 = "mhlo.reshape"(%4) : (tensor<1xi1>) -> tensor<i1>
    "mhlo.return"(%5) : (tensor<i1>) -> ()
  },  {
  ^bb0(%arg1: tensor<1xi32>, %arg2: tensor<2xi32>, %arg3: tensor<1xf32>, %arg4: tensor<3xf32>):
    %3 = "mhlo.broadcast_in_dim"(%arg3) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<1xf32>) -> tensor<3xf32>
    %4 = mhlo.add %3, %arg4 : tensor<3xf32>
  // expected-error @+1 {{'mhlo.return' op expects body to return a many value as the operands (4), got 3}}
    "mhlo.return"(%arg1, %arg2, %arg3) : (tensor<1xi32>, tensor<2xi32>, tensor<1xf32>) -> ()
  }) : (tensor<1xi32>, tensor<2xi32>, tensor<1xf32>, tensor<3xf32>) -> (tensor<1xi32>, tensor<2xi32>, tensor<1xf32>, tensor<3xf32>)
  func.return %1#3: tensor<3xf32>
}
