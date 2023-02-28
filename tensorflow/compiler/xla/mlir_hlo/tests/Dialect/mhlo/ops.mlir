// RUN: mlir-hlo-opt %s -verify-diagnostics -split-input-file -allow-unregistered-dialect | FileCheck %s

// Tests for types, ops with custom constraints, verifiers, printer or parser
// methods.

// CHECK-LABEL: func private @token_type() -> !mhlo.token
func.func private @token_type() -> !mhlo.token

// -----

// expected-error@+1 {{unknown mhlo type: foobar}}
func.func private @invalid_type() -> !mhlo.foobar

// -----

// CHECK-LABEL: func @reduce_scatter
func.func @reduce_scatter(%data: tensor<4x16xf32>) -> tensor<4x4xf32> {
  %0 = "mhlo.reduce_scatter"(%data) ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
    %1 = mhlo.add %arg2, %arg3 : tensor<f32>
    "mhlo.return"(%1) : (tensor<f32>) -> ()
  }) {replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>,
      scatter_dimension = 1 : i64,
      use_global_device_ids} : (tensor<4x16xf32>) -> tensor<4x4xf32>
  func.return %0 : tensor<4x4xf32>
}

// -----

func.func @invalid_reduce_scatter(%data: tensor<4x16xf32>) -> tensor<4x5xf32> {
  // expected-error@+1 {{operand scatter dimension has size 16, expected to be a multiple of result scatter dimension size 5}}
  %0 = "mhlo.reduce_scatter"(%data) ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
    %1 = mhlo.add %arg2, %arg3 : tensor<f32>
    "mhlo.return"(%1) : (tensor<f32>) -> ()
  }) {replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>,
      scatter_dimension = 1 : i64} : (tensor<4x16xf32>) -> tensor<4x5xf32>
  func.return %0 : tensor<4x5xf32>
}

// -----

func.func @main(%arg0: tensor<10xf32>) -> tensor<10xf32> {
  %0 = "mhlo.all_reduce"(%arg0) ({
  // Perform max reduction inside the region
  ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):
    %max = mhlo.maximum %lhs, %rhs : tensor<f32>
    "mhlo.return"(%max) : (tensor<f32>) -> ()
  })
  {
    replica_groups = dense<[[0, 2, 4, 6], [1, 3, 5, 7]]> : tensor<2x4xi64>,
    channel_handle = #mhlo.channel_handle<
      handle = 5,
      type = 2
    >,
    use_global_device_ids
  } : (tensor<10xf32>) -> tensor<10xf32>
  func.return %0 : tensor<10xf32>
}

// -----

func.func @all_reduce_invalid_reducer(%operand: tensor<10xf32>) -> tensor<10xf32> {
  // expected-error@+1 {{Reduction-region must take 2 parameters, but takes 3 parameter(s)}}
  %0 = "mhlo.all_reduce"(%operand) ({
  ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<f32>):
    %max = mhlo.maximum %arg0, %arg1 : tensor<f32>
    "mhlo.return"(%max) : (tensor<f32>) -> ()
  })
  {
    replica_groups = dense<[[0, 2, 4, -1], [1, 3, -1, -1]]> : tensor<2x4xi64>
  } : (tensor<10xf32>) -> tensor<10xf32>
  func.return %0 : tensor<10xf32>
}

// -----

func.func @all_reduce_invalid_reducer(%operand: tensor<10xf32>) -> tensor<10xf32> {
  // expected-error@+1 {{The reduction-region expected to return some value(s)}}
  %0 = "mhlo.all_reduce"(%operand) ({
  ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
    %max = mhlo.maximum %arg0, %arg1 : tensor<f32>
    "mhlo.return"() : () -> ()
  })
  {
    replica_groups = dense<[[0, 2, 4, 6], [1, 3, 5, 7]]> : tensor<2x4xi64>
  } : (tensor<10xf32>) -> tensor<10xf32>
  func.return %0 : tensor<10xf32>
}

// -----

func.func @all_reduce_invalid_reducer(%operand: tensor<10xf32>) -> tensor<10xf32> {
  // expected-error@+1 {{Reduction-region here must produce 1 tensors, but produces 2 instead}}
  %0 = "mhlo.all_reduce"(%operand) ({
  ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
    %max = mhlo.maximum %arg0, %arg1 : tensor<f32>
    "mhlo.return"(%max, %max) : (tensor<f32>, tensor<f32>) -> ()
  })
  {
    replica_groups = dense<[[0, 2, 4, 6], [1, 3, 5, 7]]> : tensor<2x4xi64>
  } : (tensor<10xf32>) -> tensor<10xf32>
  func.return %0 : tensor<10xf32>
}

// -----

func.func @all_reduce_invalid_reducer(%operand: tensor<10xf32>) -> tensor<10xf32> {
  // expected-error@+1 {{Reduction-region here must produce tensor-typed result(s), but produces 'tuple<tensor<f32>, tensor<f32>>' instead}}
  %0 = "mhlo.all_reduce"(%operand) ({
  ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
    %max = mhlo.maximum %arg0, %arg1 : tensor<f32>
    %tup = "mhlo.tuple"(%max, %max) : (tensor<f32>, tensor<f32>) -> tuple<tensor<f32>, tensor<f32>>
    "mhlo.return"(%tup) : (tuple<tensor<f32>, tensor<f32>>) -> ()
  })
  {
    replica_groups = dense<[[0, 2, 4, 6], [1, 3, 5, 7]]> : tensor<2x4xi64>
  } : (tensor<10xf32>) -> tensor<10xf32>
  func.return %0 : tensor<10xf32>
}

// -----

func.func @all_reduce_invalid_reducer(%operand: tensor<10xf32>) -> tensor<10xf32> {
  // expected-error@+1 {{The type of reduction-region's parameter at index 1 is different than the corresponding result type: 'tensor<i32>' vs 'tensor<f32>'}}
  %0 = "mhlo.all_reduce"(%operand) ({
  ^bb0(%arg0: tensor<f32>, %arg1: tensor<i32>):
    %max = mhlo.maximum %arg0, %arg0 : tensor<f32>
    "mhlo.return"(%max) : (tensor<f32>) -> ()
  })
  {
    replica_groups = dense<[[0, 2, 4, 6], [1, 3, 5, 7]]> : tensor<2x4xi64>
  } : (tensor<10xf32>) -> tensor<10xf32>
  func.return %0 : tensor<10xf32>
}

// -----

func.func @all_reduce_invalid_reducer(%operand: tensor<10xf32>) -> tensor<10xf32> {
  // expected-error@+1 {{The type of reduction-region's parameter at index 0 is different than the corresponding result type: 'tensor<f32>' vs 'tensor<i32>'}}
  %0 = "mhlo.all_reduce"(%operand) ({
  ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
    %max = mhlo.maximum %arg0, %arg1 : tensor<f32>
    %maxint = "mhlo.convert"(%max) : (tensor<f32>) -> tensor<i32>
    "mhlo.return"(%maxint) : (tensor<i32>) -> ()
  })
  {
    replica_groups = dense<[[0, 2, 4, 6], [1, 3, 5, 7]]> : tensor<2x4xi64>
  } : (tensor<10xf32>) -> tensor<10xf32>
  func.return %0 : tensor<10xf32>
}

// -----

func.func @all_reduce_invalid_reducer(%operand: tensor<10xf32>) -> tensor<10xf32> {
  // expected-error@+1 {{The type of reduction-region's result type at index 0 differs from the op's corresponding init-value type: 'tensor<i32>' vs 'tensor<f32>'}}
  %0 = "mhlo.all_reduce"(%operand) ({
  ^bb0(%arg0: tensor<i32>, %arg1: tensor<i32>):
    %max = mhlo.maximum %arg0, %arg1 : tensor<i32>
    "mhlo.return"(%max) : (tensor<i32>) -> ()
  })
  {
    replica_groups = dense<[[0, 2, 4, 6], [1, 3, 5, 7]]> : tensor<2x4xi64>
  } : (tensor<10xf32>) -> tensor<10xf32>
  func.return %0 : tensor<10xf32>
}

// -----

func.func @all_reduce_invalid_reducer(%operand: tensor<10xf32>) -> tensor<10xf32> {
  // expected-error@+1 {{The type of reduction-region's result type at index 0 differs from the op's corresponding init-value type: 'tensor<4xf32>' vs 'tensor<f32>'}}
  %0 = "mhlo.all_reduce"(%operand) ({
  ^bb0(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>):
    %max = mhlo.maximum %arg0, %arg1 : tensor<4xf32>
    "mhlo.return"(%max) : (tensor<4xf32>) -> ()
  })
  {
    replica_groups = dense<[[0, 2, 4, 6], [1, 3, 5, 7]]> : tensor<2x4xi64>
  } : (tensor<10xf32>) -> tensor<10xf32>
  func.return %0 : tensor<10xf32>
}

// -----

func.func @all_reduce_invalid_return_type(%operand: tensor<10xf32>) -> tensor<10x4xf32> {
  // expected-error@+1 {{requires compatible types for all operands and results}}
  %0 = "mhlo.all_reduce"(%operand) ({
  ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
    %max = mhlo.maximum %arg0, %arg1 : tensor<f32>
    "mhlo.return"(%max) : (tensor<f32>) -> ()
  })
  {
    replica_groups = dense<[[0, 2, 4, 6], [1, 3, 5, 7]]> : tensor<2x4xi64>
  } : (tensor<10xf32>) -> tensor<10x4xf32>
  func.return %0 : tensor<10x4xf32>
}

// -----

func.func @all_reduce_invalid_return_type(%operand: tensor<10xf32>) -> tensor<10xi32> {
  // expected-error@+1 {{requires compatible types for all operands and results}}
  %0 = "mhlo.all_reduce"(%operand) ({
  ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
    %max = mhlo.maximum %arg0, %arg1 : tensor<f32>
    "mhlo.return"(%max) : (tensor<f32>) -> ()
  })
  {
    replica_groups = dense<[[0, 2, 4, 6], [1, 3, 5, 7]]> : tensor<2x4xi64>
  } : (tensor<10xf32>) -> tensor<10xi32>
  func.return %0 : tensor<10xi32>
}

// -----

func.func @all_reduce_invalid_replica_group(%operand: tensor<10xf32>) -> tensor<10xf32> {
  // expected-error@+1 {{replica groups should be a rank 2 tensor}}
  %0 = "mhlo.all_reduce"(%operand) ({
  ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
    %max = mhlo.maximum %arg0, %arg1 : tensor<f32>
    "mhlo.return"(%max) : (tensor<f32>) -> ()
  })
  {
    replica_groups = dense<0> : tensor<1xi64>
  } : (tensor<10xf32>) -> tensor<10xf32>
  func.return %0 : tensor<10xf32>
}

// -----

func.func @all_reduce_invalid_replica_group(%operand: tensor<10xf32>) -> tensor<10xf32> {
  // expected-error@+1 {{replica id #1 seen more than once}}
  %0 = "mhlo.all_reduce"(%operand) ({
  ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
    %max = mhlo.maximum %arg0, %arg1 : tensor<f32>
    "mhlo.return"(%max) : (tensor<f32>) -> ()
  })
  {
    replica_groups = dense<[[0, 1, 1, 3]]> : tensor<1x4xi64>
  } : (tensor<10xf32>) -> tensor<10xf32>
  func.return %0 : tensor<10xf32>
}

// -----

func.func @all_reduce_invalid_replica_group(%operand: tensor<10xf32>) -> tensor<10xf32> {
  //  expected-error@+1 {{replica id #2 not seen in replica groups}}
  %0 = "mhlo.all_reduce"(%operand) ({
  ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
    %max = mhlo.maximum %arg0, %arg1 : tensor<f32>
    "mhlo.return"(%max) : (tensor<f32>) -> ()
  })
  {
    replica_groups = dense<[[0, 1, 3]]> : tensor<1x3xi64>
  } : (tensor<10xf32>) -> tensor<10xf32>
  func.return %0 : tensor<10xf32>
}

// -----

func.func @all_reduce_invalid_replica_group(%operand: tensor<10xf32>) -> tensor<10xf32> {
  //  expected-error@+1 {{replica groups cannot be empty}}
  %0 = "mhlo.all_reduce"(%operand) ({
  ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
    %max = mhlo.maximum %arg0, %arg1 : tensor<f32>
    "mhlo.return"(%max) : (tensor<f32>) -> ()
  })
  {
    replica_groups = dense<0> : tensor<0x2xi64>,
    use_global_device_ids
  } : (tensor<10xf32>) -> tensor<10xf32>
  func.return %0 : tensor<10xf32>
}

// -----

func.func @invalid_reduce_scatter(%data: tensor<4x0xf32>) -> tensor<4x4xf32> {
  // expected-error@+1 {{operand scatter dimension cannot be zero}}
  %0 = "mhlo.reduce_scatter"(%data) ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
    %1 = mhlo.add %arg2, %arg3 : tensor<f32>
    "mhlo.return"(%1) : (tensor<f32>) -> ()
  }) {replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>,
      scatter_dimension = 1 : i64} : (tensor<4x0xf32>) -> tensor<4x4xf32>
  func.return %0 : tensor<4x4xf32>
}

// -----

func.func @invalid_reduce_scatter(%data: tensor<4x16xf32>) -> tensor<4x0xf32> {
  // expected-error@+1 {{result scatter dimension cannot be zero}}
  %0 = "mhlo.reduce_scatter"(%data) ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
    %1 = mhlo.add %arg2, %arg3 : tensor<f32>
    "mhlo.return"(%1) : (tensor<f32>) -> ()
  }) {replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>,
      scatter_dimension = 1 : i64} : (tensor<4x16xf32>) -> tensor<4x0xf32>
  func.return %0 : tensor<4x0xf32>
}

// -----

func.func @invalid_reduce_scatter(%data: tensor<4x16xf32>) -> tensor<4xf32> {
  // expected-error@+1 {{operand and result should have same rank}}
  %0 = "mhlo.reduce_scatter"(%data) ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
    %1 = mhlo.add %arg2, %arg3 : tensor<f32>
    "mhlo.return"(%1) : (tensor<f32>) -> ()
  }) {replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>,
      scatter_dimension = 1 : i64} : (tensor<4x16xf32>) -> tensor<4xf32>
  func.return %0 : tensor<4xf32>
}

// -----

func.func @invalid_reduce_scatter(%data: tensor<4x16xf32>) -> tensor<4x4xf32> {
  // expected-error@+1 {{scatter dim should be less than operand/result rank}}
  %0 = "mhlo.reduce_scatter"(%data) ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
    %1 = mhlo.add %arg2, %arg3 : tensor<f32>
    "mhlo.return"(%1) : (tensor<f32>) -> ()
  }) {replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>,
      scatter_dimension = 4 : i64} : (tensor<4x16xf32>) -> tensor<4x4xf32>
  func.return %0 : tensor<4x4xf32>
}

// -----

func.func @invalid_reduce_scatter(%data: tensor<4x16xf32>) -> tensor<3x4xf32> {
  // expected-error@+1 {{non scatter dimensions should be same for operand (4) and result (3)}}
  %0 = "mhlo.reduce_scatter"(%data) ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
    %1 = mhlo.add %arg2, %arg3 : tensor<f32>
    "mhlo.return"(%1) : (tensor<f32>) -> ()
  }) {replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>,
      scatter_dimension = 1 : i64} : (tensor<4x16xf32>) -> tensor<3x4xf32>
  func.return %0 : tensor<3x4xf32>
}

// -----

func.func @reduce_scatter(%data: tensor<4x16xf32>) -> tensor<4x4xf32> {
  // expected-error@+1 {{replica groups should be a rank 2 tensor}}
  %0 = "mhlo.reduce_scatter"(%data) ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
    %1 = mhlo.add %arg2, %arg3 : tensor<f32>
    "mhlo.return"(%1) : (tensor<f32>) -> ()
  }) {replica_groups = dense<0> : tensor<1xi64>,
      scatter_dimension = 1 : i64} : (tensor<4x16xf32>) -> tensor<4x4xf32>
  func.return %0 : tensor<4x4xf32>
}

// -----

func.func @reduce_scatter(%data: tensor<4x16xf32>) -> tensor<4x4xf32> {
  //  expected-error@+1 {{Invalid replica id -1}}
  %0 = "mhlo.reduce_scatter"(%data) ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
    %1 = mhlo.add %arg2, %arg3 : tensor<f32>
    "mhlo.return"(%1) : (tensor<f32>) -> ()
  }) {replica_groups = dense<[[0, -1, 2, 3]]> : tensor<1x4xi64>,
      scatter_dimension = 1 : i64} : (tensor<4x16xf32>) -> tensor<4x4xf32>
  func.return %0 : tensor<4x4xf32>
}

// -----

func.func @reduce_scatter(%data: tensor<4x16xf32>) -> tensor<4x4xf32> {
  //  expected-error@+1 {{replica id #1 seen more than once}}
  %0 = "mhlo.reduce_scatter"(%data) ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
    %1 = mhlo.add %arg2, %arg3 : tensor<f32>
    "mhlo.return"(%1) : (tensor<f32>) -> ()
  }) {replica_groups = dense<[[0, 1, 1, 3]]> : tensor<1x4xi64>,
      scatter_dimension = 1 : i64} : (tensor<4x16xf32>) -> tensor<4x4xf32>
  func.return %0 : tensor<4x4xf32>
}

// -----

func.func @reduce_scatter(%data: tensor<4x16xf32>) -> tensor<4x4xf32> {
  //  expected-error@+1 {{replica id #2 not seen in replica groups}}
  %0 = "mhlo.reduce_scatter"(%data) ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
    %1 = mhlo.add %arg2, %arg3 : tensor<f32>
    "mhlo.return"(%1) : (tensor<f32>) -> ()
  }) {replica_groups = dense<[[0, 1, 3]]> : tensor<1x3xi64>,
      scatter_dimension = 1 : i64} : (tensor<4x16xf32>) -> tensor<4x4xf32>
  func.return %0 : tensor<4x4xf32>
}

// -----

func.func @invalid_reduce_scatter(%data: tensor<4x16xf32>) -> tensor<4x4xf32> {
  // expected-error@+1 {{The reduction-region expected to return some value(s)}}
  %0 = "mhlo.reduce_scatter"(%data) ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
    %1 = mhlo.add %arg2, %arg3 : tensor<f32>
    "mhlo.return"() : () -> ()
  }) {replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>,
      scatter_dimension = 1 : i64} : (tensor<4x16xf32>) -> tensor<4x4xf32>
  func.return %0 : tensor<4x4xf32>
}

// -----

func.func @reduce_scatter(%data: tensor<4x16xf32>) -> tensor<4x4xf32> {
  // expected-error@+1 {{expects scatter_dimension >= 0}}
  %0 = "mhlo.reduce_scatter"(%data) ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
    %1 = mhlo.add %arg2, %arg3 : tensor<f32>
    "mhlo.return"(%1) : (tensor<f32>) -> ()
  }) {replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>,
      scatter_dimension = -1 : i64} : (tensor<4x16xf32>) -> tensor<4x4xf32>
  func.return %0 : tensor<4x4xf32>
}

// -----

// CHECK-LABEL: func @reduce_scatter_dynamic
func.func @reduce_scatter_dynamic(%data: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = "mhlo.reduce_scatter"(%data) ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
    %1 = mhlo.add %arg2, %arg3 : tensor<f32>
    "mhlo.return"(%1) : (tensor<f32>) -> ()
  }) {replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>,
      scatter_dimension = 1 : i64,
      use_global_device_ids} : (tensor<?x?xf32>) -> tensor<?x?xf32>
  func.return %0 : tensor<?x?xf32>
}

// -----

// CHECK-LABEL: func @alltoall
func.func @alltoall(%data: tensor<4x16xf32>) -> tensor<16x4xf32> {
  %0 = "mhlo.all_to_all"(%data) {
    split_dimension = 1 : i64,
    concat_dimension = 0 : i64,
    split_count = 4 : i64,
    replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>
  } : (tensor<4x16xf32>) -> tensor<16x4xf32>
  func.return %0 : tensor<16x4xf32>
}

// -----

// CHECK-LABEL: func @alltoall_unranked_input
func.func @alltoall_unranked_input(%data: tensor<*xf32>) -> tensor<*xf32> {
  %0 = "mhlo.all_to_all"(%data) {
    split_dimension = 1 : i64,
    concat_dimension = 0 : i64,
    split_count = 5 : i64,
    replica_groups = dense<[[0, 1, 2, 3, 4]]> : tensor<1x5xi64>
  } : (tensor<*xf32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// -----

// CHECK-LABEL: func @alltoall_dynamic_split_dim
func.func @alltoall_dynamic_split_dim(%data: tensor<4x?xf32>) -> tensor<20x?xf32> {
  %0 = "mhlo.all_to_all"(%data) {
    split_dimension = 1 : i64,
    concat_dimension = 0 : i64,
    split_count = 5 : i64,
    replica_groups = dense<[[0, 1, 2, 3, 4]]> : tensor<1x5xi64>
  } : (tensor<4x?xf32>) -> tensor<20x?xf32>
  func.return %0 : tensor<20x?xf32>
}

// -----

// CHECK-LABEL: func @alltoall_dynamic_concat_dim
func.func @alltoall_dynamic_concat_dim(%data: tensor<?x16xf32>) -> tensor<?x4xf32> {
  %0 = "mhlo.all_to_all"(%data) {
    split_dimension = 1 : i64,
    concat_dimension = 0 : i64,
    split_count = 4 : i64,
    replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>
  } : (tensor<?x16xf32>) -> tensor<?x4xf32>
  func.return %0 : tensor<?x4xf32>
}

// -----

// CHECK-LABEL: func @alltoall_dynamic_split_dim
func.func @alltoall_dynamic_split_dim(%data: tensor<4x?xf32>) -> tensor<20x?xf32> {
  %0 = "mhlo.all_to_all"(%data) {
    split_dimension = 1 : i64,
    concat_dimension = 0 : i64,
    split_count = 5 : i64,
    replica_groups = dense<[[0, 1, 2, 3, 4]]> : tensor<1x5xi64>
  } : (tensor<4x?xf32>) -> tensor<20x?xf32>
  func.return %0 : tensor<20x?xf32>
}

// -----

// CHECK-LABEL: func @alltoall_dynamic_concat_dim
func.func @alltoall_dynamic_concat_dim(%data: tensor<?x16xf32>) -> tensor<?x4xf32> {
  %0 = "mhlo.all_to_all"(%data) {
    split_dimension = 1 : i64,
    concat_dimension = 0 : i64,
    split_count = 4 : i64,
    replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>
  } : (tensor<?x16xf32>) -> tensor<?x4xf32>
  func.return %0 : tensor<?x4xf32>
}

// -----

func.func @alltoall_negative_split_dimension(%data: tensor<4x16xf32>) -> tensor<16x4xf32> {
  // expected-error@+1 {{AllToAll split_dimension cannot be negative}}
  %0 = "mhlo.all_to_all"(%data) {
    split_dimension = -1 : i64,
    concat_dimension = 0 : i64,
    split_count = 4 : i64,
    replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>
  } : (tensor<4x16xf32>) -> tensor<16x4xf32>
  func.return %0 : tensor<16x4xf32>
}

// -----

func.func @alltoall_out_bound_split_dimension(%data: tensor<4x16xf32>) -> tensor<16x4xf32> {
  // expected-error@+1 {{AllToAll split_dimension 2 is out-of-bounds for input rank 2}}
  %0 = "mhlo.all_to_all"(%data) {
    split_dimension = 2 : i64,
    concat_dimension = 0 : i64,
    split_count = 4 : i64,
    replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>
  } : (tensor<4x16xf32>) -> tensor<16x4xf32>
  func.return %0 : tensor<16x4xf32>
}

// -----

func.func @alltoall_negative_concat_dimension(%data: tensor<4x16xf32>) -> tensor<16x4xf32> {
  // expected-error@+1 {{AllToAll concat_dimension cannot be negative}}
  %0 = "mhlo.all_to_all"(%data) {
    split_dimension = 1 : i64,
    concat_dimension = -1 : i64,
    split_count = 4 : i64,
    replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>
  } : (tensor<4x16xf32>) -> tensor<16x4xf32>
  func.return %0 : tensor<16x4xf32>
}

// -----

func.func @alltoall_out_bound_concat_dimension(%data: tensor<4x16xf32>) -> tensor<16x4xf32> {
  // expected-error@+1 {{AllToAll concat_dimension 2 is out-of-bounds for input rank 2}}
  %0 = "mhlo.all_to_all"(%data) {
    split_dimension = 1 : i64,
    concat_dimension = 2 : i64,
    split_count = 4 : i64,
    replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>
  } : (tensor<4x16xf32>) -> tensor<16x4xf32>
  func.return %0 : tensor<16x4xf32>
}

// -----

func.func @alltoall_invalid_split_count(%data: tensor<4x16xf32>) -> tensor<16x4xf32> {
  // expected-error@+1 {{AllToAll split_count must be > 0}}
  %0 = "mhlo.all_to_all"(%data) {
    split_dimension = 1 : i64,
    concat_dimension = 0 : i64,
    split_count = 0 : i64,
    replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>
  } : (tensor<4x16xf32>) -> tensor<16x4xf32>
  func.return %0 : tensor<16x4xf32>
}

// -----

func.func @alltoall_invalid_split_dim_size(%data: tensor<4x16xf32>) -> tensor<16x4xf32> {
// expected-error@+1 {{split dimension has size 16, expected to be a multiple of split_count 5}}
  %0 = "mhlo.all_to_all"(%data) {
    split_dimension = 1 : i64,
    concat_dimension = 0 : i64,
    split_count = 5 : i64,
    replica_groups = dense<[[0, 1, 2, 3, 4]]> : tensor<1x5xi64>
  } : (tensor<4x16xf32>) -> tensor<16x4xf32>
  func.return %0 : tensor<16x4xf32>
}

// -----

func.func @alltoall_invalid_replica_group(%data: tensor<4x16xf32>) -> tensor<16x4xf32> {
  // expected-error@+1 {{replica groups should be a rank 2 tensor}}
  %0 = "mhlo.all_to_all"(%data) {
    split_dimension = 1 : i64,
    concat_dimension = 0 : i64,
    split_count = 4 : i64,
    replica_groups = dense<[[[0], [1], [2], [3]]]> : tensor<1x4x1xi64>
  } : (tensor<4x16xf32>) -> tensor<16x4xf32>
  func.return %0 : tensor<16x4xf32>
}

// -----

func.func @alltoall_invalid_replica_group(%data: tensor<4x16xf32>) -> tensor<16x4xf32> {
  // expected-error@+1 {{replica id #1 not seen in replica groups}}
  %0 = "mhlo.all_to_all"(%data) {
    split_dimension = 1 : i64,
    concat_dimension = 0 : i64,
    split_count = 4 : i64,
    replica_groups = dense<[[-5, -4, -3, 0]]> : tensor<1x4xi64>
  } : (tensor<4x16xf32>) -> tensor<16x4xf32>
  func.return %0 : tensor<16x4xf32>
}

// -----

func.func @alltoall_invalid_replica_group(%data: tensor<4x16xf32>) -> tensor<16x4xf32> {
  // expected-error@+1 {{replica id #2 seen more than once}}
  %0 = "mhlo.all_to_all"(%data) {
    split_dimension = 1 : i64,
    concat_dimension = 0 : i64,
    split_count = 4 : i64,
    replica_groups = dense<[[0, 2, 4, 6], [1, 3, 5, 2]]> : tensor<2x4xi64>
  } : (tensor<4x16xf32>) -> tensor<16x4xf32>
  func.return %0 : tensor<16x4xf32>
}

// -----

func.func @alltoall_invalid_replica_group(%data: tensor<4x16xf32>) -> tensor<16x4xf32> {
  // expected-error@+1 {{replica id #4 not seen in replica groups}}
  %0 = "mhlo.all_to_all"(%data) {
    split_dimension = 1 : i64,
    concat_dimension = 0 : i64,
    split_count = 4 : i64,
    replica_groups = dense<[[0, 2, 6, 8], [1, 3, 5, 7]]> : tensor<2x4xi64>
  } : (tensor<4x16xf32>) -> tensor<16x4xf32>
  func.return %0 : tensor<16x4xf32>
}

// -----

func.func @alltoall_invalid_replica_group(%data: tensor<4x16xf32>) -> tensor<16x4xf32> {
  // expected-error@+1 {{group size of replica_groups must be 4}}
  %0 = "mhlo.all_to_all"(%data) {
    split_dimension = 1 : i64,
    concat_dimension = 0 : i64,
    split_count = 4 : i64,
    replica_groups = dense<[[0, 2, 4], [1, 3, 5]]> : tensor<2x3xi64>
  } : (tensor<4x16xf32>) -> tensor<16x4xf32>
  func.return %0 : tensor<16x4xf32>
}

// -----

func.func @allgather_incompatible_types(%arg0: tensor<128x32xf32>) -> tensor<128x100xf32> {
  // expected-error@+1 {{result gather dimension has size 100, expected to be a multiple of operand gather dimension size 32}}
  %0 = "mhlo.all_gather"(%arg0) {
    all_gather_dim = 1 : i64,
    channel_handle = #mhlo.channel_handle<handle = 1, type = 0>,
    replica_groups = dense<[[0, 2, 4, 6], [1, 3, 5, 7]]> : tensor<2x4xi64>
  } : (tensor<128x32xf32>) -> tensor<128x100xf32>
  func.return %0 : tensor<128x100xf32>
}

// -----

func.func @allgather_gather_along_zero_dimension(%arg0: tensor<128x0x32xf32>) -> tensor<128x100xf32> {
  // expected-error@+1 {{dimension size of operand at 'all_gather_dim' cannot be zero}}
  %0 = "mhlo.all_gather"(%arg0) {
    all_gather_dim = 1 : i64,
    channel_handle = #mhlo.channel_handle<handle = 1, type = 0>,
    replica_groups = dense<[[0, 2, 4, 6], [1, 3, 5, 7]]> : tensor<2x4xi64>
  } : (tensor<128x0x32xf32>) -> tensor<128x100xf32>
  func.return %0 : tensor<128x100xf32>
}

// -----

// CHECK-LABEL: func @allgather_dynamic_gather_dim
func.func @allgather_dynamic_gather_dim(%arg0: tensor<128x32xf32>) -> tensor<128x?xf32> {
  %0 = "mhlo.all_gather"(%arg0) {
    all_gather_dim = 1 : i64,
    channel_handle = #mhlo.channel_handle<handle = 1, type = 0>,
    replica_groups = dense<[[0, 2, 4, 6], [1, 3, 5, 7]]> : tensor<2x4xi64>,
    use_global_device_ids
  } : (tensor<128x32xf32>) -> tensor<128x?xf32>
  func.return %0 : tensor<128x?xf32>
}

// -----

// CHECK-LABEL: func @allgather_dynamic_non_gather_dim
func.func @allgather_dynamic_non_gather_dim(%arg0: tensor<128x32xf32>) -> tensor<?x64xf32> {
  %0 = "mhlo.all_gather"(%arg0) {
    all_gather_dim = 1 : i64,
    channel_handle = #mhlo.channel_handle<handle = 1, type = 0>,
    replica_groups = dense<[[0, 2, 4, 6], [1, 3, 5, 7]]> : tensor<2x4xi64>,
    use_global_device_ids
  } : (tensor<128x32xf32>) -> tensor<?x64xf32>
  func.return %0 : tensor<?x64xf32>
}

// -----

func.func @all_gather_invalid_dim(%arg0: tensor<8x2xf32>) -> tensor<8x8xf32> {
  // expected-error@+1 {{all_gather_dim must be a valid index of operand}}
  %0 = "mhlo.all_gather"(%arg0) {
    all_gather_dim = 2 : i64,
    channel_handle = #mhlo.channel_handle<handle = 1, type = 0>,
    replica_groups = dense<[[0, 2, 4, 6], [1, 3, 5, 7]]> : tensor<2x4xi64>
  } : (tensor<8x2xf32>) -> tensor<8x8xf32>
  func.return %0 : tensor<8x8xf32>
}

// -----

func.func @all_gather_invalid_dim(%arg0: tensor<8x2xf32>) -> tensor<8x8xf32> {
  // expected-error@+1 {{all_gather_dim cannot be negative}}
  %0 = "mhlo.all_gather"(%arg0) {
    all_gather_dim = -1 : i64,
    channel_handle = #mhlo.channel_handle<handle = 1, type = 0>,
    replica_groups = dense<[[0, 2, 4, 6], [1, 3, 5, 7]]> : tensor<2x4xi64>
  } : (tensor<8x2xf32>) -> tensor<8x8xf32>
  func.return %0 : tensor<8x8xf32>
}

// -----

func.func @all_gather_invalid_result_shape(%arg0: tensor<8x2x32xf32>) -> tensor<8x8xf32> {
  // expected-error@+1 {{operand and return must have the same rank}}
  %0 = "mhlo.all_gather"(%arg0) {
    all_gather_dim = 1 : i64,
    channel_handle = #mhlo.channel_handle<handle = 1, type = 0>,
    replica_groups = dense<[[0, 2, 4, 6], [1, 3, 5, 7]]> : tensor<2x4xi64>
  } : (tensor<8x2x32xf32>) -> tensor<8x8xf32>
  func.return %0 : tensor<8x8xf32>
}

// -----

func.func @all_gather_invalid_result_shape(%arg0: tensor<8x2xf32>) -> tensor<4x8xf32> {
  // expected-error@+1 {{operand and result should have the same shape except for the dimension size at 'all_gather_dim'}}
  %0 = "mhlo.all_gather"(%arg0) {
    all_gather_dim = 1 : i64,
    channel_handle = #mhlo.channel_handle<handle = 1, type = 0>,
    replica_groups = dense<[[0, 2, 4, 6], [1, 3, 5, 7]]> : tensor<2x4xi64>
  } : (tensor<8x2xf32>) -> tensor<4x8xf32>
  func.return %0 : tensor<4x8xf32>
}

// -----

func.func @all_gather_invalid_replica_group_shape(%arg0: tensor<8x2xf32>) -> tensor<8x8xf32> {
  // expected-error@+1 {{replica groups should be a rank 2 tensor}}
  %0 = "mhlo.all_gather"(%arg0) {
    all_gather_dim = 1 : i64,
    channel_handle = #mhlo.channel_handle<handle = 1, type = 0>,
    replica_groups = dense<[[[0], [1], [2], [3]]]> : tensor<1x4x1xi64>
  } : (tensor<8x2xf32>) -> tensor<8x8xf32>
  func.return %0 : tensor<8x8xf32>
}

// -----

func.func @all_gather_invalid_replica_group_shape(%arg0: tensor<8x2xf32>) -> tensor<8x8xf32> {
  // expected-error@+1 {{replica groups cannot be empty}}
  %0 = "mhlo.all_gather"(%arg0) {
    all_gather_dim = 1 : i64,
    channel_handle = #mhlo.channel_handle<handle = 1, type = 0>,
    replica_groups = dense<0> : tensor<0x2xi64>,
    use_global_device_ids
  } : (tensor<8x2xf32>) -> tensor<8x8xf32>
  func.return %0 : tensor<8x8xf32>
}

// -----

func.func @all_gather_invalid_replica_group(%arg0: tensor<8x2xf32>) -> tensor<8x8xf32> {
  // expected-error@+1 {{replica id #1 not seen in replica groups}}
  %0 = "mhlo.all_gather"(%arg0) {
    all_gather_dim = 1 : i64,
    channel_handle = #mhlo.channel_handle<handle = 1, type = 0>,
    replica_groups = dense<[[-5, -4, -3, 0]]> : tensor<1x4xi64>
  } : (tensor<8x2xf32>) -> tensor<8x8xf32>
  func.return %0 : tensor<8x8xf32>
}

// -----

func.func @all_gather_invalid_replica_group(%arg0: tensor<8x2xf32>) -> tensor<8x8xf32> {
  // expected-error@+1 {{replica id #2 seen more than once}}
  %0 = "mhlo.all_gather"(%arg0) {
    all_gather_dim = 1 : i64,
    channel_handle = #mhlo.channel_handle<handle = 1, type = 0>,
    replica_groups = dense<[[0, 2, 4, 6], [1, 3, 5, 2]]> : tensor<2x4xi64>
  } : (tensor<8x2xf32>) -> tensor<8x8xf32>
  func.return %0 : tensor<8x8xf32>
}

// -----

func.func @all_gather_invalid_replica_group(%arg0: tensor<8x2xf32>) -> tensor<8x8xf32> {
  // expected-error@+1 {{replica id #4 not seen in replica groups}}
  %0 = "mhlo.all_gather"(%arg0) {
    all_gather_dim = 1 : i64,
    channel_handle = #mhlo.channel_handle<handle = 1, type = 0>,
    replica_groups = dense<[[0, 2, 6, 8], [1, 3, 5, 7]]> : tensor<2x4xi64>
  } : (tensor<8x2xf32>) -> tensor<8x8xf32>
  func.return %0 : tensor<8x8xf32>
}

// -----

// CHECK-LABEL: func @broadcast
func.func @broadcast(%arg0: tensor<3xi32>) -> tensor<1x2x3xi32> {
  %0 = "mhlo.broadcast"(%arg0) {broadcast_sizes = dense<[1, 2]> : tensor<2xi64>} : (tensor<3xi32>) -> tensor<1x2x3xi32>
  func.return %0 : tensor<1x2x3xi32>
}

// -----

func.func @broadcast_bad_sizes_rank(%arg0: tensor<3xi32>) -> tensor<1x2x3xi32> {
  // expected-error@+1 {{broadcast_sizes has rank 2 instead of rank 1}}
  %0 = "mhlo.broadcast"(%arg0) {broadcast_sizes = dense<[[1, 2]]> : tensor<1x2xi64>} : (tensor<3xi32>) -> tensor<1x2x3xi32>
  func.return %0 : tensor<1x2x3xi32>
}

// -----

func.func @broadcast_bad_result_rank(%arg0: tensor<3xi32>) -> tensor<1x2x3xi32> {
  // expected-error@+1 {{'mhlo.broadcast' op inferred type(s) 'tensor<2x3xi32>' are incompatible with return type(s) of operation 'tensor<1x2x3xi32>'}}
  %0 = "mhlo.broadcast"(%arg0) {broadcast_sizes = dense<[2]> : tensor<1xi64>} : (tensor<3xi32>) -> tensor<1x2x3xi32>
  func.return %0 : tensor<1x2x3xi32>
}

// -----

func.func @broadcast_bad_first_part_result_shape(%arg0: tensor<3xi32>) -> tensor<1x2x3xi32> {
  // expected-error@+1 {{'mhlo.broadcast' op inferred type(s) 'tensor<2x3xi32>' are incompatible with return type(s) of operation 'tensor<1x3xi32>'}}
  %0 = "mhlo.broadcast"(%arg0) {broadcast_sizes = dense<[2]> : tensor<1xi64>} : (tensor<3xi32>) -> tensor<1x3xi32>
  func.return %0 : tensor<1x3xi32>
}

// -----

func.func @broadcast_bad_second_part_result_shape(%arg0: tensor<3xi32>) -> tensor<1x2x3xi32> {
  // expected-error@+1 {{'mhlo.broadcast' op inferred type(s) 'tensor<2x3xi32>' are incompatible with return type(s) of operation 'tensor<2x1xi32>'}}
  %0 = "mhlo.broadcast"(%arg0) {broadcast_sizes = dense<[2]> : tensor<1xi64>} : (tensor<3xi32>) -> tensor<2x1xi32>
  func.return %0 : tensor<2x1xi32>
}

// -----

// CHECK-LABEL: func @broadcast_in_dim
func.func @broadcast_in_dim(%arg0: tensor<1x2xi32>) -> tensor<1x2x2xi32> {
  %0 = "mhlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<[1, 2]> : tensor<2xi64>} : (tensor<1x2xi32>) -> tensor<1x2x2xi32>
  func.return %0 : tensor<1x2x2xi32>
}

// -----

// CHECK-LABEL: func @broadcast_in_dim_zero_rank
func.func @broadcast_in_dim_zero_rank(%arg0: tensor<i32>) -> tensor<1x2x3xi32> {
  %0 = "mhlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<[]> : tensor<0xi64>} : (tensor<i32>) -> tensor<1x2x3xi32>
  func.return %0 : tensor<1x2x3xi32>
}

// -----

// CHECK-LABEL: func @dynamic_broadcast_in_dim
func.func @dynamic_broadcast_in_dim(%arg0: tensor<?x?xi32>, %shape: tensor<3xi64>) -> tensor<?x?x?xi32> {
  %0 = "mhlo.dynamic_broadcast_in_dim"(%arg0, %shape) {broadcast_dimensions = dense<[1, 2]> : tensor<2xi64>} : (tensor<?x?xi32>, tensor<3xi64>) -> tensor<?x?x?xi32>
  func.return %0 : tensor<?x?x?xi32>
}

// -----

// CHECK-LABEL: func @dynamic_broadcast_in_dim_unknown_dim
func.func @dynamic_broadcast_in_dim_unknown_dim(%arg0: tensor<32xf32>, %shape: tensor<3xi64>) -> tensor<?x?x?xf32> {
  %0 = "mhlo.dynamic_broadcast_in_dim"(%arg0, %shape) {broadcast_dimensions = dense<[2]> : tensor<1xi64>} : (tensor<32xf32>, tensor<3xi64>) -> tensor<?x?x?xf32>
  func.return %0 : tensor<?x?x?xf32>
}

// -----

// CHECK-LABEL: func @dynamic_broadcast_in_dim_ok_dim
func.func @dynamic_broadcast_in_dim_ok_dim(%arg0: tensor<1xf32>, %shape: tensor<3xi64>) -> tensor<7x8x9xf32> {
  %0 = "mhlo.dynamic_broadcast_in_dim"(%arg0, %shape) {broadcast_dimensions = dense<[2]> : tensor<1xi64>} : (tensor<1xf32>, tensor<3xi64>) -> tensor<7x8x9xf32>
  func.return %0 : tensor<7x8x9xf32>
}

// -----

func.func @dynamic_broadcast_in_dim_shape_mismatch(%arg0: tensor<32xf32>, %shape: tensor<3xi64>) -> tensor<7x8x9xf32> {
  // expected-error@+1 {{size of operand dimension 0 (32) is not compatible with size of result dimension 2 (9)}}
  %0 = "mhlo.dynamic_broadcast_in_dim"(%arg0, %shape) {broadcast_dimensions = dense<[2]> : tensor<1xi64>} : (tensor<32xf32>, tensor<3xi64>) -> tensor<7x8x9xf32>
  func.return %0 : tensor<7x8x9xf32>
}

// -----

func.func @broadcast_in_dim_bad_dimension_rank(%arg0: tensor<1x2xi32>) -> tensor<1x2x3xi32> {
  // expected-error@+1 {{broadcast_dimensions has rank 2 instead of rank 1}}
  %0 = "mhlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<[[1,1],[1,1]]> : tensor<2x2xi64>} : (tensor<1x2xi32>) -> tensor<1x2x3xi32>
  func.return %0 : tensor<1x2x3xi32>
}

// -----

func.func @broadcast_in_dim_bad_dimension_size(%arg0: tensor<1x2xi32>) -> tensor<1x2x3xi32> {
  // expected-error@+1 {{broadcast_dimensions size (1) does not match operand rank (2)}}
  %0 = "mhlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<[1]> : tensor<1xi64>} : (tensor<1x2xi32>) -> tensor<1x2x3xi32>
  func.return %0 : tensor<1x2x3xi32>
}

// -----

func.func @broadcast_in_dim_bad_rank_decrease(%arg0: tensor<1x2x3xi32>) -> tensor<3xi32> {
  // expected-error@+1 {{broadcast_dimensions contains invalid value 1 for result with rank 1}}
  %0 = "mhlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<[0,1,2]> : tensor<3xi64>} : (tensor<1x2x3xi32>) -> tensor<3xi32>
  func.return %0 : tensor<3xi32>
}

// -----

func.func @broadcast_in_dim_duplicate_bcast_dimensions(%arg0: tensor<1x1x3xi32>) -> tensor<1x2x3xi32> {
  // expected-error@+1 {{broadcast_dimensions should not have duplicates}}
  %0 = "mhlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<[0,0,2]> : tensor<3xi64>} : (tensor<1x1x3xi32>) -> tensor<1x2x3xi32>
  func.return %0 : tensor<1x2x3xi32>
}

// -----

func.func @broadcast_in_dim_dimension_values_too_large(%arg0: tensor<1x2xi32>) -> tensor<1x2x3xi32> {
  // expected-error@+1 {{broadcast_dimensions contains invalid value 9 for result with rank 3}}
  %0 = "mhlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<[9, 2]> : tensor<2xi64>} : (tensor<1x2xi32>) -> tensor<1x2x3xi32>
  func.return %0 : tensor<1x2x3xi32>
}

// -----

func.func @broadcast_in_dim_bad_shape_mismatch(%arg0: tensor<3xi32>) -> tensor<1x2x3xi32> {
  // expected-error@+1 {{size of operand dimension 0 (3) is not equal to 1 or size of result dimension 1 (2)}}
  %0 = "mhlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<[1]> : tensor<1xi64>} : (tensor<3xi32>) -> tensor<1x2x3xi32>
  func.return %0 : tensor<1x2x3xi32>
}

// -----

// Regression test for b/180052624, where this was improperly marked as an
// invalid mhlo.broadcast_in_dim op.
// CHECK-LABEL: func @broadcast_in_dim_dynamic_shaped_operand
func.func @broadcast_in_dim_dynamic_shaped_operand(%arg0 : tensor<?xf32>) -> tensor<2xf32> {
  %0 = "mhlo.broadcast_in_dim"(%arg0) {
    broadcast_dimensions = dense<0> : tensor<1xi64>
  } : (tensor<?xf32>) -> tensor<2xf32>
  func.return %0 : tensor<2xf32>
}

// -----

// Regression test for b/180052624, where this crashed verification given the
// unranked operand.
// CHECK-LABEL: func @broadcast_in_dim_unranked_operand
func.func @broadcast_in_dim_unranked_operand(%arg0 : tensor<*xf32>) -> tensor<2xf32> {
  %0 = "mhlo.broadcast_in_dim"(%arg0) {
    broadcast_dimensions = dense<0> : tensor<1xi64>
  } : (tensor<*xf32>) -> tensor<2xf32>
  func.return %0 : tensor<2xf32>
}

// -----

// CHECK-LABEL: func @if
func.func @if(%pred : tensor<i1>, %branch_operand : tensor<2xf32>) -> tensor<2xf32> {
  %0 = "mhlo.if"(%pred) ({
      "mhlo.return"(%branch_operand) : (tensor<2xf32>) -> ()
    }, {
      "mhlo.return"(%branch_operand) : (tensor<2xf32>) -> ()
    }) : (tensor<i1>) -> tensor<2xf32>
  func.return %0 : tensor<2xf32>
}

// -----

func.func @if_c1(%pred : tensor<i1>, %branch_operand : tensor<f32>) -> tensor<f32> {
  // @expected-error@+1 {{branch 0 must have 0 arguments, but found 1}}
  %0 = "mhlo.if"(%pred) ({
      ^bb0(%arg0: tensor<f32>):
        "mhlo.return"(%branch_operand) : (tensor<f32>) -> ()
    }, {
      "mhlo.return"(%branch_operand) : (tensor<f32>) -> ()
    }) : (tensor<i1>) -> tensor<f32>
  func.return %0 : tensor<f32>
}

// -----

func.func @if_c1(%pred : tensor<i1>, %branch_operand : tensor<f32>) -> tensor<f32> {
  // @expected-error@+1 {{branch 1 must have 0 arguments, but found 1}}
  %0 = "mhlo.if"(%pred) ({
      "mhlo.return"(%branch_operand) : (tensor<f32>) -> ()
    }, {
      ^bb0(%arg0: tensor<f32>):
        "mhlo.return"(%branch_operand) : (tensor<f32>) -> ()
    }) : (tensor<i1>) -> tensor<f32>
  func.return %0 : tensor<f32>
}

// -----

func.func @if_c2(%pred : tensor<i1>, %branch_operand : tensor<f32>) -> tensor<f32> {
  // @expected-error@+1 {{branch 0 and branch 1 have mismatched return types: 'tensor<f32>', 'tensor<f32>' vs 'tensor<f32>'}}
  %0 = "mhlo.if"(%pred) ({
      "mhlo.return"(%branch_operand, %branch_operand) : (tensor<f32>, tensor<f32>) -> ()
    }, {
      "mhlo.return"(%branch_operand) : (tensor<f32>) -> ()
    }) : (tensor<i1>) -> tensor<f32>
  func.return %0 : tensor<f32>
}

// -----

func.func @if_c3(%pred : tensor<i1>, %branch_operand : tensor<f32>) -> tensor<i32> {
  // @expected-error@+1 {{inferred type(s) 'tensor<f32>' are incompatible with return type(s) of operation 'tensor<i32>'}}
  %0 = "mhlo.if"(%pred) ({
      "mhlo.return"(%branch_operand) : (tensor<f32>) -> ()
    }, {
      "mhlo.return"(%branch_operand) : (tensor<f32>) -> ()
    }) : (tensor<i1>) -> tensor<i32>
  func.return %0 : tensor<i32>
}

// -----

// CHECK-LABEL: if_dynamic_branch_result
func.func @if_dynamic_branch_result(%pred : tensor<i1>, %true_branch_operand: tensor<2xf32>, %false_branch_operand : tensor<?xf32>) -> tensor<2xf32> {
  %0 = "mhlo.if"(%pred) ({
      "mhlo.return"(%true_branch_operand) : (tensor<2xf32>) -> ()
    }, {
      "mhlo.return"(%false_branch_operand) : (tensor<?xf32>) -> ()
    }) : (tensor<i1>) -> tensor<2xf32>
  func.return %0 : tensor<2xf32>
}

// -----

// CHECK-LABEL: if_dynamic_op_result
func.func @if_dynamic_op_result(%pred : tensor<i1>, %branch_operand: tensor<2xf32>) -> tensor<?xf32> {
  %0 = "mhlo.if"(%pred) ({
      "mhlo.return"(%branch_operand) : (tensor<2xf32>) -> ()
    }, {
      "mhlo.return"(%branch_operand) : (tensor<2xf32>) -> ()
    }) : (tensor<i1>) -> tensor<?xf32>
  func.return %0 : tensor<?xf32>
}

// -----

func.func @if_i1(%pred : tensor<1xi1>, %branch_operand : tensor<f32>) -> tensor<f32> {
  // @expected-error@+1 {{operand should be rank 0 tensor but got rank 1}}
  %0 = "mhlo.if"(%pred) ({
      "mhlo.return"(%branch_operand) : (tensor<f32>) -> ()
    }, {
      "mhlo.return"(%branch_operand) : (tensor<f32>) -> ()
    }) : (tensor<1xi1>) -> tensor<f32>
  func.return %0 : tensor<f32>
}

// -----

// CHECK-LABEL: if_unranked
func.func @if_unranked(%pred : tensor<i1>, %true_branch_operand: tensor<2xf32>, %false_branch_operand : tensor<*xf32>) -> tensor<*xf32> {
  %0 = "mhlo.if"(%pred) ({
      "mhlo.return"(%true_branch_operand) : (tensor<2xf32>) -> ()
    }, {
      "mhlo.return"(%false_branch_operand) : (tensor<*xf32>) -> ()
    }) : (tensor<i1>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// -----

// CHECK-LABEL: @case
func.func @case(%index : tensor<i32>, %branch_operand : tensor<2xf32>) {
  %0 = "mhlo.case"(%index) ({
      "mhlo.return"(%branch_operand) : (tensor<2xf32>) -> ()
  }, {
      "mhlo.return"(%branch_operand) : (tensor<2xf32>) -> ()
  }) : (tensor<i32>) -> tensor<2xf32>
  func.return
}

// -----

func.func @case_zero_branches(%index : tensor<i32>, %branch_operand : tensor<2xf32>) {
  // @expected-error@+1 {{expect at least one branch}}
  %0 = "mhlo.case"(%index) : (tensor<i32>) -> tensor<2xf32>
  func.return
}

// -----

// CHECK-LABEL: @case_dynamic_op_result
func.func @case_dynamic_op_result(%index : tensor<i32>, %branch_operand : tensor<2xf32>) {
  %0 = "mhlo.case"(%index) ({
      "mhlo.return"(%branch_operand) : (tensor<2xf32>) -> ()
  }, {
      "mhlo.return"(%branch_operand) : (tensor<2xf32>) -> ()
  }) : (tensor<i32>) -> tensor<?xf32>
  func.return
}

// -----

// CHECK-LABEL: @case_dynamic_branch_result
func.func @case_dynamic_branch_result(%index : tensor<i32>, %branch_operand : tensor<?xf32>) {
  %0 = "mhlo.case"(%index) ({
      "mhlo.return"(%branch_operand) : (tensor<?xf32>) -> ()
  }, {
      "mhlo.return"(%branch_operand) : (tensor<?xf32>) -> ()
  }) : (tensor<i32>) -> tensor<2xf32>
  func.return
}

// -----

// CHECK-LABEL: @case_unranked
func.func @case_unranked(%index : tensor<i32>, %branch_operand : tensor<*xf32>) {
  %0 = "mhlo.case"(%index) ({
      "mhlo.return"(%branch_operand) : (tensor<*xf32>) -> ()
  }, {
      "mhlo.return"(%branch_operand) : (tensor<*xf32>) -> ()
  }) : (tensor<i32>) -> tensor<*xf32>
  func.return
}

// -----

// CHECK-LABEL: @case_nested_different_return_types(
func.func @case_nested_different_return_types(%index : tensor<i32>, %branch_operand : tensor<f32>) {
  %0 = "mhlo.case"(%index) ({
      "mhlo.return"(%branch_operand) : (tensor<f32>) -> ()
  }, {
      %2 = "mhlo.case"(%index) ({
          "mhlo.return"(%index) : (tensor<i32>) -> ()
      }) : (tensor<i32>) -> tensor<i32>
      "mhlo.return"(%branch_operand) : (tensor<f32>) -> ()
  }) : (tensor<i32>) -> tensor<f32>
  func.return
}

// -----

func.func @case_unexpected_arguments_in_region_of_branch_1(%index : tensor<i32>, %branch_operand : tensor<f32>) {
  // @expected-error@+1 {{branch 1 must have 0 arguments, but found 1}}
  %0 = "mhlo.case"(%index) ({
      "mhlo.return"(%branch_operand) : (tensor<f32>) -> ()
  }, {
      ^bb0(%arg0: tensor<f32>):
        "mhlo.return"(%branch_operand) : (tensor<f32>) -> ()
  }) : (tensor<i32>) -> tensor<f32>
  func.return
}

// -----

func.func @case_mismatch_types_in_branches(%index: tensor<i32>, %operand_1: tensor<f32>, %operand_2: tensor<f32>, %operand_3: tensor<f32>) -> tensor<f32> {
  // expected-error@+1 {{branch 0 and branch 1 have mismatched return types: 'tensor<f32>' vs 'tensor<i32>'}}
  %0 = "mhlo.case"(%index) ({
      %1 = "mhlo.negate"(%operand_1) : (tensor<f32>) -> tensor<f32>
      "mhlo.return"(%1) : (tensor<f32>) -> ()
    },  {
      %1 = mhlo.constant dense<2> : tensor<i32>
      "mhlo.return"(%1) : (tensor<i32>) -> ()
    },  {
      %1 = "mhlo.floor"(%operand_3) : (tensor<f32>) -> tensor<f32>
      "mhlo.return"(%1) : (tensor<f32>) -> ()
    }
  ) : (tensor<i32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}

// -----

func.func @case_mismatch_return_type(%index : tensor<i32>, %branch_operand : tensor<f32>) {
  // @expected-error@+1 {{inferred type(s) 'tensor<f32>' are incompatible with return type(s) of operation 'tensor<i32>'}}
  %0 = "mhlo.case"(%index) ({
      "mhlo.return"(%branch_operand) : (tensor<f32>) -> ()
  }, {
      "mhlo.return"(%branch_operand) : (tensor<f32>) -> ()
  }) : (tensor<i32>) -> tensor<i32>
  func.return
}

// -----

// CHECK-LABEL: func @comp_eq
func.func @comp_eq(%arg0: tensor<3xi32>, %arg1: tensor<3xi32>) -> tensor<3xi1> {
  %0 = "mhlo.compare"(%arg0, %arg1) {comparison_direction = #mhlo<comparison_direction EQ>} : (tensor<3xi32>, tensor<3xi32>) -> tensor<3xi1>
  func.return %0 : tensor<3xi1>
}

// -----

func.func @comp_bad_direction(%arg0: tensor<3xi32>, %arg1: tensor<3xi32>) -> tensor<3xi1> {
  // expected-error@+1 {{'comparison_direction' failed to satisfy constraint}}
  %0 = "mhlo.compare"(%arg0, %arg1) {comparison_direction = "FOOBAR"} : (tensor<3xi32>, tensor<3xi32>) -> tensor<3xi1>
  func.return %0 : tensor<3xi1>
}

// -----

// CHECK-LABEL: func @comp_compatible_types
func.func @comp_compatible_types(%arg0: tensor<3xi32>, %arg1: tensor<3xi32>) -> tensor<?xi1> {
  %0 = "mhlo.compare"(%arg0, %arg1) {comparison_direction = #mhlo<comparison_direction EQ>} : (tensor<3xi32>, tensor<3xi32>) -> tensor<?xi1>
  func.return %0 : tensor<?xi1>
}

// -----

// CHECK-LABEL: func @comp_compatible_operand_types
func.func @comp_compatible_operand_types(%arg0: tensor<3xi32>, %arg1: tensor<?xi32>) -> tensor<?xi1> {
  %0 = "mhlo.compare"(%arg0, %arg1) {comparison_direction = #mhlo<comparison_direction EQ>} : (tensor<3xi32>, tensor<?xi32>) -> tensor<?xi1>
  func.return %0 : tensor<?xi1>
}

// -----

func.func @comp_mismatch_return_element_type(%arg0: tensor<3xi32>, %arg1: tensor<3xi32>) -> tensor<3xf16> {
  // expected-error@+1 {{result #0 must be tensor of pred (AKA boolean or 1-bit integer) values, but got 'tensor<3xf16>'}}
  %0 = "mhlo.compare"(%arg0, %arg1) {comparison_direction = #mhlo<comparison_direction EQ>} : (tensor<3xi32>, tensor<3xi32>) -> tensor<3xf16>
  func.return %0 : tensor<3xf16>
}

// -----

func.func @comp_mismatch_return_shape(%arg0: tensor<3xi32>, %arg1: tensor<3xi32>) -> tensor<2xi1> {
  // expected-error@+1 {{requires the same shape for all operands and results}}
  %0 = "mhlo.compare"(%arg0, %arg1) {comparison_direction = #mhlo<comparison_direction EQ>} : (tensor<3xi32>, tensor<3xi32>) -> tensor<2xi1>
  func.return %0 : tensor<2xi1>
}

// -----

func.func @collective_permute_invalid_sources(%arg0: tensor<128x32xf32>) -> tensor<128x32xf32> {
  // expected-error@+1 {{duplicate sources not allowed}}
  %0 = "mhlo.collective_permute"(%arg0) {
    source_target_pairs = dense<[[0, 1], [0, 2], [2, 3]]> : tensor<3x2xi64>
  } : (tensor<128x32xf32>) -> tensor<128x32xf32>
  func.return %0 : tensor<128x32xf32>
}

// -----

func.func @collective_permute_invalid_destinations(%arg0: tensor<128x32xf32>) -> tensor<128x32xf32> {
  // expected-error@+1 {{duplicate targets not allowed}}
  %0 = "mhlo.collective_permute"(%arg0) {
    source_target_pairs = dense<[[0, 1], [1, 2], [2, 1]]> : tensor<3x2xi64>
  } : (tensor<128x32xf32>) -> tensor<128x32xf32>
  func.return %0 : tensor<128x32xf32>
}

// -----

func.func @collective_permute_invalid_source_target_pairs(%arg0: tensor<128x32xf32>) -> tensor<128x32xf32> {
  // expected-error@+1 {{expect source_target_pairs attribute to be of rank 2, but got rank 1}}
  %0 = "mhlo.collective_permute"(%arg0) {
    source_target_pairs = dense<[0, 1]> : tensor<2xi64>
  } : (tensor<128x32xf32>) -> tensor<128x32xf32>
  func.return %0 : tensor<128x32xf32>
}

// -----

func.func @collective_permute_invalid_source_target_pairs(%arg0: tensor<128x32xf32>) -> tensor<128x32xf32> {
  // expected-error@+1 {{expect source_target_pairs attribute of shape (N, 2), but got (2, 3)}}
  %0 = "mhlo.collective_permute"(%arg0) {
    source_target_pairs = dense<[[0, 1, 2], [3, 4, 5]]> : tensor<2x3xi64>
  } : (tensor<128x32xf32>) -> tensor<128x32xf32>
  func.return %0 : tensor<128x32xf32>
}

// -----

func.func @collective_permute_invalid_source_target_pairs(%arg0: tensor<128x32xf32>) -> tensor<128x32xf32> {
  // expected-error@+1 {{replica ids in source_target_pairs must be >= 0}}
  %0 = "mhlo.collective_permute"(%arg0) {
    source_target_pairs = dense<[[0, 1], [-1, 0]]> : tensor<2x2xi64>
  } : (tensor<128x32xf32>) -> tensor<128x32xf32>
  func.return %0 : tensor<128x32xf32>
}

// -----

func.func @concat_0D(%arg0: tensor<i32>, %arg1: tensor<i32>)  -> tensor<2xi32> {
  // expected-error@+1 {{rank-0 values cannot be concatenated}}
  %0 = "mhlo.concatenate"(%arg0, %arg1) { dimension = 0 : i64 } : (tensor<i32>, tensor<i32>) -> tensor<2xi32>
  func.return %0 : tensor<2xi32>
}

// -----

func.func @concat_no_operands()  -> tensor<2xi32> {
  // expected-error@+1 {{expected 1 or more operands, but found 0}}
  %0 = "mhlo.concatenate"() { dimension = 0 : i64 } : () -> tensor<2xi32>
  func.return %0 : tensor<2xi32>
}

// -----

// CHECK-LABEL: @concat_1D
func.func @concat_1D(%arg0: tensor<1xi32>, %arg1: tensor<2xi32>)  -> tensor<3xi32> {
  %0 = "mhlo.concatenate"(%arg0, %arg1) { dimension = 0 : i64 } : (tensor<1xi32>, tensor<2xi32>) -> tensor<3xi32>
  func.return %0 : tensor<3xi32>
}

// -----

// CHECK-LABEL: @concat_1D
// Verifies that an error is not thrown if the inferred type is compatible with
// the result type.
func.func @concat_1D(%arg0: tensor<1xi32>, %arg1: tensor<*xi32>)  -> tensor<3xi32> {
  %0 = "mhlo.concatenate"(%arg0, %arg1) { dimension = 0 : i64 } : (tensor<1xi32>, tensor<*xi32>) -> tensor<3xi32>
  func.return %0 : tensor<3xi32>
}

// -----

func.func @concat_1D_type_error(%arg0: tensor<1xi32>, %arg1: tensor<2xf32>)  -> tensor<3xi32> {
  // expected-error@+1 {{'mhlo.concatenate' op requires the same element type for all operands and results}}
  %0 = "mhlo.concatenate"(%arg0, %arg1) { dimension = 0 : i64 } : (tensor<1xi32>, tensor<2xf32>) -> tensor<3xi32>
  func.return %0 : tensor<3xi32>
}

// -----

// CHECK-LABEL: @concat_1D_unranked
func.func @concat_1D_unranked(%arg0: tensor<1xi32>, %arg1: tensor<*xi32>)  -> tensor<*xi32> {
  %0 = "mhlo.concatenate"(%arg0, %arg1) { dimension = 0 : i64 } : (tensor<1xi32>, tensor<*xi32>) -> tensor<*xi32>
  func.return %0 : tensor<*xi32>
}

// -----

func.func @concat_1D_error(%arg0: tensor<1xi32>, %arg1: tensor<2xi32>)  -> tensor<4xi32> {
  // expected-error@+1 {{op inferred type(s) 'tensor<3xi32>' are incompatible with return type(s) of operation 'tensor<4xi32>'}}
  %0 = "mhlo.concatenate"(%arg0, %arg1) { dimension = 0 : i64 } : (tensor<1xi32>, tensor<2xi32>) -> tensor<4xi32>
  func.return %0 : tensor<4xi32>
}

// -----

func.func @concat_nagetive_dim(%arg0: tensor<1xi32>, %arg1: tensor<2xi32>)  -> tensor<3xi32> {
  // expected-error@+1 {{dimension -1 is negative}}
  %0 = "mhlo.concatenate"(%arg0, %arg1) { dimension = -1 : i64 } : (tensor<1xi32>, tensor<2xi32>) -> tensor<3xi32>
  func.return %0 : tensor<3xi32>
}

// -----

func.func @concat_nagetive_dim_with_all_unranked_operands(%arg0: tensor<*xi32>, %arg1: tensor<*xi32>)  -> tensor<*xi32> {
  // expected-error@+1 {{dimension -1 is negative}}
  %0 = "mhlo.concatenate"(%arg0, %arg1) { dimension = -1 : i64 } : (tensor<*xi32>, tensor<*xi32>) -> tensor<*xi32>
  func.return %0 : tensor<*xi32>
}

// -----

func.func @concat_outofbounds_dim(%arg0: tensor<1xi32>, %arg1: tensor<2xi32>)  -> tensor<3xi32> {
  // expected-error@+1 {{dimension 10 is out-of-bounds for input rank 1}}
  %0 = "mhlo.concatenate"(%arg0, %arg1) { dimension = 10 : i64 } : (tensor<1xi32>, tensor<2xi32>) -> tensor<3xi32>
  func.return %0 : tensor<3xi32>
}

// -----

func.func @concat_mismatch_rank(%arg0: tensor<1xi32>, %arg1: tensor<2x2xi32>)  -> tensor<3xi32> {
  // expected-error@+1 {{operands (0) and (1) do not match rank}}
  %0 = "mhlo.concatenate"(%arg0, %arg1) { dimension = 0 : i64 } : (tensor<1xi32>, tensor<2x2xi32>) -> tensor<3xi32>
  func.return %0 : tensor<3xi32>
}

// -----

func.func @concat_mismatch_dim(%arg0: tensor<1x3xi32>, %arg1: tensor<2x2xi32>)  -> tensor<3x3xi32> {
  // expected-error@+1 {{shapes of operand (0) and (1) do not match at non-concat index: (1, 3) != (2, 2) at non-concat index 1}}
  %0 = "mhlo.concatenate"(%arg0, %arg1) { dimension = 0 : i64 } : (tensor<1x3xi32>, tensor<2x2xi32>) -> tensor<3x3xi32>
  func.return %0 : tensor<3x3xi32>
}

// -----

// CHECK-LABEL: func @clamp
func.func @clamp(%arg0: tensor<1xi32>) -> tensor<1xi32> {
  %0 = "mhlo.clamp"(%arg0, %arg0, %arg0) : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
  func.return %0: tensor<1xi32>
}

// -----

// CHECK-LABEL: func @clamp_scalar
func.func @clamp_scalar(%arg0: tensor<1xi32>, %arg1: tensor<i32>) -> tensor<1xi32> {
  %0 = "mhlo.clamp"(%arg1, %arg0, %arg1) : (tensor<i32>, tensor<1xi32>, tensor<i32>) -> tensor<1xi32>
  func.return %0: tensor<1xi32>
}

// CHECK-LABEL: func @clamp_compatible_dynamic
func.func @clamp_compatible_dynamic(%arg0: tensor<?xi32>, %arg1: tensor<i32>, %arg2: tensor<3xi32>) -> tensor<?xi32> {
  %0 = "mhlo.clamp"(%arg1, %arg0, %arg2) : (tensor<i32>, tensor<?xi32>, tensor<3xi32>) -> tensor<?xi32>
  func.return %0: tensor<?xi32>
}

// CHECK-LABEL: func @clamp_compatible_dynamic_match_static
func.func @clamp_compatible_dynamic_match_static(%arg0: tensor<?xi32>, %arg1: tensor<i32>, %arg2: tensor<3xi32>) -> tensor<3xi32> {
  %0 = "mhlo.clamp"(%arg1, %arg0, %arg2) : (tensor<i32>, tensor<?xi32>, tensor<3xi32>) -> tensor<3xi32>
  func.return %0: tensor<3xi32>
}

// -----

func.func @clamp_invalid_clamp_element_type(%arg0: tensor<1xi32>, %arg1: tensor<1xf32>) -> tensor<1xi32> {
  // expected-error@+1 {{'mhlo.clamp' op requires the same element type for all operands and results}}
  %0 = "mhlo.clamp"(%arg1, %arg0, %arg0) : (tensor<1xf32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
  func.return %0: tensor<1xi32>
}

// -----

func.func @clamp_invalid_clamp_min_shape(%arg0: tensor<1xi32>, %arg1: tensor<2xi32>) -> tensor<1xi32> {
  // expected-error@+1 {{min shape [2] is not scalar and is not compatible to operand shape [1]}}
  %0 = "mhlo.clamp"(%arg1, %arg0, %arg0) : (tensor<2xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
  func.return %0: tensor<1xi32>
}

// -----

func.func @clamp_invalid_clamp_max_shape(%arg0: tensor<1xi32>, %arg1: tensor<2xi32>) -> tensor<1xi32> {
  // expected-error@+1 {{max shape [2] is not scalar and is not compatible to operand shape [1]}}
  %0 = "mhlo.clamp"(%arg0, %arg0, %arg1) : (tensor<1xi32>, tensor<1xi32>, tensor<2xi32>) -> tensor<1xi32>
  func.return %0: tensor<1xi32>
}

// -----

func.func @clamp(%arg0: tensor<1xi32>) -> tensor<1x2xi32> {
  // // expected-error@+1{{inferred type(s) 'tensor<1xi32>' are incompatible with return type(s) of operation 'tensor<1x2xi32>'}}
  %0 = "mhlo.clamp"(%arg0, %arg0, %arg0) : (tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1x2xi32>
  func.return %0: tensor<1x2xi32>
}

// -----

// CHECK-LABEL: func @cholesky
func.func @cholesky(%arg0: tensor<1x2x2xf32>) -> tensor<1x2x2xf32> {
  %0 = "mhlo.cholesky"(%arg0) { lower = true } : (tensor<1x2x2xf32>) -> tensor<1x2x2xf32>
  func.return %0: tensor<1x2x2xf32>
}

// -----

func.func @cholesky_error_nonsquare(%arg0: tensor<1x2x1xf32>) -> tensor<1x2x1xf32> {
  // expected-error@+1 {{minor dimensions of 'a' must have equal size, got shape 1, 2, 1}}
  %0 = "mhlo.cholesky"(%arg0) { lower = true } : (tensor<1x2x1xf32>) -> tensor<1x2x1xf32>
  func.return %0: tensor<1x2x1xf32>
}

// -----

func.func @cholesky_invalid_rank(%arg0: tensor<1xf32>) -> tensor<1xf32> {
  // expected-error@+1 {{argument 'a' must have rank >= 2, got shape 1}}
  %0 = "mhlo.cholesky"(%arg0) { lower = true } : (tensor<1xf32>) -> tensor<1xf32>
  func.return %0: tensor<1xf32>
}

// -----

func.func @cholesky_invalid_elt(%arg0: tensor<1x2x2xi32>) -> tensor<1x2x2xi32> {
  // expected-error@+1 {{operand #0 must be tensor of f8E4M3FN type or f8E5M2 type or 16-bit float or 32-bit float or 64-bit float or bfloat16 type or complex type with 32-bit float or 64-bit float elements values, but got 'tensor<1x2x2xi32>}}
  %0 = "mhlo.cholesky"(%arg0) { lower = true } : (tensor<1x2x2xi32>) -> tensor<1x2x2xi32>
  func.return %0: tensor<1x2x2xi32>
}

// -----

func.func @cholesky_wrong_infer_shape(%arg0: tensor<1x2x2xf32>) -> tensor<1x2x2x2xf32> {
  // expected-error@+1 {{'mhlo.cholesky' op inferred type(s) 'tensor<1x2x2xf32>' are incompatible with return type(s) of operation 'tensor<1x2x2x2xf32>'}}
  %0 = "mhlo.cholesky"(%arg0) { lower = true } : (tensor<1x2x2xf32>) -> tensor<1x2x2x2xf32>
  func.return %0: tensor<1x2x2x2xf32>
}

// -----

func.func @create_token() -> !mhlo.token {
  %0 = "mhlo.create_token"() : () -> !mhlo.token
  func.return %0: !mhlo.token
}

// -----

// CHECK-LABEL: func @dot_vector
func.func @dot_vector(%arg0: tensor<1x2xi32>, %arg1: tensor<2x1xi32>) -> tensor<1x1xi32> {
  %0 = "mhlo.dot"(%arg0, %arg1) : (tensor<1x2xi32>, tensor<2x1xi32>) -> tensor<1x1xi32>
  func.return %0: tensor<1x1xi32>
}

// -----

// CHECK-LABEL: func @dot_matrix
func.func @dot_matrix(%arg0: tensor<2x2xi32>, %arg1: tensor<2x2xi32>) -> tensor<2x2xi32> {
  %0 = "mhlo.dot"(%arg0, %arg1) : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
  func.return %0: tensor<2x2xi32>
}

// -----

// CHECK-LABEL: func @dot_precision_config
func.func @dot_precision_config(%arg0: tensor<2x2xi32>, %arg1: tensor<2x2xi32>) -> tensor<2x2xi32> {
  %0 = "mhlo.dot"(%arg0, %arg1) {precision_config = [#mhlo<precision HIGH>, #mhlo<precision HIGHEST>]} : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
  func.return %0: tensor<2x2xi32>
}

// -----

func.func @dot_precision_invalid_precision_config(%arg0: tensor<2x2xi32>, %arg1: tensor<2x2xi32>) -> tensor<2x2xi32> {
  // expected-error@+1 {{expects precision config to be empty or have <= 2 elements}}
  %0 = "mhlo.dot"(%arg0, %arg1) {precision_config = [#mhlo<precision HIGH>, #mhlo<precision HIGH>, #mhlo<precision HIGH>]} : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
  func.return %0: tensor<2x2xi32>
}

// -----

func.func @dot_bad_precision_config(%arg0: tensor<2x2xi32>, %arg1: tensor<2x2xi32>) -> tensor<2x2xi32> {
  // expected-error@+1 {{'precision_config' failed to satisfy constraint}}
  %0 = "mhlo.dot"(%arg0, %arg1) {precision_config = ["FOO", #mhlo<precision HIGHEST>]} : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
  func.return %0: tensor<2x2xi32>
}

// -----

func.func @dot_more_dynamic_output_type(%arg0: tensor<3xf32>, %arg1: tensor<?x3xf32>) -> tensor<?xf32> {
  %0 = "mhlo.dot"(%arg0, %arg1) : (tensor<3xf32>, tensor<?x3xf32>) -> tensor<?xf32>
  func.return %0 : tensor<?xf32>
}

// -----

func.func @dot_cannot_infer_type(%arg0: tensor<?x?x3xf32>, %arg1: tensor<?x3x?xf32>) -> tensor<*xf32> {
  // expected-error@+1 {{expected both lhs/rhs ranks to be either 1 or 2}}
  %0 = "mhlo.dot"(%arg0, %arg1) : (tensor<?x?x3xf32>, tensor<?x3x?xf32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// -----

func.func @dot_result_type_mismatch_with_inferred_type(%arg0: tensor<?x3xf32>, %arg1: tensor<3xf32>) -> tensor<3x?xf32> {
  // expected-error@+1 {{inferred shape '[?]' is incompatible with return type of operation 'tensor<3x?xf32>'}}
  %0 = "mhlo.dot"(%arg0, %arg1) : (tensor<?x3xf32>, tensor<3xf32>) -> tensor<3x?xf32>
  func.return %0 : tensor<3x?xf32>
}

// -----

func.func @dot_result_type_match_with_inferred_type(%arg0: tensor<?x3xf32>, %arg1: tensor<3xf32>) -> tensor<*xf32> {
  %0 = "mhlo.dot"(%arg0, %arg1) : (tensor<?x3xf32>, tensor<3xf32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// -----

// CHECK-LABEL: func @dot_legal_unranked_rank_type
func.func @dot_legal_unranked_rank_type(%arg0: tensor<*xf32>, %arg1: tensor<*xf32>) -> tensor<2x2xf32> {
  // unrank legal test
  %0 = "mhlo.dot"(%arg0, %arg1) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
  // vector dot vector
  %1 = tensor.cast %arg0 : tensor<*xf32> to tensor<3xf32>
  %2 = tensor.cast %arg0 : tensor<*xf32> to tensor<3xf32>
  %3 = "mhlo.dot"(%1, %2) : (tensor<3xf32>, tensor<3xf32>) -> tensor<f32>
  // matrix dot vector
  %4 = tensor.cast %arg0 : tensor<*xf32> to tensor<2x3xf32>
  %5 = tensor.cast %arg1 : tensor<*xf32> to tensor<3xf32>
  %6 = "mhlo.dot"(%4, %5) : (tensor<2x3xf32>, tensor<3xf32>) -> tensor<2xf32>
  // matrix dot matrix
  %7 = tensor.cast %arg0 : tensor<*xf32> to tensor<2x3xf32>
  %8 = tensor.cast %arg1 : tensor<*xf32> to tensor<3x2xf32>
  %9 = "mhlo.dot"(%7, %8) : (tensor<2x3xf32>, tensor<3x2xf32>) -> tensor<2x2xf32>

  func.return %9 : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @imag_fp_input
func.func @imag_fp_input(%arg0: tensor<*xf32>) -> tensor<*xf32> {
  %0 = "mhlo.imag"(%arg0) : (tensor<*xf32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// -----

func.func @imag_int_input(%arg0: tensor<*xi32>) -> tensor<*xi32> {
  // expected-error@+1 {{operand #0 must be tensor of f8E4M3FN type or f8E5M2 type or 16-bit float or 32-bit float or 64-bit float or bfloat16 type or complex type with 32-bit float or 64-bit float elements values, but got 'tensor<*xi32>'}}
  %0 = "mhlo.imag"(%arg0) : (tensor<*xi32>) -> tensor<*xi32>
  func.return %0 : tensor<*xi32>
}

// -----

// CHECK-LABEL: func @imag_complex_input
func.func @imag_complex_input(%arg0: tensor<2x3xcomplex<f32>>) -> tensor<2x3xf32> {
  %0 = "mhlo.imag"(%arg0) : (tensor<2x3xcomplex<f32>>) -> tensor<2x3xf32>
  func.return %0 : tensor<2x3xf32>
}

// -----

func.func @imag_mismatch_return_shape(%arg0: tensor<2xf32>) -> tensor<4xf32> {
  // expected-error@+1 {{all non-scalar operands/results must have the same shape and base type}}
  %0 = "mhlo.imag"(%arg0) : (tensor<2xf32>) -> tensor<4xf32>
  func.return %0 : tensor<4xf32>
}

// -----

func.func @imag_mismatch_return_element_type(%arg0: tensor<2xf32>) -> tensor<2xf16> {
  // expected-error@+1 {{inferred type(s) 'tensor<2xf32>' are incompatible with return type(s) of operation 'tensor<2xf16>'}}
  %0 = "mhlo.imag"(%arg0) : (tensor<2xf32>) -> tensor<2xf16>
  func.return %0 : tensor<2xf16>
}

// -----

func.func @infeed_non_token_second_result(%token: !mhlo.token) -> tuple<tensor<i32>, tensor<i32>> {
  // expected-error@+1 {{last element of result types is expected to be of token type, but got 'tensor<i32>'}}
  %0:2 = "mhlo.infeed"(%token) {infeed_config = "foobar", layout = [[[0]], [0]]} : (!mhlo.token) -> (tensor<i32>, tensor<i32>)
  %1 = "mhlo.tuple"(%0#0, %0#1) : (tensor<i32>, tensor<i32>) -> tuple<tensor<i32>, tensor<i32>>
  func.return %1 : tuple<tensor<i32>, tensor<i32>>
}

// -----

func.func @main(%arg0: !mhlo.token) -> tensor<3x3xi32> {
  // expected-error@+1 {{layout-attribute size must be 2 (which is the number of op-results - 1 (for token result)), but got 1}}
  %0:3 = "mhlo.infeed"(%arg0) {infeed_config = "foobar", layout=[[0, 1]]} : (!mhlo.token) -> (tensor<3x3xi32>, tensor<i1>, !mhlo.token)
  func.return %0#0 : tensor<3x3xi32>
}

// -----

func.func @main(%arg0: !mhlo.token) -> !mhlo.token {
  // expected-error@+1 {{layout-attribute size must be 0 (which is the number of op-results - 1 (for token result)), but got 1}}
  %0:1 = "mhlo.infeed"(%arg0) {infeed_config = "foobar", layout=[[]]} : (!mhlo.token) -> (!mhlo.token)
  func.return %0#0 : !mhlo.token
}

// -----

func.func @main(%arg0: !mhlo.token) -> tensor<3x3xi32> {
  // expected-error@+1 {{layout-attribute expected to have elements of type array, but got 0 : i64}}
  %0:2 = "mhlo.infeed"(%arg0) {infeed_config = "foobar", layout=[0]} : (!mhlo.token) -> (tensor<3x3xi32>, !mhlo.token)
  func.return %0#0 : tensor<3x3xi32>
}

// -----

func.func @main(%arg0: !mhlo.token) -> tensor<3x3xi32> {
  // expected-error@+1 {{ayout-attribute's leaf elements are expected to be of type integer, but got []}}
  %0:2 = "mhlo.infeed"(%arg0) {infeed_config = "foobar", layout=[[0,[]]]} : (!mhlo.token) -> (tensor<3x3xi32>, !mhlo.token)
  func.return %0#0 : tensor<3x3xi32>
}

// -----

func.func @iota_scalar() -> tensor<i32> {
  // expected-error@+1 {{does not support scalars}}
  %0 = "mhlo.iota"() {iota_dimension = 0 : i64} : () -> tensor<i32>
  func.return %0 : tensor<i32>
}

// -----

func.func @iota_invalid_iota_dimension() -> tensor<4xi32> {
  // expected-error@+1 {{iota dimension cannot go beyond the output rank or be negative}}
  %0 = "mhlo.iota"() {iota_dimension = 1 : i64} : () -> tensor<4xi32>
  func.return %0 : tensor<4xi32>
}

// -----

// CHECK-LABEL: func @map
func.func @map(%arg0: tensor<4x5xf32>, %arg1: tensor<4x5xf32>) -> tensor<4x5xf32> {
  %0 = "mhlo.map"(%arg0, %arg1) ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
    %1 = mhlo.constant dense<2.0> : tensor<f32>
    "mhlo.return"(%1) : (tensor<f32>) -> ()
  }) {dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<4x5xf32>, tensor<4x5xf32>) -> tensor<4x5xf32>
  func.return %0 : tensor<4x5xf32>
}

// -----

// CHECK-LABEL: func @map_heterogeneous_inputs
func.func @map_heterogeneous_inputs(%arg0: tensor<2xf32>, %arg1: tensor<2xi32>) -> tensor<2xf32> {
  %0 = "mhlo.map"(%arg0, %arg1) ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<i32>):
    "mhlo.return"(%arg2) : (tensor<f32>) -> ()
  }) {dimensions = dense<0> : tensor<1xi64>} : (tensor<2xf32>, tensor<2xi32>) -> tensor<2xf32>
  func.return %0 : tensor<2xf32>
}

// -----

// CHECK-LABEL: func @map_scalar_operands
func.func @map_scalar_operands(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
  %0 = "mhlo.map"(%arg0, %arg1) ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
    %1 = mhlo.add %arg2, %arg3 {name = "add"} : tensor<f32>
    "mhlo.return"(%1) : (tensor<f32>) -> ()
  }) {dimensions = dense<> : tensor<0xi64>} : (tensor<f32>, tensor<f32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}

// -----

// CHECK-LABEL: func @map_unranked
func.func @map_unranked(%arg0: tensor<*xf32>, %arg1: tensor<*xf32>) -> tensor<*xf32> {
  %0 = "mhlo.map"(%arg0, %arg1) ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
    %1 = mhlo.add %arg2, %arg3 {name = "add"} : tensor<f32>
    "mhlo.return"(%1) : (tensor<f32>) -> ()
  }) {dimensions = dense<0> : tensor<1xi64>} : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// -----

func.func @map_mismatched_args(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  // expected-error@+1 {{expects number of operands to match the arity of map computation, but got: 2 and 1}}
  %0 = "mhlo.map"(%arg0, %arg1) ({
    ^bb0(%arg: tensor<f32>):
    %1 = mhlo.add %arg, %arg {name = "add"} : tensor<f32>
    "mhlo.return"(%1) : (tensor<f32>) -> ()
  }) {dimensions = dense<0> : tensor<1xi64>} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  func.return %0 : tensor<4xf32>
}

// -----

func.func @map_non_scalar_computation_operand(%arg0: tensor<4x5xf32>, %arg1: tensor<4x5xf32>) -> tensor<4x5xf32> {
  // expected-error@+1 {{computation arguments must be 0-rank tensor, but got: arg #1 of type 'tensor<5xf32>'}}
  %0 = "mhlo.map"(%arg0, %arg1) ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<5xf32>):
    %1 = mhlo.constant dense<2.0> : tensor<f32>
    "mhlo.return"(%1) : (tensor<f32>) -> ()
  }) {dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<4x5xf32>, tensor<4x5xf32>) -> tensor<4x5xf32>
  func.return %0 : tensor<4x5xf32>
}

// -----

func.func @map_mismatch_operand_and_computation_args(%arg0: tensor<4x5xf32>, %arg1: tensor<4x5xf32>) -> tensor<4x5xf32> {
  // expected-error@+1 {{element type of operands and computation arguments must match, but got: 'f32' and 'i32'}}
  %0 = "mhlo.map"(%arg0, %arg1) ({
    ^bb0(%arg2: tensor<i32>, %arg3: tensor<i32>):
    %1 = mhlo.constant dense<2.0> : tensor<f32>
    "mhlo.return"(%1) : (tensor<f32>) -> ()
  }) {dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<4x5xf32>, tensor<4x5xf32>) -> tensor<4x5xf32>
  func.return %0 : tensor<4x5xf32>
}

// -----

func.func @map_invalid_number_of_computation_output(%arg0: tensor<4x5xf32>, %arg1: tensor<4x5xf32>) -> tensor<4x5xf32> {
  // expected-error@+1 {{computation must return single output, but got: 0}}
  %0 = "mhlo.map"(%arg0, %arg1) ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
    %1 = mhlo.constant dense<2.0> : tensor<f32>
    "mhlo.return"() : () -> ()
  }) {dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<4x5xf32>, tensor<4x5xf32>) -> tensor<4x5xf32>
  func.return %0 : tensor<4x5xf32>
}

// -----

func.func @main_non_scalar_computation_output(%arg0: tensor<4x5xf32>, %arg1: tensor<4x5xf32>) -> tensor<4x5xf32> {
  // expected-error@+1 {{computation must return 0-rank tensor, but got: 'tensor<5xf32>'}}
  %0 = "mhlo.map"(%arg0, %arg1) ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
    %1 = mhlo.constant dense<2.0> : tensor<5xf32>
    "mhlo.return"(%1) : (tensor<5xf32>) -> ()
  }) {dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<4x5xf32>, tensor<4x5xf32>) -> tensor<4x5xf32>
  func.return %0 : tensor<4x5xf32>
}

// -----

func.func @mismatch_computation_output_type(%arg0: tensor<4x5xf32>, %arg1: tensor<4x5xf32>) -> tensor<4x5xf32> {
  // expected-error@+1 {{inferred type(s) 'tensor<4x5xi32>' are incompatible with return type(s) of operation 'tensor<4x5xf32>'}}
  %0 = "mhlo.map"(%arg0, %arg1) ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
    %1 = mhlo.constant dense<2> : tensor<i32>
    "mhlo.return"(%1) : (tensor<i32>) -> ()
  }) {dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<4x5xf32>, tensor<4x5xf32>) -> tensor<4x5xf32>
  func.return %0 : tensor<4x5xf32>
}

// -----

func.func @map_invalid_dimension_numbers(%arg0: tensor<4x5xf32>, %arg1: tensor<4x5xf32>) -> tensor<4x5xf32> {
  // expected-error@+1 {{requires monotonically increasing dimension numbers, but got: dense<[1, 0]> : tensor<2xi64>}}
  %0 = "mhlo.map"(%arg0, %arg1) ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
    %1 = mhlo.add %arg2, %arg3 {name = "add"} : tensor<f32>
    "mhlo.return"(%1) : (tensor<f32>) -> ()
  }) {dimensions = dense<[1, 0]> : tensor<2xi64>} : (tensor<4x5xf32>, tensor<4x5xf32>) -> tensor<4x5xf32>
  func.return %0 : tensor<4x5xf32>
}

// -----

func.func @map_mismatch_arguments_and_dimensions(%arg0: tensor<4x5xf32>, %arg1: tensor<4x5xf32>) -> tensor<4x5xf32> {
  // expected-error@+1 {{applied to a subset of dimensions currently not supported: operand dimensions = 2, requested map dimensions size = 3}}
  %0 = "mhlo.map"(%arg0, %arg1) ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
    %1 = mhlo.add %arg2, %arg3 {name = "add"} : tensor<f32>
    "mhlo.return"(%1) : (tensor<f32>) -> ()
  }) {dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<4x5xf32>, tensor<4x5xf32>) -> tensor<4x5xf32>
  func.return %0 : tensor<4x5xf32>
}

// -----

// CHECK-LABEL: func @outfeed
func.func @outfeed(%arg0: tensor<3x3x3xi32>, %arg1: !mhlo.token) -> !mhlo.token {
  %0 = "mhlo.outfeed"(%arg0, %arg1) {
    outfeed_config = ""
  } : (tensor<3x3x3xi32>, !mhlo.token) -> !mhlo.token
  func.return %0 : !mhlo.token
}

// -----

// CHECK-LABEL: func @real_fp_input
func.func @real_fp_input(%arg0: tensor<*xf32>) -> tensor<*xf32> {
  %0 = "mhlo.real"(%arg0) : (tensor<*xf32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// -----

func.func @real_int_input(%arg0: tensor<*xi32>) -> tensor<*xi32> {
  // expected-error@+1 {{operand #0 must be tensor of f8E4M3FN type or f8E5M2 type or 16-bit float or 32-bit float or 64-bit float or bfloat16 type or complex type with 32-bit float or 64-bit float elements values, but got 'tensor<*xi32>'}}
  %0 = "mhlo.real"(%arg0) : (tensor<*xi32>) -> tensor<*xi32>
  func.return %0 : tensor<*xi32>
}

// -----

// CHECK-LABEL: func @real_complex_input
func.func @real_complex_input(%arg0: tensor<2x3xcomplex<f32>>) -> tensor<2x3xf32> {
  %0 = "mhlo.real"(%arg0) : (tensor<2x3xcomplex<f32>>) -> tensor<2x3xf32>
  func.return %0 : tensor<2x3xf32>
}

// -----

func.func @real_mismatch_return_shape(%arg0: tensor<2x3xcomplex<f32>>) -> tensor<2x10xf32> {
  // expected-error@+1 {{all non-scalar operands/results must have the same shape and base type}}
  %0 = "mhlo.real"(%arg0) : (tensor<2x3xcomplex<f32>>) -> tensor<2x10xf32>
  func.return %0 : tensor<2x10xf32>
}

// -----

func.func @real_mismatch_return_element_type(%arg0: tensor<2x3xcomplex<f32>>) -> tensor<2x3xf16> {
  // expected-error@+1 {{inferred type(s) 'tensor<2x3xf32>' are incompatible with return type(s) of operation 'tensor<2x3xf16>'}}
  %0 = "mhlo.real"(%arg0) : (tensor<2x3xcomplex<f32>>) -> tensor<2x3xf16>
  func.return %0 : tensor<2x3xf16>
}

// -----

func.func @recv_non_token_second_result(%token: !mhlo.token) -> tuple<tensor<3x4xi32>, tensor<i32>> {
  // expected-error@+1 {{last element of result types is expected to be of token type, but got 'tensor<i32>'}}
  %0:2 = "mhlo.recv"(%token) {
    channel_handle = #mhlo.channel_handle<
      handle = 5,
      type = 3  // Host to device channel
    >,
    is_host_transfer = true
  } : (!mhlo.token) -> (tensor<3x4xi32>, tensor<i32>)
  %1 =  "mhlo.tuple"(%0#0, %0#1) : (tensor<3x4xi32>, tensor<i32>) -> tuple<tensor<3x4xi32>, tensor<i32>>
  func.return %1 : tuple<tensor<3x4xi32>, tensor<i32>>
}

// -----

// CHECK-LABEL: func @replica_id
func.func @replica_id() -> tensor<ui32> {
  %0 = "mhlo.replica_id"() : () -> tensor<ui32>
  func.return %0 : tensor<ui32>
}

// -----

// CHECK-LABEL: func @rng_bit_generator
func.func @rng_bit_generator(%arg0: tensor<2xui64>) -> (tensor<2xui64>, tensor<10x12xui32>) {
  %0, %1 = "mhlo.rng_bit_generator"(%arg0) {rng_algorithm = #mhlo.rng_algorithm<DEFAULT>} : (tensor<2xui64>) -> (tensor<2xui64>, tensor<10x12xui32>)
  func.return %0, %1 : tensor<2xui64>, tensor<10x12xui32>
}

// -----

func.func @rng_bit_generator(%arg0: tensor<2xui64>) -> (tensor<2xui64>, tensor<10x12xui32>) {
  // expected-error@+1 {{output state shape must be compatible with initial state shape. Got: 'tensor<2xui64>' and 'tensor<3xui64>'}}
  %0, %1 = "mhlo.rng_bit_generator"(%arg0) {rng_algorithm = #mhlo.rng_algorithm<DEFAULT>} : (tensor<2xui64>) -> (tensor<3xui64>, tensor<10x12xui32>)
  func.return %0, %1 : tensor<3xui64>, tensor<10x12xui32>
}

// -----

// CHECK-LABEL: func @rng_bit_generator_dynamic
func.func @rng_bit_generator_dynamic(%arg0: tensor<?xui64>) -> (tensor<?xui64>, tensor<10x12xui32>) {
  %0, %1 = "mhlo.rng_bit_generator"(%arg0) {rng_algorithm = #mhlo.rng_algorithm<DEFAULT>} : (tensor<?xui64>) -> (tensor<?xui64>, tensor<10x12xui32>)
  func.return %0, %1 : tensor<?xui64>, tensor<10x12xui32>
}

// -----

// CHECK-LABEL: func @rng_normal
func.func @rng_normal(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<2x3x5xf32> {
  %cst = "mhlo.constant"() {value = dense<[2, 3, 5]> : tensor<3xi64>} : () -> tensor<3xi64>
  %0 = "mhlo.rng"(%arg0, %arg1, %cst) {rng_distribution = #mhlo.rng_distribution<NORMAL>}: (tensor<f32>, tensor<f32>, tensor<3xi64>) -> tensor<2x3x5xf32>
  func.return %0 : tensor<2x3x5xf32>
}

// -----

// CHECK-LABEL: func @rng_normal_no_constant
func.func @rng_normal_no_constant(%a: tensor<f32>, %b: tensor<f32>, %shape: tensor<3xi64>) -> tensor<?x?x?xf32> {
  %0 = "mhlo.rng"(%a, %b, %shape) {rng_distribution = #mhlo.rng_distribution<NORMAL>}: (tensor<f32>, tensor<f32>, tensor<3xi64>) -> tensor<?x?x?xf32>
  func.return %0 : tensor<?x?x?xf32>
}

// -----

// CHECK-LABEL: func @rng_normal_dynamic_dim
func.func @rng_normal_dynamic_dim(%a: tensor<f32>, %b: tensor<f32>, %shape: tensor<?xi64>) -> tensor<*xf32> {
  %0 = "mhlo.rng"(%a, %b, %shape) {rng_distribution = #mhlo.rng_distribution<NORMAL>}: (tensor<f32>, tensor<f32>, tensor<?xi64>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// -----

func.func @rng_normal_invalid_shape(%arg0: tensor<f32>, %arg1: tensor<f32>) {
  %cst = "mhlo.constant"() {value = dense<7> : tensor<1xi64>} : () -> tensor<1xi64>
  // expected-error @+1 {{inferred type(s) 'tensor<7xf32>' are incompatible with return type(s) of operation 'tensor<12xf32>'}}
  %0 = "mhlo.rng"(%arg0, %arg1, %cst) {rng_distribution = #mhlo.rng_distribution<NORMAL>}: (tensor<f32>, tensor<f32>, tensor<1xi64>) -> tensor<12xf32>
  func.return
}

// -----

func.func @rng_normal_invalid_mu_rank(%mu: tensor<1xf32>, %sigma: tensor<f32>) -> tensor<2x3x5xf32> {
  %shape = mhlo.constant dense<[2, 3, 5]> : tensor<3xi64>
  // expected-error@+1 {{#0 must be 0D tensor of pred (AKA boolean or 1-bit integer) or 4/8/16/32/64-bit signless integer or 4/8/16/32/64-bit unsigned integer or f8E4M3FN type or f8E5M2 type or 16-bit float or 32-bit float or 64-bit float or bfloat16 type values, but got 'tensor<1xf32>'}}
  %0 = "mhlo.rng"(%mu, %sigma, %shape) {rng_distribution = #mhlo.rng_distribution<NORMAL>}: (tensor<1xf32>, tensor<f32>, tensor<3xi64>) -> tensor<2x3x5xf32>
  func.return %0 : tensor<2x3x5xf32>
}

// -----

func.func @rng_normal_invalid_sigma_rank(%mu: tensor<f32>, %sigma: tensor<1xf32>) -> tensor<2x3x5xf32> {
  %shape = mhlo.constant dense<[2, 3, 5]> : tensor<3xi64>
  // expected-error@+1 {{#1 must be 0D tensor of pred (AKA boolean or 1-bit integer) or 4/8/16/32/64-bit signless integer or 4/8/16/32/64-bit unsigned integer or f8E4M3FN type or f8E5M2 type or 16-bit float or 32-bit float or 64-bit float or bfloat16 type values, but got 'tensor<1xf32>'}}
  %0 = "mhlo.rng"(%mu, %sigma, %shape) {rng_distribution = #mhlo.rng_distribution<NORMAL>}: (tensor<f32>, tensor<1xf32>, tensor<3xi64>) -> tensor<2x3x5xf32>
  func.return %0 : tensor<2x3x5xf32>
}

// -----

func.func @rng_normal_invalid_shape_rank(%mu: tensor<f32>, %sigma: tensor<f32>) -> tensor<2x3x5xf32> {
  %shape = mhlo.constant dense<[[2, 3, 5]]> : tensor<1x3xi64>
  // expected-error@+1 {{operand #2 must be 1D tensor of index or 4/8/16/32/64-bit signless integer or 4/8/16/32/64-bit unsigned integer values, but got 'tensor<1x3xi64>'}}
  %0 = "mhlo.rng"(%mu, %sigma, %shape) {rng_distribution = #mhlo.rng_distribution<NORMAL>}: (tensor<f32>, tensor<f32>, tensor<1x3xi64>) -> tensor<2x3x5xf32>
  func.return %0 : tensor<2x3x5xf32>
}

// -----

func.func @rng_normal_invalid_type(%arg0: tensor<complex<f32>>, %arg1: tensor<f32>) {
  %cst = "mhlo.constant"() {value = dense<7> : tensor<1xi64>} : () -> tensor<1xi64>
  // expected-error @+1 {{#0 must be 0D tensor of pred (AKA boolean or 1-bit integer) or 4/8/16/32/64-bit signless integer or 4/8/16/32/64-bit unsigned integer or f8E4M3FN type or f8E5M2 type or 16-bit float or 32-bit float or 64-bit float or bfloat16 type values, but got 'tensor<complex<f32>>'}}
  %0 = "mhlo.rng"(%arg0, %arg1, %cst) {rng_distribution = #mhlo.rng_distribution<NORMAL>}: (tensor<complex<f32>>, tensor<f32>, tensor<1xi64>) -> tensor<7xf32>
  func.return
}

// -----

// CHECK-LABEL: func @rng_uniform
func.func @rng_uniform(%a: tensor<f32>, %b: tensor<f32>) -> tensor<2x3x5xf32> {
  %shape = mhlo.constant dense<[2, 3, 5]> : tensor<3xi64>
  %0 = "mhlo.rng"(%a, %b, %shape) {rng_distribution = #mhlo.rng_distribution<UNIFORM>}: (tensor<f32>, tensor<f32>, tensor<3xi64>) -> tensor<2x3x5xf32>
  func.return %0 : tensor<2x3x5xf32>
}

// -----

// CHECK-LABEL: func @rng_uniform_no_constant
func.func @rng_uniform_no_constant(%a: tensor<f32>, %b: tensor<f32>, %shape: tensor<3xi64>) -> tensor<?x?x?xf32> {
  %0 = "mhlo.rng"(%a, %b, %shape) {rng_distribution = #mhlo.rng_distribution<UNIFORM>}: (tensor<f32>, tensor<f32>, tensor<3xi64>) -> tensor<?x?x?xf32>
  func.return %0 : tensor<?x?x?xf32>
}

// -----

// CHECK-LABEL: func @rng_uniform_dynamic_dim
func.func @rng_uniform_dynamic_dim(%a: tensor<f32>, %b: tensor<f32>, %shape: tensor<?xi64>) -> tensor<*xf32> {
  %0 = "mhlo.rng"(%a, %b, %shape) {rng_distribution = #mhlo.rng_distribution<UNIFORM>}: (tensor<f32>, tensor<f32>, tensor<?xi64>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// -----

func.func @rng_uniform_invalid_shape(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<7xi64>) {
  // expected-error @+1 {{inferred type(s) 'tensor<?x?x?x?x?x?x?xf32>' are incompatible with return type(s) of operation 'tensor<?xf32>'}}
  %0 = "mhlo.rng"(%arg0, %arg1, %arg2) {rng_distribution = #mhlo.rng_distribution<UNIFORM>}: (tensor<f32>, tensor<f32>, tensor<7xi64>) -> tensor<?xf32>
  func.return
}

// -----

func.func @rng_uniform_invalid_a_rank(%a: tensor<1xf32>, %b: tensor<f32>) -> tensor<2x3x5xf32> {
  %shape = mhlo.constant dense<[2, 3, 5]> : tensor<3xi64>
  // expected-error@+1 {{operand #0 must be 0D tensor of pred (AKA boolean or 1-bit integer) or 4/8/16/32/64-bit signless integer or 4/8/16/32/64-bit unsigned integer or f8E4M3FN type or f8E5M2 type or 16-bit float or 32-bit float or 64-bit float or bfloat16 type values, but got 'tensor<1xf32>'}}
  %0 = "mhlo.rng"(%a, %b, %shape) {rng_distribution = #mhlo.rng_distribution<UNIFORM>}: (tensor<1xf32>, tensor<f32>, tensor<3xi64>) -> tensor<2x3x5xf32>
  func.return %0 : tensor<2x3x5xf32>
}


// -----

func.func @rng_uniform_invalid_b_rank(%a: tensor<f32>, %b: tensor<1xf32>) -> tensor<2x3x5xf32> {
  %shape = mhlo.constant dense<[2, 3, 5]> : tensor<3xi64>
  // expected-error@+1 {{operand #1 must be 0D tensor of pred (AKA boolean or 1-bit integer) or 4/8/16/32/64-bit signless integer or 4/8/16/32/64-bit unsigned integer or f8E4M3FN type or f8E5M2 type or 16-bit float or 32-bit float or 64-bit float or bfloat16 type values, but got 'tensor<1xf32>'}}
  %0 = "mhlo.rng"(%a, %b, %shape) {rng_distribution = #mhlo.rng_distribution<UNIFORM>}: (tensor<f32>, tensor<1xf32>, tensor<3xi64>) -> tensor<2x3x5xf32>
  func.return %0 : tensor<2x3x5xf32>
}

// -----

func.func @rng_uniform_invalid_shape_rank(%a: tensor<f32>, %b: tensor<f32>) -> tensor<2x3x5xf32> {
  %shape = mhlo.constant dense<[[2, 3, 5]]> : tensor<1x3xi64>
  // expected-error@+1 {{operand #2 must be 1D tensor of index or 4/8/16/32/64-bit signless integer or 4/8/16/32/64-bit unsigned integer values, but got 'tensor<1x3xi64>'}}
  %0 = "mhlo.rng"(%a, %b, %shape) {rng_distribution = #mhlo.rng_distribution<UNIFORM>}: (tensor<f32>, tensor<f32>, tensor<1x3xi64>) -> tensor<2x3x5xf32>
  func.return %0 : tensor<2x3x5xf32>
}

// -----

func.func @rng_uniform_invalid_type(%a: tensor<complex<f32>>, %b: tensor<f32>) -> tensor<2x3x5xf32> {
  %shape = mhlo.constant dense<[2, 3, 5]> : tensor<3xi64>
  // expected-error@+1 {{operand #0 must be 0D tensor of pred (AKA boolean or 1-bit integer) or 4/8/16/32/64-bit signless integer or 4/8/16/32/64-bit unsigned integer or f8E4M3FN type or f8E5M2 type or 16-bit float or 32-bit float or 64-bit float or bfloat16 type values, but got 'tensor<complex<f32>>'}}
  %0 = "mhlo.rng"(%a, %b, %shape) {rng_distribution = #mhlo.rng_distribution<UNIFORM>}: (tensor<complex<f32>>, tensor<f32>, tensor<3xi64>) -> tensor<2x3x5xf32>
  func.return %0 : tensor<2x3x5xf32>
}

// -----

// CHECK-LABEL: func @select
func.func @select(%arg0: tensor<2x3xi1>, %arg1: tensor<2x3xi32>, %arg2: tensor<2x3xi32>) -> tensor<2x3xi32> {
  %0 = "mhlo.select"(%arg0, %arg1, %arg2) : (tensor<2x3xi1>, tensor<2x3xi32>, tensor<2x3xi32>) -> tensor<2x3xi32>
  func.return %0 : tensor<2x3xi32>
}

// -----

// CHECK-LABEL: func @select_scalar_pred
func.func @select_scalar_pred(%arg0: tensor<i1>, %arg1: tensor<2x3xi32>, %arg2: tensor<2x3xi32>) -> tensor<2x3xi32> {
  %0 = "mhlo.select"(%arg0, %arg1, %arg2) : (tensor<i1>, tensor<2x3xi32>, tensor<2x3xi32>) -> tensor<2x3xi32>
  func.return %0 : tensor<2x3xi32>
}

// -----

// CHECK-LABEL: func @select_cast_compatible_types
func.func @select_cast_compatible_types(%arg0: tensor<i1>, %arg1: tensor<*xi32>, %arg2: tensor<2x3xi32>) -> tensor<*xi32> {
  %0 = "mhlo.select"(%arg0, %arg1, %arg2) : (tensor<i1>, tensor<*xi32>, tensor<2x3xi32>) -> tensor<*xi32>
  func.return %0 : tensor<*xi32>
}

// -----

// CHECK-LABEL: func @select_cast_compatible_types
func.func @select_cast_compatible_types(%arg0: tensor<i1>, %arg1: tensor<2x3xi32>, %arg2: tensor<*xi32>) -> tensor<*xi32> {
  %0 = "mhlo.select"(%arg0, %arg1, %arg2) : (tensor<i1>, tensor<2x3xi32>, tensor<*xi32>) -> tensor<*xi32>
  func.return %0 : tensor<*xi32>
}

// -----

// CHECK-LABEL: func @select_cast_compatible_types
func.func @select_cast_compatible_types(%arg0: tensor<i1>, %arg1: tensor<2x?xi32>, %arg2: tensor<?x3xi32>) -> tensor<?x?xi32> {
  %0 = "mhlo.select"(%arg0, %arg1, %arg2) : (tensor<i1>, tensor<2x?xi32>, tensor<?x3xi32>) -> tensor<?x?xi32>
  func.return %0 : tensor<?x?xi32>
}

// -----

// CHECK-LABEL: func @select_cast_compatible_types
func.func @select_cast_compatible_types(%arg0: tensor<i1>, %arg1: tensor<?x3xi32>, %arg2: tensor<2x?xi32>) -> tensor<?x?xi32> {
  %0 = "mhlo.select"(%arg0, %arg1, %arg2) : (tensor<i1>, tensor<?x3xi32>, tensor<2x?xi32>) -> tensor<?x?xi32>
  func.return %0 : tensor<?x?xi32>
}

// -----

// CHECK-LABEL: func @select_scalar_x_y
func.func @select_scalar_x_y(%arg0: tensor<i1>, %arg1: tensor<i32>, %arg2: tensor<i32>) -> tensor<i32> {
  %0 = "mhlo.select"(%arg0, %arg1, %arg2) : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
  func.return %0 : tensor<i32>
}

// -----

func.func @select_bad_pred_type(%arg0: tensor<i32>, %arg1: tensor<2x3xi32>, %arg2: tensor<2x3xi32>) -> tensor<2x3xi32> {
  // expected-error@+1 {{must be tensor of pred (AKA boolean or 1-bit integer) values}}
  %0 = "mhlo.select"(%arg0, %arg1, %arg2) : (tensor<i32>, tensor<2x3xi32>, tensor<2x3xi32>) -> tensor<2x3xi32>
  func.return %0 : tensor<2x3xi32>
}

// -----

func.func @select_bad_pred_shape(%arg0: tensor<3xi1>, %arg1: tensor<2x3xi32>, %arg2: tensor<2x3xi32>) -> tensor<2x3xi32> {
  // expected-error@+1 {{requires the same shape for all operands}}
  %0 = "mhlo.select"(%arg0, %arg1, %arg2) : (tensor<3xi1>, tensor<2x3xi32>, tensor<2x3xi32>) -> tensor<2x3xi32>
  func.return %0 : tensor<2x3xi32>
}

// -----

func.func @select_bad_shape_mismatch(%arg0: tensor<3xi1>, %arg1: tensor<2x4xi32>, %arg2: tensor<2x3xi32>) -> tensor<2x3xi32> {
  // expected-error@+1 {{requires compatible types for non-predicate operands}}
  %0 = "mhlo.select"(%arg0, %arg1, %arg2) : (tensor<3xi1>, tensor<2x4xi32>, tensor<2x3xi32>) -> tensor<2x3xi32>
  func.return %0 : tensor<2x3xi32>
}

// -----

func.func @select_when_pred_is_scalar(%arg0: tensor<i1>, %arg1: tensor<2x3xi32>, %arg2: tensor<2x3xi32>) -> tensor<2x3xi32> {
  %0 = "mhlo.select"(%arg0, %arg1, %arg2) : (tensor<i1>, tensor<2x3xi32>, tensor<2x3xi32>) -> tensor<2x3xi32>
  func.return %0 : tensor<2x3xi32>
}

// -----

func.func @select_element_type_mismatch(%arg0: tensor<i1>, %arg1: tensor<2x3xf32>, %arg2: tensor<2x3xi32>) -> tensor<2x3xi32> {
  // expected-error@+1 {{requires compatible types for non-predicate operands}}
  %0 = "mhlo.select"(%arg0, %arg1, %arg2) : (tensor<i1>, tensor<2x3xf32>, tensor<2x3xi32>) -> tensor<2x3xi32>
  func.return %0 : tensor<2x3xi32>
}

// -----

func.func @select_element_type_mismatch(%arg0: tensor<i1>, %arg1: tensor<2x3xf32>, %arg2: tensor<2x3xf64>) -> tensor<2x3xf64> {
  // expected-error@+1 {{requires compatible types for non-predicate operands}}
  %0 = "mhlo.select"(%arg0, %arg1, %arg2) : (tensor<i1>, tensor<2x3xf32>, tensor<2x3xf64>) -> tensor<2x3xf64>
  func.return %0 : tensor<2x3xf64>
}

// -----

// CHECK-LABEL: func @slice
func.func @slice(%arg0: tensor<3x4xi32>) -> tensor<1x2xi32> {
  %0 = "mhlo.slice"(%arg0) {start_indices = dense<[1, 0]> : tensor<2xi64>, limit_indices = dense<[2, 4]> : tensor<2xi64>, strides = dense<[1, 2]> : tensor<2xi64>} : (tensor<3x4xi32>) -> tensor<1x2xi32>
  func.return %0 : tensor<1x2xi32>
}

// -----

func.func @slice_c2(%arg0: tensor<3x4xi32>) -> tensor<1x2xi32> {
  // expected-error@+1 {{the number of elements in start_indices (3) does not match the rank of the operand (2)}}
  %0 = "mhlo.slice"(%arg0) {
    start_indices = dense<[1, 0, 0]> : tensor<3xi64>,
    limit_indices = dense<[2, 4, 0]> : tensor<3xi64>,
    strides = dense<[1, 2, 0]> : tensor<3xi64>
  } : (tensor<3x4xi32>) -> tensor<1x2xi32>
  func.return %0 : tensor<1x2xi32>
}

// -----

func.func @slice_c3(%arg0: tensor<3x4xi32>) -> tensor<1x2xi32> {
  // expected-error@+1 {{negative start index -1 in dimension 0}}
  %0 = "mhlo.slice"(%arg0) {
    start_indices = dense<[-1, 0]> : tensor<2xi64>,
    limit_indices = dense<[2, 4]> : tensor<2xi64>,
    strides = dense<[1, 2]> : tensor<2xi64>
  } : (tensor<3x4xi32>) -> tensor<1x2xi32>
  func.return %0 : tensor<1x2xi32>
}

// -----

func.func @slice_c3(%arg0: tensor<3x4xi32>) -> tensor<1x2xi32> {
  // expected-error@+1 {{limit index 5 is larger than dimension size 4 in dimension 1}}
  %0 = "mhlo.slice"(%arg0) {
    start_indices = dense<[1, 0]> : tensor<2xi64>,
    limit_indices = dense<[2, 5]> : tensor<2xi64>,
    strides = dense<[1, 2]> : tensor<2xi64>
  } : (tensor<3x4xi32>) -> tensor<1x2xi32>
  func.return %0 : tensor<1x2xi32>
}

// -----

func.func @slice_c3(%arg0: tensor<3x4xi32>) -> tensor<1x2xi32> {
  // expected-error@+1 {{start index 3 is larger than limit index 2 in dimension 1}}
  %0 = "mhlo.slice"(%arg0) {
    start_indices = dense<[1, 3]> : tensor<2xi64>,
    limit_indices = dense<[2, 2]> : tensor<2xi64>,
    strides = dense<[1, 2]> : tensor<2xi64>
  } : (tensor<3x4xi32>) -> tensor<1x2xi32>
  func.return %0 : tensor<1x2xi32>
}

// -----

func.func @slice_c4(%arg0: tensor<3x4xi32>) -> tensor<1x2xi32> {
  // expected-error@+1 {{stride must be positive but got 0 in dimension 0}}
  %0 = "mhlo.slice"(%arg0) {
    start_indices = dense<[1, 0]> : tensor<2xi64>,
    limit_indices = dense<[2, 4]> : tensor<2xi64>,
    strides = dense<[0, 2]> : tensor<2xi64>
  } : (tensor<3x4xi32>) -> tensor<1x2xi32>
  func.return %0 : tensor<1x2xi32>
}

// -----

// CHECK-LABEL: func @slice_dynamic_dim
func.func @slice_dynamic_dim(%arg0: tensor<3x?xi32>) -> tensor<1x?xi32> {
  %0 = "mhlo.slice"(%arg0) {
    start_indices = dense<[1, 1]> : tensor<2xi64>,
    limit_indices = dense<[2, 2]> : tensor<2xi64>,
    strides = dense<[1, 1]> : tensor<2xi64>
  } : (tensor<3x?xi32>) -> tensor<1x?xi32>
  func.return %0 : tensor<1x?xi32>
}

// -----

func.func @slice_i2(%arg0: tensor<3x4xi32>) -> tensor<1x2xi32> {
  // expected-error@+1 {{start_indices has rank 2 instead of required rank 1}}
  %0 = "mhlo.slice"(%arg0) {
    start_indices = dense<[[1, 0]]> : tensor<1x2xi64>,
    limit_indices = dense<[[2, 4]]> : tensor<1x2xi64>,
    strides = dense<[[1, 2]]> : tensor<1x2xi64>
  } : (tensor<3x4xi32>) -> tensor<1x2xi32>
  func.return %0 : tensor<1x2xi32>
}

// -----

// CHECK-LABEL: func @slice_unranked
func.func @slice_unranked(%arg0: tensor<*xi32>) -> tensor<*xi32> {
  %0 = "mhlo.slice"(%arg0) {start_indices = dense<[1, 0]> : tensor<2xi64>, limit_indices = dense<[2, 4]> : tensor<2xi64>, strides = dense<[1, 2]> : tensor<2xi64>} : (tensor<*xi32>) -> tensor<*xi32>
  func.return %0 : tensor<*xi32>
}

// -----

// CHECK-LABEL: func @dynamic_slice
func.func @dynamic_slice(%arg0: tensor<3x4xi32>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> tensor<1x4xi32> {
  %0 = "mhlo.dynamic_slice"(%arg0, %arg1, %arg2) {slice_sizes = dense<[1, 4]> : tensor<2xi64>} : (tensor<3x4xi32>, tensor<i64>, tensor<i64>) -> tensor<1x4xi32>
  func.return %0 : tensor<1x4xi32>
}

// -----

// CHECK-LABEL: func @dynamic_slice_dynamic_dim
func.func @dynamic_slice_dynamic_dim(%arg0: tensor<?x4xi32>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> tensor<1x4xi32> {
  %0 = "mhlo.dynamic_slice"(%arg0, %arg1, %arg2) {slice_sizes = dense<[1, 4]> : tensor<2xi64>} : (tensor<?x4xi32>, tensor<i64>, tensor<i64>) -> tensor<1x4xi32>
  func.return %0 : tensor<1x4xi32>
}

// -----

func.func @dynamic_slice_mismatch_indices(%arg0: tensor<3x4xi32>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> tensor<1x4xi32> {
  // expected-error@+1 {{has mismatched number of slice sizes (1) and number of start indices (2)}}
  %0 = "mhlo.dynamic_slice"(%arg0, %arg1, %arg2) {slice_sizes = dense<[4]> : tensor<1xi64>} : (tensor<3x4xi32>, tensor<i64>, tensor<i64>) -> tensor<1x4xi32>
  func.return %0 : tensor<1x4xi32>
}

// -----

func.func @dynamic_slice_mismatch_indices_element_type(%arg0: tensor<3x4xi32>, %arg1: tensor<i32>, %arg2: tensor<i64>) -> tensor<1x4xi32> {
  // expected-error@+1 {{start indices must have same element type}}
  %0 = "mhlo.dynamic_slice"(%arg0, %arg1, %arg2) {slice_sizes = dense<[1, 4]> : tensor<2xi64>} : (tensor<3x4xi32>, tensor<i32>, tensor<i64>) -> tensor<1x4xi32>
  func.return %0 : tensor<1x4xi32>
}

// -----

func.func @dynamic_slice(%arg0: tensor<3x4xi32>, %arg1: tensor<i64>) -> tensor<1x4xi32> {
  // expected-error@+1 {{has mismatched number of start indices (1) and the rank of operand (2)}}
  %0 = "mhlo.dynamic_slice"(%arg0, %arg1) {slice_sizes = dense<[1]> : tensor<1xi64>} : (tensor<3x4xi32>, tensor<i64>) -> tensor<1x4xi32>
  func.return %0 : tensor<1x4xi32>
}

// -----

func.func @dynamic_slice_mismatch_element_types(%arg0: tensor<3x4xi32>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> tensor<1x4xf32> {
  // expected-error@+1 {{failed to verify that all of {operand, result} have same element type}}
  %0 = "mhlo.dynamic_slice"(%arg0, %arg1, %arg2) {slice_sizes = dense<[1, 4]> : tensor<2xi64>} : (tensor<3x4xi32>, tensor<i64>, tensor<i64>) -> tensor<1x4xf32>
  func.return %0 : tensor<1x4xf32>
}

// -----

func.func @dynamic_slice_mismatch_return_shape(%arg0: tensor<3x4xi32>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> tensor<2x4xi32> {
  // expected-error@+1 {{inferred type(s) 'tensor<1x4xi32>' are incompatible with return type(s) of operation 'tensor<2x4xi32>'}}
  %0 = "mhlo.dynamic_slice"(%arg0, %arg1, %arg2) {slice_sizes = dense<[1, 4]> : tensor<2xi64>} : (tensor<3x4xi32>, tensor<i64>, tensor<i64>) -> tensor<2x4xi32>
  func.return %0 : tensor<2x4xi32>
}

// -----

func.func @dynamic_slice_start_not_0d(%arg0: tensor<3x4xi32>, %arg1: tensor<2xi64>) -> tensor<1x4xi32> {
  // expected-error@+1 {{operand #1 must be 0D tensor of 4/8/16/32/64-bit signless integer or 4/8/16/32/64-bit unsigned integer values, but got 'tensor<2xi64>'}}
  %0 = "mhlo.dynamic_slice"(%arg0, %arg1) {slice_sizes = dense<[1, 4]> : tensor<2xi64>} : (tensor<3x4xi32>, tensor<2xi64>) -> tensor<1x4xi32>
  func.return %0 : tensor<1x4xi32>
}

// -----

func.func @dynamic_slice_start_not_int(%arg0: tensor<3x4xi32>, %arg1: tensor<f64>, %arg2: tensor<f64>) -> tensor<1x4xi32> {
  // expected-error@+1 {{operand #1 must be 0D tensor of 4/8/16/32/64-bit signless integer or 4/8/16/32/64-bit unsigned integer values, but got 'tensor<f64>'}}
  %0 = "mhlo.dynamic_slice"(%arg0, %arg1, %arg2) {slice_sizes = dense<[1, 4]> : tensor<2xi64>} : (tensor<3x4xi32>, tensor<f64>, tensor<f64>) -> tensor<1x4xi32>
  func.return %0 : tensor<1x4xi32>
}

// -----

func.func @dynamic_slice_slice_size_negative(%arg0: tensor<3x4xi32>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> tensor<1x4xi32> {
  // expected-error@+1 {{has negative size index to dynamic slice: -1}}
  %0 = "mhlo.dynamic_slice"(%arg0, %arg1, %arg2) {slice_sizes = dense<[-1, 4]> : tensor<2xi64>} : (tensor<3x4xi32>, tensor<i64>, tensor<i64>) -> tensor<1x4xi32>
  func.return %0 : tensor<1x4xi32>
}

// -----

func.func @dynamic_slice_slice_size_too_large(%arg0: tensor<3x4xi32>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> tensor<1x4xi32> {
  // expected-error@+1 {{has slice size 10 greater than dimension size 4 in dimension 1 of operand}}
  %0 = "mhlo.dynamic_slice"(%arg0, %arg1, %arg2) {slice_sizes = dense<[1, 10]> : tensor<2xi64>} : (tensor<3x4xi32>, tensor<i64>, tensor<i64>) -> tensor<1x4xi32>
  func.return %0 : tensor<1x4xi32>
}

// -----

// CHECK-LABEL: @dynamic_update_slice
func.func @dynamic_update_slice(%operand: tensor<3x4xi64>, %update: tensor<1x4xi64>, %start_indices0: tensor<i64>, %start_indices1: tensor<i64>) -> tensor<3x4xi64> {
  %0 = "mhlo.dynamic_update_slice"(%operand, %update, %start_indices0, %start_indices1) : (tensor<3x4xi64>, tensor<1x4xi64>, tensor<i64>, tensor<i64>) -> tensor<3x4xi64>
  func.return %0 : tensor<3x4xi64>
}

// -----

func.func @dynamic_update_slice_c1(%operand: tensor<3x4xi64>, %update: tensor<1x4xi64>, %start_indices0: tensor<i64>, %start_indices1: tensor<i64>) -> tensor<3x5xi64> {
  // expected-error@+1 {{op inferred type(s) 'tensor<3x4xi64>' are incompatible with return type(s) of operation 'tensor<3x5xi64>'}}
  %0 = "mhlo.dynamic_update_slice"(%operand, %update, %start_indices0, %start_indices1) : (tensor<3x4xi64>, tensor<1x4xi64>, tensor<i64>, tensor<i64>) -> tensor<3x5xi64>
  func.return %0 : tensor<3x5xi64>
}

// -----

func.func @dynamic_update_slice_c3(%operand: tensor<3x4xi64>, %update: tensor<2xi64>, %start_indices0: tensor<i64>, %start_indices1: tensor<i64>) -> tensor<3x4xi64> {
  // expected-error@+1 {{update rank does not match operand rank: 1 vs 2.}}
  %0 = "mhlo.dynamic_update_slice"(%operand, %update, %start_indices0, %start_indices1) : (tensor<3x4xi64>, tensor<2xi64>, tensor<i64>, tensor<i64>) -> tensor<3x4xi64>
  func.return %0 : tensor<3x4xi64>
}

// -----

func.func @dynamic_update_slice_c4(%operand: tensor<3x4xi64>, %update: tensor<1x2xi64>, %start_indices0: tensor<i64>) -> tensor<3x4xi64> {
  // expected-error@+1 {{expects number of start_indices to match operand rank: 1 vs 2.}}
  %0 = "mhlo.dynamic_update_slice"(%operand, %update, %start_indices0) : (tensor<3x4xi64>, tensor<1x2xi64>, tensor<i64>) -> tensor<3x4xi64>
  func.return %0 : tensor<3x4xi64>
}

// -----

func.func @dynamic_update_slice_c5(%operand: tensor<11x3x4xi32>, %update: tensor<1x3x4xi32>, %start_indices0: tensor<i32>, %start_indices1: tensor<i64>, %start_indices2: tensor<i64>) -> tensor<11x3x4xi32> {
  // expected-error@+1 {{start indices must have same element type}}
  %0 = "mhlo.dynamic_update_slice"(%operand, %update, %start_indices0, %start_indices1, %start_indices2) : (tensor<11x3x4xi32>, tensor<1x3x4xi32>, tensor<i32>, tensor<i64>, tensor<i64>) -> tensor<11x3x4xi32>
  func.return %0 : tensor<11x3x4xi32>
}

// -----

func.func @dynamic_update_slice_c6(%operand: tensor<3x4xi64>, %update: tensor<1x5xi64>, %start_indices0: tensor<i64>, %start_indices1: tensor<i64>) -> tensor<3x4xi64> {
  // expected-error@+1 {{expects size at dimension 1 of update to be in range [0, 4]. Got: 5.}}
  %0 = "mhlo.dynamic_update_slice"(%operand, %update, %start_indices0, %start_indices1) : (tensor<3x4xi64>, tensor<1x5xi64>, tensor<i64>, tensor<i64>) -> tensor<3x4xi64>
  func.return %0 : tensor<3x4xi64>
}

// -----

// CHECK-LABEL: @dynamic_update_slice_dynamic_dim
func.func @dynamic_update_slice_dynamic_dim(%operand: tensor<?x4xi64>, %update: tensor<1x4xi64>, %start_indices0: tensor<i64>, %start_indices1: tensor<i64>) -> tensor<3x4xi64> {
  %0 = "mhlo.dynamic_update_slice"(%operand, %update, %start_indices0, %start_indices1) : (tensor<?x4xi64>, tensor<1x4xi64>, tensor<i64>, tensor<i64>) -> tensor<3x4xi64>
  func.return %0 : tensor<3x4xi64>
}

// -----

// CHECK-LABEL: func @dynamic_update_slice_dynamic_rank_operand
func.func @dynamic_update_slice_dynamic_rank_operand(%operand: tensor<*xi64>, %update: tensor<1x4xi64>, %start_indices0: tensor<i64>, %start_indices1: tensor<i64>) -> tensor<*xi64> {
  %0 = "mhlo.dynamic_update_slice"(%operand, %update, %start_indices0, %start_indices1) : (tensor<*xi64>, tensor<1x4xi64>, tensor<i64>, tensor<i64>) -> tensor<*xi64>
  func.return %0 : tensor<*xi64>
}

// -----

// CHECK-LABEL: func @dynamic_update_slice_dynamic_rank_update
func.func @dynamic_update_slice_dynamic_rank_update(%operand: tensor<3x4xi64>, %update: tensor<*xi64>, %start_indices0: tensor<i64>, %start_indices1: tensor<i64>) -> tensor<3x4xi64> {
  %0 = "mhlo.dynamic_update_slice"(%operand, %update, %start_indices0, %start_indices1) : (tensor<3x4xi64>, tensor<*xi64>, tensor<i64>, tensor<i64>) -> tensor<3x4xi64>
  func.return %0 : tensor<3x4xi64>
}

// -----

// CHECK-LABEL: func @dynamic_update_slice_dynamic_sizes
func.func @dynamic_update_slice_dynamic_sizes(%operand: tensor<?x4xi64>, %update: tensor<1x?xi64>, %start_indices0: tensor<i64>, %start_indices1: tensor<i64>) -> tensor<?x4xi64> {
  %0 = "mhlo.dynamic_update_slice"(%operand, %update, %start_indices0, %start_indices1) : (tensor<?x4xi64>, tensor<1x?xi64>, tensor<i64>, tensor<i64>) -> tensor<?x4xi64>
  func.return %0 : tensor<?x4xi64>
}

// -----

// CHECK-LABEL: func @transpose
func.func @transpose(%arg0: tensor<1x2x3x4xi32>) -> tensor<2x1x4x3xi32> {
  %0 = "mhlo.transpose"(%arg0) {permutation = dense<[1, 0, 3, 2]> : tensor<4xi64>} : (tensor<1x2x3x4xi32>) -> tensor<2x1x4x3xi32>
  func.return %0: tensor<2x1x4x3xi32>
}

// -----

func.func @transpose_ranked(%arg0: tensor<?x?x?x?xi32>) ->  tensor<?x?x?x?xi32> {
  %0 = "mhlo.transpose"(%arg0) {permutation = dense<[1, 0, 3, 2]> : tensor<4xi64>} : (tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32>
  func.return %0: tensor<?x?x?x?xi32>
}

// -----

func.func @transpose_unranked(%arg0: tensor<*xi32>) ->  tensor<*xi32> {
  %0 = "mhlo.transpose"(%arg0) {permutation = dense<[1, 0, 3, 2]> : tensor<4xi64>} : (tensor<*xi32>) -> tensor<*xi32>
  func.return %0: tensor<*xi32>
}

// -----

func.func @transpose_missing_permutation(%arg0: tensor<1x2x3x4xi32>) -> tensor<2x1x4x3xi32> {
  // expected-error@+1 {{requires attribute 'permutation'}}
  %0 = "mhlo.transpose"(%arg0) {} : (tensor<1x2x3x4xi32>) -> tensor<2x1x4x3xi32>
  func.return %0: tensor<2x1x4x3xi32>
}

// -----

func.func @transpose_bad_permutations_rank(%arg0: tensor<1x2x3x4xi32>) ->  tensor<2x1x4x3xi32> {
  // expected-error@+1 {{permutation has rank 2 instead of rank 1}}
  %0 = "mhlo.transpose"(%arg0) {permutation = dense<[[1]]> : tensor<1x1xi64>} : (tensor<1x2x3x4xi32>) -> tensor<2x1x4x3xi32>
  func.return %0: tensor<2x1x4x3xi32>
}

// -----

func.func @transpose_bad_permutations_size(%arg0: tensor<1x2x3x4xi32>) ->  tensor<2x1x4x3xi32> {
  // expected-error@+1 {{TransposeOp operand rank 4 does not match permutation size 1}}
  %0 = "mhlo.transpose"(%arg0) {permutation = dense<[1]> : tensor<1xi64>} : (tensor<1x2x3x4xi32>) -> tensor<2x1x4x3xi32>
  func.return %0: tensor<2x1x4x3xi32>
}

// -----

func.func @transpose_bad_permutation(%arg0: tensor<1x2x3x4xi32>) ->  tensor<2x1x4x3xi32> {
  // expected-error@+1 {{attribute permutation must be a permutation of [0, 1, 2, 3] but got dense<[1, 0, 3, 9]> : tensor<4xi64>}}
  %0 = "mhlo.transpose"(%arg0) {permutation = dense<[1, 0, 3, 9]> : tensor<4xi64>} : (tensor<1x2x3x4xi32>) -> tensor<2x1x4x3xi32>
  func.return %0: tensor<2x1x4x3xi32>
}

// -----

func.func @transpose_operand_result_rank_mismatch(%arg0: tensor<1x2x3x4xi32>) ->  tensor<2xi32> {
  // expected-error@+1 {{op inferred type(s) 'tensor<2x1x4x3xi32>' are incompatible with return type(s) of operation 'tensor<2xi32>'}}
  %0 = "mhlo.transpose"(%arg0) {permutation = dense<[1, 0, 3, 2]> : tensor<4xi64>} : (tensor<1x2x3x4xi32>) -> tensor<2xi32>
  func.return %0: tensor<2xi32>
}

// -----

func.func @transpose_operand_result_permutation_mismatch(%arg0: tensor<1x?x3x?xi32>) ->  tensor<?x2x?x?xi32> {
  // expected-error@+1 {{op inferred type(s) 'tensor<?x1x?x3xi32>' are incompatible with return type(s) of operation 'tensor<?x2x?x?xi32>}}
  %0 = "mhlo.transpose"(%arg0) {permutation = dense<[1, 0, 3, 2]> : tensor<4xi64>} : (tensor<1x?x3x?xi32>) -> tensor<?x2x?x?xi32>
  func.return %0: tensor<?x2x?x?xi32>
}

// -----

// CHECK-LABEL: func @triangular_solve
func.func @triangular_solve(%arg0: tensor<10x5x4x4xf32>, %arg1: tensor<10x5x4x4xf32>) -> tensor<10x5x4x4xf32> {
  %0 = "mhlo.triangular_solve"(%arg0, %arg1) {left_side = true, lower = true, transpose_a = #mhlo<transpose NO_TRANSPOSE>, unit_diagonal = true} : (tensor<10x5x4x4xf32>, tensor<10x5x4x4xf32>) -> tensor<10x5x4x4xf32>
  func.return %0 : tensor<10x5x4x4xf32>
}

// -----

// CHECK-LABEL: func @triangular_solve_dynamic_dims_minor
func.func @triangular_solve_dynamic_dims_minor(%arg0: tensor<10x5x?x4xf32>, %arg1: tensor<10x5x4x4xf32>) -> tensor<10x5x4x4xf32> {
  %0 = "mhlo.triangular_solve"(%arg0, %arg1) {left_side = true, lower = true, transpose_a = #mhlo<transpose NO_TRANSPOSE>, unit_diagonal = true} : (tensor<10x5x?x4xf32>, tensor<10x5x4x4xf32>) -> tensor<10x5x4x4xf32>
  func.return %0 : tensor<10x5x4x4xf32>
}

// -----

// CHECK-LABEL: func @triangular_solve_dynamic_dims_shared
func.func @triangular_solve_dynamic_dims_shared(%arg0: tensor<10x5x4x?xf32>, %arg1: tensor<10x5x4x4xf32>) -> tensor<10x5x4x4xf32> {
  %0 = "mhlo.triangular_solve"(%arg0, %arg1) {left_side = true, lower = true, transpose_a = #mhlo<transpose NO_TRANSPOSE>, unit_diagonal = true} : (tensor<10x5x4x?xf32>, tensor<10x5x4x4xf32>) -> tensor<10x5x4x4xf32>
  func.return %0 : tensor<10x5x4x4xf32>
}

// -----

// CHECK-LABEL: func @triangular_solve_dynamic_dims_batch
func.func @triangular_solve_dynamic_dims_batch(%arg0: tensor<?x5x4x4xf32>, %arg1: tensor<10x?x4x4xf32>) -> tensor<10x5x4x4xf32> {
  %0 = "mhlo.triangular_solve"(%arg0, %arg1) {left_side = true, lower = true, transpose_a = #mhlo<transpose NO_TRANSPOSE>, unit_diagonal = true} : (tensor<?x5x4x4xf32>, tensor<10x?x4x4xf32>) -> tensor<10x5x4x4xf32>
  func.return %0 : tensor<10x5x4x4xf32>
}

// -----

// CHECK-LABEL: func @triangular_solve_unranked
func.func @triangular_solve_unranked(%arg0: tensor<*xf32>, %arg1: tensor<*xf32>) -> tensor<*xf32> {
  %0 = "mhlo.triangular_solve"(%arg0, %arg1) {left_side = true, lower = true, transpose_a = #mhlo<transpose NO_TRANSPOSE>, unit_diagonal = true} : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// -----

// CHECK-LABEL: func @triangular_solve_a_is_unranked
func.func @triangular_solve_a_is_unranked(%arg0: tensor<*xf32>, %arg1: tensor<4x4xf32>) -> tensor<*xf32> {
  %0 = "mhlo.triangular_solve"(%arg0, %arg1) {left_side = true, lower = true, transpose_a = #mhlo<transpose NO_TRANSPOSE>, unit_diagonal = true} : (tensor<*xf32>, tensor<4x4xf32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// -----

// CHECK-LABEL: func @triangular_solve_b_is_unranked
func.func @triangular_solve_b_is_unranked(%arg0: tensor<4x4xf32>, %arg1: tensor<*xf32>) -> tensor<*xf32> {
  %0 = "mhlo.triangular_solve"(%arg0, %arg1) {left_side = true, lower = true, transpose_a = #mhlo<transpose NO_TRANSPOSE>, unit_diagonal = true} : (tensor<4x4xf32>, tensor<*xf32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// -----

func.func @triangular_solve_rank_less_than_2(%arg0: tensor<4xf32>, %arg1: tensor<4x3xf32>) -> tensor<4x3xf32> {
  // expected-error@+1 {{operand 'a' must have rank >= 2, but got 'tensor<4xf32>'}}
  %0 = "mhlo.triangular_solve"(%arg0, %arg1) {left_side = true, lower = true, transpose_a = #mhlo<transpose NO_TRANSPOSE>, unit_diagonal = true} : (tensor<4xf32>, tensor<4x3xf32>) -> tensor<4x3xf32>
  func.return %0 : tensor<4x3xf32>
}

// -----

func.func @triangular_solve_unequal_minor_dims_a(%arg0: tensor<4x3xf32>, %arg1: tensor<4x3xf32>) -> tensor<4x3xf32> {
  // expected-error@+1 {{two minor dimensions of operand 'a' must be compatible, but got 'tensor<4x3xf32>'}}
  %0 = "mhlo.triangular_solve"(%arg0, %arg1) {left_side = true, lower = true, transpose_a = #mhlo<transpose NO_TRANSPOSE>, unit_diagonal = true} : (tensor<4x3xf32>, tensor<4x3xf32>) -> tensor<4x3xf32>
  func.return %0 : tensor<4x3xf32>
}

// -----

func.func @triangular_solve_unequal_rank(%arg0: tensor<10x4x4xf32>, %arg1: tensor<4x3xf32>) -> tensor<4x3xf32> {
  // expected-error@+1 {{operands must have equal rank, but got 'tensor<10x4x4xf32>' and 'tensor<4x3xf32>'}}
  %0 = "mhlo.triangular_solve"(%arg0, %arg1) {left_side = true, lower = true, transpose_a = #mhlo<transpose NO_TRANSPOSE>, unit_diagonal = true} : (tensor<10x4x4xf32>, tensor<4x3xf32>) -> tensor<4x3xf32>
  func.return %0 : tensor<4x3xf32>
}

// -----

func.func @triangular_solve_mismatch_shared_dim(%arg0: tensor<4x4xf32>, %arg1: tensor<3x4xf32>) -> tensor<3x4xf32> {
  // expected-error@+1 {{shared dimension of operands 'a' and 'b' must be compatible, but got 'tensor<4x4xf32>' and 'tensor<3x4xf32>'}}
  %0 = "mhlo.triangular_solve"(%arg0, %arg1) {left_side = true, lower = true, transpose_a = #mhlo<transpose NO_TRANSPOSE>, unit_diagonal = true} : (tensor<4x4xf32>, tensor<3x4xf32>) -> tensor<3x4xf32>
  func.return %0 : tensor<3x4xf32>
}

// -----

func.func @triangular_solve_mismatch_leading_dims(%arg0: tensor<10x5x4x4xf32>, %arg1: tensor<10x6x4x3xf32>) -> tensor<10x6x4x3xf32> {
  // expected-error@+1 {{batch dimensions of the operands must be compatible, but got 'tensor<10x5x4x4xf32>' and 'tensor<10x6x4x3xf32>'}}
  %0 = "mhlo.triangular_solve"(%arg0, %arg1) {left_side = true, lower = true, transpose_a = #mhlo<transpose NO_TRANSPOSE>, unit_diagonal = true} : (tensor<10x5x4x4xf32>, tensor<10x6x4x3xf32>) -> tensor<10x6x4x3xf32>
  func.return %0 : tensor<10x6x4x3xf32>
}

// -----

func.func @triangular_solve_mismatch_result_and_b_type(%arg0: tensor<4x4xf32>, %arg1: tensor<4x3xf32>) -> tensor<4x4xf32> {
  // expected-error@+1 {{inferred type(s) 'tensor<4x3xf32>' are incompatible with return type(s) of operation 'tensor<4x4xf32>'}}
  %0 = "mhlo.triangular_solve"(%arg0, %arg1) {left_side = true, lower = true, transpose_a = #mhlo<transpose NO_TRANSPOSE>, unit_diagonal = true} : (tensor<4x4xf32>, tensor<4x3xf32>) -> tensor<4x4xf32>
  func.return %0 : tensor<4x4xf32>
}

// -----

func.func @triangular_solve(%arg0: tensor<10x5x4x4xf32>, %arg1: tensor<10x5x4x4xf32>) -> tensor<10x5x4x4xf32> {
  // expected-error@+1 {{Invalid transpose option value for triangular solve}}
  %0 = "mhlo.triangular_solve"(%arg0, %arg1) {left_side = true, lower = true, transpose_a = #mhlo<transpose TRANSPOSE_INVALID>, unit_diagonal = true} : (tensor<10x5x4x4xf32>, tensor<10x5x4x4xf32>) -> tensor<10x5x4x4xf32>
  func.return %0 : tensor<10x5x4x4xf32>
}

// -----

// CHECK-LABEL: func @tuple
func.func @tuple(%arg0: tensor<1xi32>, %arg1: tensor<1x2xf32>) -> tuple<tensor<1xi32>, tensor<1x2xf32>> {
  %0 = "mhlo.tuple"(%arg0, %arg1) : (tensor<1xi32>, tensor<1x2xf32>) -> tuple<tensor<1xi32>, tensor<1x2xf32>>
  func.return %0: tuple<tensor<1xi32>, tensor<1x2xf32>>
}

// -----

func.func @tuple_token(%arg0: tensor<f32>, %arg1: !mhlo.token) -> tuple<tensor<f32>, !mhlo.token> {
  %0 = "mhlo.tuple"(%arg0, %arg1) : (tensor<f32>, !mhlo.token) -> tuple<tensor<f32>, !mhlo.token>
  func.return %0 : tuple<tensor<f32>, !mhlo.token>
}

// -----

func.func @tuple_arg_size_mismatch(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tuple<tensor<f32>, tensor<f32>, tensor<f32>> {
  // expected-error@+1 {{inferred type(s) 'tuple<tensor<f32>, tensor<f32>>' are incompatible with return type(s) of operation 'tuple<tensor<f32>, tensor<f32>, tensor<f32>>'}}
  %0 = "mhlo.tuple"(%arg0, %arg1) : (tensor<f32>, tensor<f32>) -> tuple<tensor<f32>, tensor<f32>, tensor<f32>>
  func.return %0 : tuple<tensor<f32>, tensor<f32>, tensor<f32>>
}

// -----

func.func @tuple_type_mismatch(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tuple<tensor<f32>, tensor<i32>> {
  // expected-error@+1 {{inferred type(s) 'tuple<tensor<f32>, tensor<f32>>' are incompatible with return type(s) of operation 'tuple<tensor<f32>, tensor<i32>>'}}
  %0 = "mhlo.tuple"(%arg0, %arg1) : (tensor<f32>, tensor<f32>) -> tuple<tensor<f32>, tensor<i32>>
  func.return %0 : tuple<tensor<f32>, tensor<i32>>
}

// -----

func.func @get_tuple_element(%arg0: tuple<tensor<f32>, tensor<i32>>) -> tensor<f32> {
  %0 = "mhlo.get_tuple_element"(%arg0) {index = 0 : i32} : (tuple<tensor<f32>, tensor<i32>>) -> tensor<f32>
  func.return %0 : tensor<f32>
}

// -----

func.func @get_tuple_element_token(%arg0: tuple<tensor<f32>, !mhlo.token>) -> !mhlo.token {
  %0 = "mhlo.get_tuple_element"(%arg0) {index = 1 : i32} : (tuple<tensor<f32>, !mhlo.token>) -> !mhlo.token
  func.return %0 : !mhlo.token
}

// -----

func.func @get_tuple_element_bad_type(%arg0: tuple<tensor<f32>, tensor<i32>>) -> tensor<i32> {
  // expected-error@+1 {{inferred type(s) 'tensor<f32>' are incompatible with return type(s) of operation 'tensor<i32>'}}
  %0 = "mhlo.get_tuple_element"(%arg0) {index = 0 : i32} : (tuple<tensor<f32>, tensor<i32>>) -> tensor<i32>
  func.return %0 : tensor<i32>
}

// -----

func.func @get_tuple_element_index_out_of_bounds(%arg0: tuple<tensor<f32>, tensor<i32>>) -> tensor<f32> {
  // expected-error@+1 {{index 2 is out of bounds of operand with size 2}}
  %0 = "mhlo.get_tuple_element"(%arg0) {index = 2 : i32} : (tuple<tensor<f32>, tensor<i32>>) -> tensor<f32>
  func.return %0 : tensor<f32>
}

// -----

// CHECK-LABEL: func @and_i32_type
func.func @and_i32_type(%arg0: tensor<4xi32>, %arg1: tensor<4xi32>) -> tensor<4xi32> {
  %0 = "mhlo.and"(%arg0, %arg1) : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
  func.return %0 : tensor<4xi32>
}

// -----
// CHECK-LABEL: func @or_i1_type
func.func @or_i1_type(%arg0: tensor<4xi1>, %arg1: tensor<4xi1>) -> tensor<4xi1> {
  %0 = "mhlo.or"(%arg0, %arg1) : (tensor<4xi1>, tensor<4xi1>) -> tensor<4xi1>
  func.return %0 : tensor<4xi1>
}

// -----

func.func @or_invalid_f32_type(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  // expected-error@+1 {{but got 'tensor<4xf32>'}}
  %0 = "mhlo.or"(%arg0, %arg1) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  func.return %0 : tensor<4xf32>
}

// -----

func.func @floor_invalid_i32_type(%arg0: tensor<4xi32>) -> tensor<4xi32> {
  // expected-error@+1 {{must be tensor of f8E4M3FN type or f8E5M2 type or 16-bit float or 32-bit float or 64-bit float or bfloat16 type values, but got 'tensor<4xi32>'}}
  %0 = "mhlo.floor"(%arg0) : (tensor<4xi32>) -> tensor<4xi32>
  func.return %0 : tensor<4xi32>
}

// -----

// Verifiers HLO constant op custom printing and parsing.
// CHECK-LABEL: func @constants
func.func @constants() -> () {
  // CHECK: mhlo.constant dense<0> : tensor<i32>
  %0 = "mhlo.constant"() {value = dense<0> : tensor<i32>} : () -> (tensor<i32>)

  // CHECK: mhlo.constant {extra_attr = 3 : i32} dense<0> : tensor<i32>
  %1 = "mhlo.constant"() {extra_attr = 3 : i32, value = dense<0> : tensor<i32>} : () -> (tensor<i32>)
  func.return
}

// -----

func.func @constant_invalid() -> () {
  // expected-error@+1 {{'mhlo.constant' op inferred type(s) 'tensor<i32>' are incompatible with return type(s) of operation 'tensor<3xi32>'}}
  %0 = "mhlo.constant"() {value = dense<0> : tensor<i32>} : () -> (tensor<3xi32>)
  func.return
}

// -----

func.func @constant_invalid() -> () {
  // expected-error@+1 {{op result #0 must be statically shaped tensor}}
  %0 = "mhlo.constant"() {value = dense<1> : tensor<i32>} : () -> tensor<?xi32>
  func.return
}

// -----

func.func @constant_invalid() -> () {
  // expected-error@+1 {{elements literal type must have static shape}}
  %0 = "mhlo.constant"() {value = dense<1> : tensor<?xi32>} : () -> tensor<?xi32>
  func.return
}

// -----

// CHECK-LABEL: func @sort
func.func @sort(%input0: tensor<16x16xf32>, %input1: tensor<16x16xi32>) {
  %0:2 = "mhlo.sort"(%input0, %input1) ({
  ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<i32>, %arg3: tensor<i32>):
    %7 = "mhlo.compare"(%arg0, %arg1) {comparison_direction = #mhlo<comparison_direction GT>} : (tensor<f32>, tensor<f32>) -> tensor<i1>
    "mhlo.return"(%7) : (tensor<i1>) -> ()
  }) {dimension = 1 : i64, is_stable = true} : (tensor<16x16xf32>, tensor<16x16xi32>) -> (tensor<16x16xf32>, tensor<16x16xi32>)
  func.return
}

// -----

func.func @sort_no_operands() {
  // expected-error @+1 {{expected named operation to have at least 1 result}}
  %0:0 = "mhlo.sort"() ({
  ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>, %arg3: tensor<i32>, %arg4: tensor<i32>):
    %7 = "mhlo.compare"(%arg1, %arg2) {comparison_direction = #mhlo<comparison_direction GT>} : (tensor<f32>, tensor<f32>) -> tensor<i1>
    "mhlo.return"(%7) : (tensor<i1>) -> ()
  }) {dimension = 1 : i64, is_stable = true} : () -> ()
  func.return
}

// -----

// CHECK-LABEL: func @sort_unknown_rank
func.func @sort_unknown_rank(%input0: tensor<*xf32>, %input1: tensor<16x16xi32>) {
  %0:2 = "mhlo.sort"(%input0, %input1) ({
  ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<i32>, %arg3: tensor<i32>):
    %7 = "mhlo.compare"(%arg0, %arg1) {comparison_direction = #mhlo<comparison_direction GT>} : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %8 = "mhlo.select"(%7, %7, %7) : (tensor<i1>, tensor<i1>, tensor<i1>) -> tensor<*xi1>
    "mhlo.return"(%8) : (tensor<*xi1>) -> ()
  }) {dimension = 1 : i64, is_stable = true} : (tensor<*xf32>, tensor<16x16xi32>) -> (tensor<16x16xf32>, tensor<16x16xi32>)
  func.return
}

// -----

func.func @sort_dynamism(%input0: tensor<?x16xf32>, %input1: tensor<16x16xi32>) {
  %0:2 = "mhlo.sort"(%input0, %input1) ({
  ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<i32>, %arg3: tensor<i32>):
    %7 = "mhlo.compare"(%arg0, %arg1) {comparison_direction = #mhlo<comparison_direction GT>} : (tensor<f32>, tensor<f32>) -> tensor<i1>
    "mhlo.return"(%7) : (tensor<i1>) -> ()
  }) {dimension = 1 : i64, is_stable = true} : (tensor<?x16xf32>, tensor<16x16xi32>) -> (tensor<16x16xf32>, tensor<16x16xi32>)
  func.return
}

// -----

func.func @sort_unknown_rank(%input0: tensor<*xf32>, %input1: tensor<16x16xi32>) {
  // expected-error @+1 {{comparator block argument #0 should be of type 'tensor<f32>' but got 'tensor<i32>'}}
  %0:2 = "mhlo.sort"(%input0, %input1) ({
  ^bb0(%arg0: tensor<i32>, %arg1: tensor<i32>, %arg2: tensor<i32>, %arg3: tensor<i32>):
    %7 = "mhlo.compare"(%arg0, %arg1) {comparison_direction = #mhlo<comparison_direction GT>} : (tensor<i32>, tensor<i32>) -> tensor<i1>
    "mhlo.return"(%7) : (tensor<i1>) -> ()
  }) {dimension = 1 : i64, is_stable = true} : (tensor<*xf32>, tensor<16x16xi32>) -> (tensor<16x16xf32>, tensor<16x16xi32>)
  func.return
}

// -----

func.func @sort_different_dims(%input0: tensor<16x8xf32>, %input1: tensor<16x16xi32>) {
  // expected-error @+1 {{op requires the same shape for all operands and results}}
  %0:2 = "mhlo.sort"(%input0, %input1) ({
  ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<i32>, %arg3: tensor<i32>):
    %7 = "mhlo.compare"(%arg0, %arg1) {comparison_direction = #mhlo<comparison_direction GT>} : (tensor<f32>, tensor<f32>) -> tensor<i1>
    "mhlo.return"(%7) : (tensor<i1>) -> ()
  }) {dimension = 1 : i64, is_stable = true} : (tensor<16x8xf32>, tensor<16x16xi32>) -> (tensor<16x16xf32>, tensor<16x16xi32>)
  func.return
}

// -----

func.func @sort_dim_out_of_range(%input0: tensor<16x16xf32>, %input1: tensor<16x16xi32>) {
  // expected-error @+1 {{dimension attribute value must be in range [-2, 2), but found 10}}
  %0:2 = "mhlo.sort"(%input0, %input1) ({
  ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<i32>, %arg3: tensor<i32>):
    %7 = "mhlo.compare"(%arg0, %arg1) {comparison_direction = #mhlo<comparison_direction GT>} : (tensor<f32>, tensor<f32>) -> tensor<i1>
    "mhlo.return"(%7) : (tensor<i1>) -> ()
  }) {dimension = 10 : i64, is_stable = true} : (tensor<16x16xf32>, tensor<16x16xi32>) -> (tensor<16x16xf32>, tensor<16x16xi32>)
  func.return
}

// -----

func.func @sort_dim_out_of_range(%input0: tensor<16x16xf32>, %input1: tensor<16x16xi32>) {
  // expected-error @+1 {{dimension attribute value must be in range [-2, 2), but found -3}}
  %0:2 = "mhlo.sort"(%input0, %input1) ({
  ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<i32>, %arg3: tensor<i32>):
    %7 = "mhlo.compare"(%arg0, %arg1) {comparison_direction = #mhlo<comparison_direction GT>} : (tensor<f32>, tensor<f32>) -> tensor<i1>
    "mhlo.return"(%7) : (tensor<i1>) -> ()
  }) {dimension = -3 : i64, is_stable = true} : (tensor<16x16xf32>, tensor<16x16xi32>) -> (tensor<16x16xf32>, tensor<16x16xi32>)
  func.return
}

// -----

func.func @sort_wrong_block_arg_count(%input0: tensor<16x16xf32>, %input1: tensor<16x16xi32>) {
  // expected-error @+1 {{comparator block should have 4 arguments}}
  %0:2 = "mhlo.sort"(%input0, %input1) ({
  ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
    %7 = "mhlo.compare"(%arg0, %arg1) {comparison_direction = #mhlo<comparison_direction GT>} : (tensor<f32>, tensor<f32>) -> tensor<i1>
    "mhlo.return"(%7) : (tensor<i1>) -> ()
  }) {dimension = 1 : i64, is_stable = true} : (tensor<16x16xf32>, tensor<16x16xi32>) -> (tensor<16x16xf32>, tensor<16x16xi32>)
  func.return
}

// -----

func.func @sort_wrong_block_arg_type(%input0: tensor<16x16xf32>, %input1: tensor<16x16xi32>) {
  // expected-error @+1 {{comparator block argument #3 should be of type 'tensor<i32>' but got 'tensor<f32>'}}
  %0:2 = "mhlo.sort"(%input0, %input1) ({
  ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<i32>, %arg3: tensor<f32>):
    %7 = "mhlo.compare"(%arg0, %arg1) {comparison_direction = #mhlo<comparison_direction GT>} : (tensor<f32>, tensor<f32>) -> tensor<i1>
    "mhlo.return"(%7) : (tensor<i1>) -> ()
  }) {dimension = 1 : i64, is_stable = true} : (tensor<16x16xf32>, tensor<16x16xi32>) -> (tensor<16x16xf32>, tensor<16x16xi32>)
  func.return
}

// -----

func.func @sort_invalid_comparator_return_type(%input0: tensor<16x16xf32>, %input1: tensor<16x16xi32>) {
  // expected-error @+1 {{comparator must return tensor<i1> but got 'tensor<3xi64>'}}
  %0:2 = "mhlo.sort"(%input0, %input1) ({
  ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<i32>, %arg3: tensor<i32>):
    %cst = mhlo.constant dense<[2, 3, 5]> : tensor<3xi64>
    "mhlo.return"(%cst) : (tensor<3xi64>) -> ()
  }) {dimension = 1 : i64, is_stable = true} : (tensor<16x16xf32>, tensor<16x16xi32>) -> (tensor<16x16xf32>, tensor<16x16xi32>)
  func.return
}

// -----

func.func @sort_invalid_comparator_return_type(%input0: tensor<16x16xf32>, %input1: tensor<16x16xi32>) {
  // expected-error @+1 {{comparator must return tensor<i1> but got 'tensor<i32>'}}
  %0:2 = "mhlo.sort"(%input0, %input1) ({
  ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<i32>, %arg3: tensor<i32>):
    "mhlo.return"(%arg2) : (tensor<i32>) -> ()
  }) {dimension = 1 : i64, is_stable = true} : (tensor<16x16xf32>, tensor<16x16xi32>) -> (tensor<16x16xf32>, tensor<16x16xi32>)
  func.return
}

// -----

func.func @sort_invalid_comparator_return_type(%input0: tensor<16x16xf32>, %input1: tensor<16x16xi32>) {
  // expected-error @+1 {{comparator must return single output but got 2}}
  %0:2 = "mhlo.sort"(%input0, %input1) ({
  ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<i32>, %arg3: tensor<i32>):
    %7 = "mhlo.compare"(%arg0, %arg1) {compare_type = #mhlo<comparison_type FLOAT>, comparison_direction = #mhlo<comparison_direction GT>} : (tensor<f32>, tensor<f32>) -> tensor<i1>
    "mhlo.return"(%7, %7) : (tensor<i1>, tensor<i1>) -> ()
  }) {dimension = 1 : i64, is_stable = true} : (tensor<16x16xf32>, tensor<16x16xi32>) -> (tensor<16x16xf32>, tensor<16x16xi32>)
  func.return
}

// -----

func.func @sort_invalid_return_types(%input0: tensor<16x16xf32>, %input1: tensor<16x16xi32>) {
  // expected-error @+1 {{op inferred type(s) 'tensor<16x16xf32>', 'tensor<16x16xi32>' are incompatible with return type(s) of operation 'tensor<16x16xf32>'}}
  %0 = "mhlo.sort"(%input0, %input1) ({
  ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<i32>, %arg3: tensor<i32>):
    %7 = "mhlo.compare"(%arg0, %arg1) {compare_type = #mhlo<comparison_type FLOAT>, comparison_direction = #mhlo<comparison_direction GT>} : (tensor<f32>, tensor<f32>) -> tensor<i1>
    "mhlo.return"(%7) : (tensor<i1>) -> ()
  }) {dimension = 1 : i64, is_stable = true} : (tensor<16x16xf32>, tensor<16x16xi32>) -> (tensor<16x16xf32>)
  func.return
}

// -----

func.func @sort_invalid_return_types(%input0: tensor<16x16xf32>, %input1: tensor<16x16xi32>) {
  // expected-error @+1 {{inferred type(s) 'tensor<16x16xf32>', 'tensor<16x16xi32>' are incompatible with return type(s) of operation 'tensor<16x16xf32>', 'tensor<16x16xf32>'}}
  %0:2 = "mhlo.sort"(%input0, %input1) ({
  ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<i32>, %arg3: tensor<i32>):
    %7 = "mhlo.compare"(%arg0, %arg1) {compare_type = #mhlo<comparison_type FLOAT>, comparison_direction = #mhlo<comparison_direction GT>} : (tensor<f32>, tensor<f32>) -> tensor<i1>
    "mhlo.return"(%7) : (tensor<i1>) -> ()
  }) {dimension = 1 : i64, is_stable = true} : (tensor<16x16xf32>, tensor<16x16xi32>) -> (tensor<16x16xf32>, tensor<16x16xf32>)
  func.return
}

// -----

func.func @reshape_invalid_shapes(%operand: tensor<2x4xf32>) -> tensor<3x3xf32> {
  // expected-error @+1 {{number of output elements (9) doesn't match expected number of elements (8)}}
  %0 = "mhlo.reshape"(%operand) : (tensor<2x4xf32>) -> tensor<3x3xf32>
  func.return %0 : tensor<3x3xf32>
}

// -----

// CHECK-LABEL: func @reverse
func.func @reverse(%operand: tensor<3x2xi32>) -> tensor<3x2xi32> {
  %0 = "mhlo.reverse"(%operand) {
    dimensions = dense<[0, 1]> : tensor<2xi64>
  } : (tensor<3x2xi32>) -> tensor<3x2xi32>
  func.return %0 : tensor<3x2xi32>
}

// -----

func.func @reverse_c2(%operand: tensor<3x2xi32>) -> tensor<3x2xi32> {
  // expected-error @+1 {{dimensions should be unique. Got: 0, 0}}
  %0 = "mhlo.reverse"(%operand) {
    dimensions = dense<[0, 0]> : tensor<2xi64>
  } : (tensor<3x2xi32>) -> tensor<3x2xi32>
  func.return %0 : tensor<3x2xi32>
}

// -----

func.func @reverse_c3(%operand: tensor<*xi32>) -> tensor<*xi32> {
  // expected-error @+1 {{all dimensions should be non-negative. Got dimension: -1.}}
  %0 = "mhlo.reverse"(%operand) {
    dimensions = dense<-1> : tensor<1xi64>
  } : (tensor<*xi32>) -> tensor<*xi32>
  func.return %0 : tensor<*xi32>
}

// -----

func.func @reverse_c3(%operand: tensor<3x2xi32>) -> tensor<3x2xi32> {
  // expected-error @+1 {{all dimensions should be non-negative. Got dimension: -1.}}
  %0 = "mhlo.reverse"(%operand) {
    dimensions = dense<-1> : tensor<1xi64>
  } : (tensor<3x2xi32>) -> tensor<3x2xi32>
  func.return %0 : tensor<3x2xi32>
}

// -----

func.func @reverse_c3(%operand: tensor<3x2xi32>) -> tensor<3x2xi32> {
  // expected-error @+1 {{all dimensions should be between [0, 2). Got dimension: 2.}}
  %0 = "mhlo.reverse"(%operand) {
    dimensions = dense<2> : tensor<1xi64>
  } : (tensor<3x2xi32>) -> tensor<3x2xi32>
  func.return %0 : tensor<3x2xi32>
}

// -----

func.func @reverse_i2(%operand: tensor<3x2xi32>) -> tensor<3x2xi32> {
  // expected-error @+1 {{dimensions has rank 0 instead of required rank 1.}}
  %0 = "mhlo.reverse"(%operand) {
    dimensions = dense<2> : tensor<i64>
  } : (tensor<3x2xi32>) -> tensor<3x2xi32>
  func.return %0 : tensor<3x2xi32>
}

// -----

// CHECK-LABEL: func @dot_general
func.func @dot_general(%arg0: tensor<2x3x4xf32>, %arg1: tensor<2x3x5xf32>) -> tensor<2x4x5xf32> {
  %0 = "mhlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #mhlo.dot<
      lhs_batching_dimensions = [0],
      rhs_batching_dimensions = [0],
      lhs_contracting_dimensions = [1],
      rhs_contracting_dimensions = [1]
    >
  } : (tensor<2x3x4xf32>, tensor<2x3x5xf32>) -> tensor<2x4x5xf32>
  func.return %0 : tensor<2x4x5xf32>
}

// -----

func.func @dot_general(%arg0: tensor<1x?x1x?xf32>, %arg1: tensor<?x1x?x1x?xf32>) {
  %0 = "mhlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #mhlo.dot<
      lhs_batching_dimensions = [0, 1],
      rhs_batching_dimensions = [0, 1],
      lhs_contracting_dimensions = [2, 3],
      rhs_contracting_dimensions = [2, 3]
    >
  } : (tensor<1x?x1x?xf32>, tensor<?x1x?x1x?xf32>) -> tensor<?x?x?xf32>
  func.return
}

// -----

func.func @dot_general(%arg0: tensor<*xf32>, %arg1: tensor<*xf32>) {
  %0 = "mhlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #mhlo.dot<
      lhs_batching_dimensions = [0],
      rhs_batching_dimensions = [0],
      lhs_contracting_dimensions = [1],
      rhs_contracting_dimensions = [1]
    >
  } : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
  func.return
}

// -----

func.func @dot_general(%arg0: tensor<?x?x?xf32>, %arg1: tensor<*xf32>) {
  %0 = "mhlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #mhlo.dot<
      lhs_batching_dimensions = [0],
      rhs_batching_dimensions = [0],
      lhs_contracting_dimensions = [1],
      rhs_contracting_dimensions = [1]
    >
  } : (tensor<?x?x?xf32>, tensor<*xf32>) -> tensor<?x?x?xf32>
  func.return
}


// -----

func.func @dot_general(%arg0: tensor<?x?x?xf32>, %arg1: tensor<*xf32>) {
  %0 = "mhlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #mhlo.dot<
      lhs_batching_dimensions = [0],
      rhs_batching_dimensions = [0],
      lhs_contracting_dimensions = [1],
      rhs_contracting_dimensions = [1]
    >
  } : (tensor<?x?x?xf32>, tensor<*xf32>) -> tensor<?x?x?xf32>
  func.return
}

// -----

func.func @dot_general(%arg0: tensor<?x?x?xf32>, %arg1: tensor<?x?x?xf32>) {
  // expected-error @+1 {{lhs and rhs should have the same number of batching dimensions}}
  %0 = "mhlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #mhlo.dot<
      lhs_batching_dimensions = [0],
      rhs_batching_dimensions = [],
      lhs_contracting_dimensions = [1],
      rhs_contracting_dimensions = [1]
    >
  } : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  func.return
}

// -----

func.func @dot_general(%arg0: tensor<*xf32>, %arg1: tensor<*xf32>) {
  // expected-error @+1 {{lhs and rhs should have the same number of batching dimensions}}
  %0 = "mhlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #mhlo.dot<
      lhs_batching_dimensions = [0],
      rhs_batching_dimensions = [],
      lhs_contracting_dimensions = [1],
      rhs_contracting_dimensions = [1]
    >
  } : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
  func.return
}

// -----

func.func @dot_general(%arg0: tensor<?x?x?xf32>, %arg1: tensor<?x?x?xf32>) {
  // expected-error @+1 {{lhs and rhs should have the same number of contracting dimensions}}
  %0 = "mhlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #mhlo.dot<
      lhs_batching_dimensions = [0],
      rhs_batching_dimensions = [0],
      lhs_contracting_dimensions = [],
      rhs_contracting_dimensions = [1]
    >
  } : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  func.return
}

// -----

func.func @dot_general(%arg0: tensor<*xf32>, %arg1: tensor<*xf32>) {
  // expected-error @+1 {{lhs and rhs should have the same number of contracting dimensions}}
  %0 = "mhlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #mhlo.dot<
      lhs_batching_dimensions = [0],
      rhs_batching_dimensions = [0],
      lhs_contracting_dimensions = [],
      rhs_contracting_dimensions = [1]
    >
  } : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
  func.return
}

// -----

func.func @dot_general(%arg0: tensor<?x?x?xf32>, %arg1: tensor<?x?x?xf32>) {
  // expected-error @+1 {{has duplicated dimension from lhs_batching_dimensions and lhs_contracting_dimensions: 0}}
  %0 = "mhlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #mhlo.dot<
      lhs_batching_dimensions = [0, 0],
      rhs_batching_dimensions = [0, 0],
      lhs_contracting_dimensions = [1],
      rhs_contracting_dimensions = [1]
    >
  } : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  func.return
}

// -----

func.func @dot_general(%arg0: tensor<*xf32>, %arg1: tensor<*xf32>) {
  // expected-error @+1 {{has duplicated dimension from lhs_batching_dimensions and lhs_contracting_dimensions: 0}}
  %0 = "mhlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #mhlo.dot<
      lhs_batching_dimensions = [0, 0],
      rhs_batching_dimensions = [0, 0],
      lhs_contracting_dimensions = [1],
      rhs_contracting_dimensions = [1]
    >
  } : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
  func.return
}

// -----

func.func @dot_general(%arg0: tensor<?x?x?xf32>, %arg1: tensor<?x?x?xf32>) {
  // expected-error @+1 {{has duplicated dimension from lhs_batching_dimensions and lhs_contracting_dimensions: 1}}
  %0 = "mhlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #mhlo.dot<
      lhs_batching_dimensions = [0],
      rhs_batching_dimensions = [0],
      lhs_contracting_dimensions = [1, 1],
      rhs_contracting_dimensions = [1, 1]
    >
  } : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  func.return
}

// -----

func.func @dot_general(%arg0: tensor<*xf32>, %arg1: tensor<*xf32>) {
  // expected-error @+1 {{has duplicated dimension from lhs_batching_dimensions and lhs_contracting_dimensions: 1}}
  %0 = "mhlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #mhlo.dot<
      lhs_batching_dimensions = [0],
      rhs_batching_dimensions = [0],
      lhs_contracting_dimensions = [1, 1],
      rhs_contracting_dimensions = [1, 1]
    >
  } : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
  func.return
}

// -----

func.func @dot_general(%arg0: tensor<?x?x?xf32>, %arg1: tensor<?x?x?xf32>) {
  // expected-error @+1 {{has duplicated dimension from lhs_batching_dimensions and lhs_contracting_dimensions: 0}}
  %0 = "mhlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #mhlo.dot<
      lhs_batching_dimensions = [0],
      rhs_batching_dimensions = [0],
      lhs_contracting_dimensions = [0],
      rhs_contracting_dimensions = [1]
    >
  } : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  func.return
}

// -----

func.func @dot_general(%arg0: tensor<*xf32>, %arg1: tensor<*xf32>) {
  // expected-error @+1 {{has duplicated dimension from lhs_batching_dimensions and lhs_contracting_dimensions: 0}}
  %0 = "mhlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #mhlo.dot<
      lhs_batching_dimensions = [0],
      rhs_batching_dimensions = [0],
      lhs_contracting_dimensions = [0],
      rhs_contracting_dimensions = [1]
    >
  } : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
  func.return
}

// -----

func.func @dot_general(%arg0: tensor<?x?x?xf32>, %arg1: tensor<?x?x?xf32>) {
  // expected-error @+1 {{has duplicated dimension from rhs_batching_dimensions and rhs_contracting_dimensions: 0}}
  %0 = "mhlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #mhlo.dot<
      lhs_batching_dimensions = [0],
      rhs_batching_dimensions = [0],
      lhs_contracting_dimensions = [1],
      rhs_contracting_dimensions = [0]
    >
  } : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  func.return
}

// -----

func.func @dot_general(%arg0: tensor<*xf32>, %arg1: tensor<*xf32>) {
  // expected-error @+1 {{has duplicated dimension from rhs_batching_dimensions and rhs_contracting_dimensions: 0}}
  %0 = "mhlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #mhlo.dot<
      lhs_batching_dimensions = [0],
      rhs_batching_dimensions = [0],
      lhs_contracting_dimensions = [1],
      rhs_contracting_dimensions = [0]
    >
  } : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
  func.return
}

// -----

func.func @dot_general(%arg0: tensor<?x?x?xf32>, %arg1: tensor<?x?x?xf32>) {
  // expected-error @+1 {{lhs_batching_dimensions value: -1 is out of range: [0, 3)}}
  %0 = "mhlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #mhlo.dot<
      lhs_batching_dimensions = [-1],
      rhs_batching_dimensions = [0],
      lhs_contracting_dimensions = [1],
      rhs_contracting_dimensions = [1]
    >
  } : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  func.return
}

// -----

func.func @dot_general(%arg0: tensor<?x?x?xf32>, %arg1: tensor<?x?x?xf32>) {
  // expected-error @+1 {{lhs_batching_dimensions value: 3 is out of range: [0, 3)}}
  %0 = "mhlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #mhlo.dot<
      lhs_batching_dimensions = [3],
      rhs_batching_dimensions = [0],
      lhs_contracting_dimensions = [1],
      rhs_contracting_dimensions = [1]
    >
  } : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  func.return
}

// -----

func.func @dot_general(%arg0: tensor<?x?x?xf32>, %arg1: tensor<?x?x?xf32>) {
  // expected-error @+1 {{rhs_batching_dimensions value: -1 is out of range: [0, 3)}}
  %0 = "mhlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #mhlo.dot<
      lhs_batching_dimensions = [0],
      rhs_batching_dimensions = [-1],
      lhs_contracting_dimensions = [1],
      rhs_contracting_dimensions = [1]
    >
  } : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  func.return
}

// -----

func.func @dot_general(%arg0: tensor<?x?x?xf32>, %arg1: tensor<?x?x?xf32>) {
  // expected-error @+1 {{rhs_batching_dimensions value: 3 is out of range: [0, 3)}}
  %0 = "mhlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #mhlo.dot<
      lhs_batching_dimensions = [0],
      rhs_batching_dimensions = [3],
      lhs_contracting_dimensions = [1],
      rhs_contracting_dimensions = [1]
    >
  } : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  func.return
}

// -----

func.func @dot_general(%arg0: tensor<?x?x?xf32>, %arg1: tensor<?x?x?xf32>) {
  // expected-error @+1 {{lhs_contracting_dimensions value: -1 is out of range: [0, 3)}}
  %0 = "mhlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #mhlo.dot<
      lhs_batching_dimensions = [0],
      rhs_batching_dimensions = [0],
      lhs_contracting_dimensions = [-1],
      rhs_contracting_dimensions = [1]
    >
  } : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  func.return
}

// -----

func.func @dot_general(%arg0: tensor<?x?x?xf32>, %arg1: tensor<?x?x?xf32>) {
  // expected-error @+1 {{lhs_contracting_dimensions value: 3 is out of range: [0, 3)}}
  %0 = "mhlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #mhlo.dot<
      lhs_batching_dimensions = [0],
      rhs_batching_dimensions = [0],
      lhs_contracting_dimensions = [3],
      rhs_contracting_dimensions = [1]
    >
  } : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  func.return
}

// -----

func.func @dot_general(%arg0: tensor<?x?x?xf32>, %arg1: tensor<?x?x?xf32>) {
  // expected-error @+1 {{rhs_contracting_dimensions value: -1 is out of range: [0, 3)}}
  %0 = "mhlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #mhlo.dot<
      lhs_batching_dimensions = [0],
      rhs_batching_dimensions = [0],
      lhs_contracting_dimensions = [1],
      rhs_contracting_dimensions = [-1]
    >
  } : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  func.return
}

// -----

func.func @dot_general(%arg0: tensor<?x?x?xf32>, %arg1: tensor<?x?x?xf32>) {
  // expected-error @+1 {{rhs_contracting_dimensions value: 3 is out of range: [0, 3)}}
  %0 = "mhlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #mhlo.dot<
      lhs_batching_dimensions = [0],
      rhs_batching_dimensions = [0],
      lhs_contracting_dimensions = [1],
      rhs_contracting_dimensions = [3]
    >
  } : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  func.return
}

// -----

func.func @dot_general(%arg0: tensor<2x?x?xf32>, %arg1: tensor<3x?x?xf32>) {
  // expected-error @+1 {{batching dimension sizes must match for lhs/rhs}}
  %0 = "mhlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #mhlo.dot<
      lhs_batching_dimensions = [0],
      rhs_batching_dimensions = [0],
      lhs_contracting_dimensions = [1],
      rhs_contracting_dimensions = [1]
    >
  } : (tensor<2x?x?xf32>, tensor<3x?x?xf32>) -> tensor<?x?x?xf32>
  func.return
}

// -----

func.func @dot_general(%arg0: tensor<?x2x?xf32>, %arg1: tensor<?x3x?xf32>) {
  // expected-error @+1 {{contracting dimension sizes must match for lhs/rhs}}
  %0 = "mhlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #mhlo.dot<
      lhs_batching_dimensions = [0],
      rhs_batching_dimensions = [0],
      lhs_contracting_dimensions = [1],
      rhs_contracting_dimensions = [1]
    >
  } : (tensor<?x2x?xf32>, tensor<?x3x?xf32>) -> tensor<?x?x?xf32>
  func.return
}

// -----

// CHECK-LABEL: func @dot_general
func.func @dot_general(%arg0: tensor<2x3x4xf32>, %arg1: tensor<2x3x5xf32>) -> tensor<2x4x6xf32> {
  // expected-error@+1 {{inferred shape '[2, 4, 5]' is incompatible with return type of operation 'tensor<2x4x6xf32>'}}
  %0 = "mhlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #mhlo.dot<
      lhs_batching_dimensions = [0],
      rhs_batching_dimensions = [0],
      lhs_contracting_dimensions = [1],
      rhs_contracting_dimensions = [1]
    >
  } : (tensor<2x3x4xf32>, tensor<2x3x5xf32>) -> tensor<2x4x6xf32>
  func.return %0 : tensor<2x4x6xf32>
}


// -----

func.func @dot_general_one_element_precision_config(%arg0: tensor<2x3x4xf32>, %arg1: tensor<2x3x5xf32>) -> tensor<2x4x5xf32> {
  %0 = "mhlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #mhlo.dot<
      lhs_batching_dimensions = [0],
      rhs_batching_dimensions = [0],
      lhs_contracting_dimensions = [1],
      rhs_contracting_dimensions = [1]
    >,
    precision_config = [#mhlo<precision DEFAULT>]
  } : (tensor<2x3x4xf32>, tensor<2x3x5xf32>) -> tensor<2x4x5xf32>
  func.return %0 : tensor<2x4x5xf32>
}

// -----

func.func @dot_general_three_element_precision_config(%arg0: tensor<2x3x4xf32>, %arg1: tensor<2x3x5xf32>) -> tensor<2x4x5xf32> {
  // expected-error@+1 {{expects precision config to be empty or have <= 2 elements}}
  %0 = "mhlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #mhlo.dot<
      lhs_batching_dimensions = [0],
      rhs_batching_dimensions = [0],
      lhs_contracting_dimensions = [1],
      rhs_contracting_dimensions = [1]
    >,
    precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]
  } : (tensor<2x3x4xf32>, tensor<2x3x5xf32>) -> tensor<2x4x5xf32>
  func.return %0 : tensor<2x4x5xf32>
}

// -----

func.func @compatible_shapes(%arg0: tensor<?xf32>, %shape: tensor<2xindex>) -> tensor<?x?xf32> {
  %0 = "mhlo.dynamic_reshape"(%arg0, %shape) : (tensor<?xf32>, tensor<2xindex>) -> tensor<?x?xf32>
  func.return %0 : tensor<?x?xf32>
}

// -----

func.func @incompatible_shapes(%arg0: tensor<?xf32>, %shape: tensor<2xindex>) -> tensor<?xf32> {
  // expected-error @+1 {{output should have a rank equal to the number of elements in output_shape}}
  %0 = "mhlo.dynamic_reshape"(%arg0, %shape) : (tensor<?xf32>, tensor<2xindex>) -> tensor<?xf32>
  func.return %0 : tensor<?xf32>
}

// -----

func.func @cbrt(%arg: tensor<2x4xf32>) -> tensor<2x4xf32> {
  %0 = "mhlo.cbrt"(%arg) : (tensor<2x4xf32>) -> tensor<2x4xf32>
  func.return %0 : tensor<2x4xf32>
}

// -----

func.func @bitcast(%arg: tensor<2x4xf32>) -> tensor<2x4xf32> {
  %0 = "mhlo.bitcast"(%arg) : (tensor<2x4xf32>) -> tensor<2x4xf32>
  func.return %0 : tensor<2x4xf32>
}

// -----

func.func @bitcast_convert_int(%arg: tensor<2xf32>) -> tensor<2x4xi8> {
  %0 = "mhlo.bitcast_convert"(%arg) : (tensor<2xf32>) -> tensor<2x4xi8>
  return %0 : tensor<2x4xi8>
}

// -----

func.func @bitcast_convert_from_int(%arg: tensor<2x4xi8>) -> tensor<2xf32> {
  %0 = "mhlo.bitcast_convert"(%arg) : (tensor<2x4xi8>) -> tensor<2xf32>
  return %0 : tensor<2xf32>
}

// -----


func.func @bitcast_convert_complex(%arg: tensor<complex<f64>>) -> tensor<2xcomplex<f32>> {
  %0 = "mhlo.bitcast_convert"(%arg) : (tensor<complex<f64>>) -> tensor<2xcomplex<f32>>
  return %0 : tensor<2xcomplex<f32>>
}

// -----

func.func @invalid_bitcast_convert_decomplex(%arg: tensor<2x4xcomplex<f32>>) -> tensor<2x4xf64> {
  // expected-error@+1 {{cannot convert between real and complex types}}
  %0 = "mhlo.bitcast_convert"(%arg) : (tensor<2x4xcomplex<f32>>) -> tensor<2x2xf64>
  return %0 : tensor<2x2xf64>
}

// -----

func.func @bitcast_convert_scalar(%arg: tensor<f32>) -> tensor<f32> {
  %0 = "mhlo.bitcast_convert"(%arg) : (tensor<f32>) -> tensor<f32>
  return %0 : tensor<f32>
}

// -----

func.func @bitcast_convert_invalid_scalar(%arg: tensor<f64>) -> tensor<f32> {
  // expected-error@+1 {{does not allow the smaller element type to be part of a 0d tensor, but got: 'tensor<f64>' and 'tensor<f32>'.}}
  %0 = "mhlo.bitcast_convert"(%arg) : (tensor<f64>) -> tensor<f32>
  return %0 : tensor<f32>
}

// -----

func.func @bitcast_convert(%arg: tensor<*xf32>) -> tensor<*xf32> {
  %0 = "mhlo.bitcast_convert"(%arg) : (tensor<*xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}

// -----

func.func @invalid_bitcast_convert_width_mismatch(%arg: tensor<2x4xf64>) -> tensor<2x4xf32> {
  // expected-error@+1 {{requires compatible bitwidths. Got: 'tensor<2x4xf64>' and 'tensor<2x4xf32>', but 32 * 4 != 64.}}
  %0 = "mhlo.bitcast_convert"(%arg) : (tensor<2x4xf64>) -> tensor<2x4xf32>
  return %0 : tensor<2x4xf32>
}

// -----

func.func @bitcast_convert_width_mismatch(%arg: tensor<f32>) -> tensor<f64> {
  // expected-error@+1 {{does not allow the smaller element type to be part of a 0d tensor, but got: 'tensor<f32>' and 'tensor<f64>'.}}
  %0 = "mhlo.bitcast_convert"(%arg) : (tensor<f32>) -> tensor<f64>
  return %0 : tensor<f64>
}

// -----

func.func @bitcast_convert_empty_target(%arg: tensor<1xf64>) -> tensor<f32> {
  // expected-error@+1 {{does not allow the smaller element type to be part of a 0d tensor, but got: 'tensor<1xf64>' and 'tensor<f32>'.}}
  %0 = "mhlo.bitcast_convert"(%arg) : (tensor<1xf64>) -> tensor<f32>
  return %0 : tensor<f32>
}

// -----

func.func @bitcast_convert_empty_operand(%arg: tensor<f32>) -> tensor<1xf64> {
  // expected-error@+1 {{does not allow the smaller element type to be part of a 0d tensor, but got: 'tensor<f32>' and 'tensor<1xf64>'.}}
  %0 = "mhlo.bitcast_convert"(%arg) : (tensor<f32>) -> tensor<1xf64>
  return %0 : tensor<1xf64>
}

// -----

func.func @invalid_bitcast_convert_width_mismatch(%arg: tensor<2x4xf32>) -> tensor<2x4xf64> {
  // expected-error@+1 {{requires compatible bitwidths. Got: 'tensor<2x4xf32>' and 'tensor<2x4xf64>', but 32 * 4 != 64.}}
  %0 = "mhlo.bitcast_convert"(%arg) : (tensor<2x4xf32>) -> tensor<2x4xf64>
  return %0 : tensor<2x4xf64>
}

// -----

func.func @invalid_bitcast_convert_shape_mismatch(%arg: tensor<2x4xf32>) -> tensor<4x4xf32> {
  // expected-error@+1 {{operand and result shapes must match except for the innermost dimension of the shape with the smaller element type. Got: 'tensor<2x4xf32>' and 'tensor<4x4xf32>'.}}
  %0 = "mhlo.bitcast_convert"(%arg) : (tensor<2x4xf32>) -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}

// -----

func.func @stochastic_convert(%arg0: tensor<2x4xf32>, %arg1: tensor<2x4xui32>) -> tensor<2x4xi8> {
  %0 = "mhlo.stochastic_convert"(%arg0, %arg1) : (tensor<2x4xf32>, tensor<2x4xui32>) -> tensor<2x4xi8>
  return %0 : tensor<2x4xi8>
}

// -----

func.func @invalid_stochastic_convert_disallowed_random_type(%arg0: tensor<2x4xf32>, %arg1: tensor<2x4xi32>) -> tensor<2x4xi8> {
  // expected-error@+1 {{must be tensor of 4/8/16/32/64-bit unsigned integer values, but got 'tensor<2x4xi32>'}}
  %0 = "mhlo.stochastic_convert"(%arg0, %arg1) : (tensor<2x4xf32>, tensor<2x4xi32>) -> tensor<2x4xi8>
  return %0 : tensor<2x4xi8>
}

// -----

func.func @invalid_stochastic_convert_mismatched_input_bitwidths(%arg0: tensor<2x4xf32>, %arg1: tensor<2x4xui16>) -> tensor<2x4xi8> {
  // expected-error@+1 {{requires the random's bitwidth to match the operand's, but got: 16 and 32}}
  %0 = "mhlo.stochastic_convert"(%arg0, %arg1) : (tensor<2x4xf32>, tensor<2x4xui16>) -> tensor<2x4xi8>
  return %0 : tensor<2x4xi8>
}

// -----

func.func @reduce_precision(%arg: tensor<2x4xf32>) -> tensor<2x4xf32> {
  %0 = "mhlo.reduce_precision"(%arg) {exponent_bits=2 : i32, mantissa_bits=3 : i32} : (tensor<2x4xf32>) -> tensor<2x4xf32>
  func.return %0 : tensor<2x4xf32>
}

// -----

func.func @reduce_precision_invalid_exponent(%arg: tensor<2x4xf32>) -> tensor<2x4xf32> {
  // expected-error @+1 {{exponent_bits must be at least 1.}}
  %0 = "mhlo.reduce_precision"(%arg) {exponent_bits=0 : i32, mantissa_bits=3 : i32} : (tensor<2x4xf32>) -> tensor<2x4xf32>
  func.return %0 : tensor<2x4xf32>
}

// -----

func.func @reduce_precision_invalid_mantissa(%arg: tensor<2x4xf32>) -> tensor<2x4xf32> {
  // expected-error @+1 {{mantissa_bits must be at least 0.}}
  %0 = "mhlo.reduce_precision"(%arg) {exponent_bits=1 : i32, mantissa_bits=-1 : i32} : (tensor<2x4xf32>) -> tensor<2x4xf32>
  func.return %0 : tensor<2x4xf32>
}

// -----

func.func @gather(%operand : tensor<2x4x9xi32>, %start_indices : tensor<1x5x2xi32>) -> tensor<1x5x8xi32> {
  %res = "mhlo.gather"(%operand, %start_indices) {
    dimension_numbers = #mhlo.gather<
      collapsed_slice_dims = [0, 1],
      index_vector_dim = 2,
      offset_dims = [2],
      start_index_map = [0, 1]
    >,
    indices_are_sorted = false,
    slice_sizes = dense<[1, 1, 8]> : tensor<3xi64>
  } : (tensor<2x4x9xi32>, tensor<1x5x2xi32>) -> tensor<1x5x8xi32>
  func.return %res : tensor<1x5x8xi32>
}

// -----

// CHECK: gather
func.func @gather(%operand : tensor<*xi32>, %start_indices : tensor<1x5x2xi32>) -> tensor<1x5x8xi32> {
  %res = "mhlo.gather"(%operand, %start_indices) {
    dimension_numbers = #mhlo.gather<
      collapsed_slice_dims = [0, 1],
      index_vector_dim = 2,
      offset_dims = [2],
      start_index_map = [0, 1]
    >,
    indices_are_sorted = false,
    slice_sizes = dense<[1, 1, 8]> : tensor<3xi64>
  } : (tensor<*xi32>, tensor<1x5x2xi32>) -> tensor<1x5x8xi32>
  func.return %res : tensor<1x5x8xi32>
}

// -----

// CHECK: gather
func.func @gather(%operand : tensor<2x4x9xi32>, %start_indices : tensor<*xi32>) -> tensor<1x5x8xi32> {
  %res = "mhlo.gather"(%operand, %start_indices) {
    dimension_numbers = #mhlo.gather<
      collapsed_slice_dims = [0, 1],
      index_vector_dim = 2,
      offset_dims = [2],
      start_index_map = [0, 1]
    >,
    indices_are_sorted = false,
    slice_sizes = dense<[1, 1, 8]> : tensor<3xi64>
  } : (tensor<2x4x9xi32>, tensor<*xi32>) -> tensor<1x5x8xi32>
  func.return %res : tensor<1x5x8xi32>
}

// -----

func.func @gather(%operand : tensor<*xi32>, %start_indices : tensor<*xi32>) -> tensor<*xi32> {
  %res = "mhlo.gather"(%operand, %start_indices) {
    dimension_numbers = #mhlo.gather<
      collapsed_slice_dims = [0, 1],
      index_vector_dim = 2,
      offset_dims = [2],
      start_index_map = [0, 1]
    >,
    indices_are_sorted = false,
    slice_sizes = dense<[1, 1, 8]> : tensor<3xi64>
  } : (tensor<*xi32>, tensor<*xi32>) -> tensor<*xi32>
  func.return %res : tensor<*xi32>
}

// -----

func.func @gather(%operand : tensor<2x4x9xi32>, %start_indices : tensor<1x5x2xi32>) -> tensor<1x5x8xi32> {
  // expected-error@+1 {{index_vector_dim 4 is out of bounds for start indices with rank 3}}
  %res = "mhlo.gather"(%operand, %start_indices) {
    dimension_numbers = #mhlo.gather<
      collapsed_slice_dims = [0, 1],
      index_vector_dim = 4,
      offset_dims = [2],
      start_index_map = [0, 1]
    >,
    indices_are_sorted = false,
    slice_sizes = dense<[1, 1, 8]> : tensor<3xi64>
  } : (tensor<2x4x9xi32>, tensor<1x5x2xi32>) -> tensor<1x5x8xi32>
  func.return %res : tensor<1x5x8xi32>
}

// -----

func.func @gather(%operand : tensor<2x4x9xi32>, %start_indices : tensor<1x5x2xi32>) -> tensor<1x5x8xi32> {
  // expected-error@+1 {{offset_dims size (2) plus collapse_slice_dims size (2) is not equal to operand rank (3)}}
  %res = "mhlo.gather"(%operand, %start_indices) {
    dimension_numbers = #mhlo.gather<
      collapsed_slice_dims = [0, 1],
      index_vector_dim = 2,
      offset_dims = [1, 2],
      start_index_map = [0, 1]
    >,
    indices_are_sorted = false,
    slice_sizes = dense<[1, 1, 8]> : tensor<3xi64>
  } : (tensor<2x4x9xi32>, tensor<1x5x2xi32>) -> tensor<1x5x8xi32>
  func.return %res : tensor<1x5x8xi32>
}

// -----

func.func @gather(%operand : tensor<2x4x9xi32>, %start_indices : tensor<1x5x2xi32>) -> tensor<1x5x8xi32> {
  // expected-error@+1 {{start_index_map size (1) is not equal to size of index dimension (2) of start_indices (2)}}
  %res = "mhlo.gather"(%operand, %start_indices) {
    dimension_numbers = #mhlo.gather<
      collapsed_slice_dims = [0, 1],
      index_vector_dim = 2,
      offset_dims = [2],
      start_index_map = [0]
    >,
    indices_are_sorted = false,
    slice_sizes = dense<[1, 1, 8]> : tensor<3xi64>
  } : (tensor<2x4x9xi32>, tensor<1x5x2xi32>) -> tensor<1x5x8xi32>
  func.return %res : tensor<1x5x8xi32>
}

// -----

func.func @gather(%operand : tensor<2x4x9xi32>, %start_indices : tensor<1x5x2xi32>) -> tensor<1x5x8xi32> {
  // expected-error@+1 {{slice_sizes.rank != 1}}
  %res = "mhlo.gather"(%operand, %start_indices) {
    dimension_numbers = #mhlo.gather<
      collapsed_slice_dims = [0, 1],
      index_vector_dim = 2,
      offset_dims = [2],
      start_index_map = [0, 1]
    >,
    indices_are_sorted = false,
    slice_sizes = dense<[[1, 1, 8]]> : tensor<1x3xi64>
  } : (tensor<2x4x9xi32>, tensor<1x5x2xi32>) -> tensor<1x5x8xi32>
  func.return %res : tensor<1x5x8xi32>
}

// -----

func.func @gather(%operand : tensor<2x4x9xi32>, %start_indices : tensor<1x5x2xi32>) -> tensor<1x5x8xi32> {
  // expected-error@+1 {{slice_sizes size (2) not equal to (implied) operand rank (3)}}
  %res = "mhlo.gather"(%operand, %start_indices) {
    dimension_numbers = #mhlo.gather<
      collapsed_slice_dims = [0, 1],
      index_vector_dim = 2,
      offset_dims = [2],
      start_index_map = [0, 1]
    >,
    indices_are_sorted = false,
    slice_sizes = dense<[1, 8]> : tensor<2xi64>
  } : (tensor<2x4x9xi32>, tensor<1x5x2xi32>) -> tensor<1x5x8xi32>
  func.return %res : tensor<1x5x8xi32>
}

// -----

func.func @gather(%operand : tensor<*xi32>, %start_indices : tensor<*xi32>) -> tensor<*xi32> {
  // expected-error@+1 {{slice_sizes size (6) not equal to (implied) operand rank (3)}}
  %res = "mhlo.gather"(%operand, %start_indices) {
    dimension_numbers = #mhlo.gather<
      collapsed_slice_dims = [0, 1],
      index_vector_dim = 2,
      offset_dims = [2],
      start_index_map = [0, 1]
    >,
    indices_are_sorted = false,
    slice_sizes = dense<[1, 1, 8, 1, 2, 3]> : tensor<6xi64>
  } : (tensor<*xi32>, tensor<*xi32>) -> tensor<*xi32>
  func.return %res : tensor<*xi32>
}

// -----

func.func @gather(%operand : tensor<2x4x9xi32>, %start_indices : tensor<1x5x2xi32>) -> tensor<3xi32> {
  // expected-error@+1 {{inferred type(s) 'tensor<1x5x8xi32>' are incompatible with return type(s) of operation 'tensor<3xi32>'}}
  %res = "mhlo.gather"(%operand, %start_indices) {
    dimension_numbers = #mhlo.gather<
      collapsed_slice_dims = [0, 1],
      index_vector_dim = 2,
      offset_dims = [2],
      start_index_map = [0, 1]
    >,
    indices_are_sorted = false,
    slice_sizes = dense<[1, 1, 8]> : tensor<3xi64>
  } : (tensor<2x4x9xi32>, tensor<1x5x2xi32>) -> tensor<3xi32>
  func.return %res : tensor<3xi32>
}

// -----

func.func @gather(%operand : tensor<*xi32>, %start_indices : tensor<?x?x?xi32>) -> tensor<3xi32> {
  // expected-error@+1 {{inferred type(s) 'tensor<8x?x7x1x6x1x?xi32>' are incompatible with return type(s) of operation 'tensor<3xi32>'}}
  %res = "mhlo.gather"(%operand, %start_indices) {
    dimension_numbers = #mhlo.gather<
      collapsed_slice_dims = [0, 1, 3],
      index_vector_dim = 2,
      offset_dims = [0, 2, 3, 4, 5],
      start_index_map = [0, 1]
    >,
    indices_are_sorted = false,
    slice_sizes = dense<[1, 1, 8, 1, 7, 1, 6, 1]> : tensor<8xi64>
  } : (tensor<*xi32>, tensor<?x?x?xi32>) -> tensor<3xi32>
  func.return %res : tensor<3xi32>
}

// -----

func.func @gather(%operand : tensor<*xi32>, %start_indices : tensor<*xi32>) -> tensor<*xi32> {
  // expected-error@+1 {{slice_sizes collapsed dimension 2 should <= 1 but got 8}}
  %res = "mhlo.gather"(%operand, %start_indices) {
    dimension_numbers = #mhlo.gather<
      collapsed_slice_dims = [0, 2],
      index_vector_dim = 2,
      offset_dims = [2],
      start_index_map = [0, 1]
    >,
    indices_are_sorted = false,
    slice_sizes = dense<[1, 1, 8]> : tensor<3xi64>
  } : (tensor<*xi32>, tensor<*xi32>) -> tensor<*xi32>
  func.return %res : tensor<*xi32>
}

// -----

func.func @gather(%operand : tensor<2x4x9xi32>, %start_indices : tensor<1x5x2xi32>) -> tensor<1x5x8xi32> {
  // expected-error@+1 {{collapsed dimension -1 is out of bounds for slice_sizes.size (3)}}
  %res = "mhlo.gather"(%operand, %start_indices) {
    dimension_numbers = #mhlo.gather<
      collapsed_slice_dims = [-1, 1],
      index_vector_dim = 2,
      offset_dims = [2],
      start_index_map = [0, 1]
    >,
    indices_are_sorted = false,
    slice_sizes = dense<[1, 1, 8]> : tensor<3xi64>
  } : (tensor<2x4x9xi32>, tensor<1x5x2xi32>) -> tensor<1x5x8xi32>
  func.return %res : tensor<1x5x8xi32>
}

// -----

func.func @gather(%operand : tensor<2x4x9xi32>, %start_indices : tensor<1x5x2xi32>) -> tensor<1x5x8xi32> {
  // expected-error@+1 {{collapsed dimension 17 is out of bounds for slice_sizes.size (3)}}
  %res = "mhlo.gather"(%operand, %start_indices) {
    dimension_numbers = #mhlo.gather<
      collapsed_slice_dims = [0, 17],
      index_vector_dim = 2,
      offset_dims = [2],
      start_index_map = [0, 1]
    >,
    indices_are_sorted = false,
    slice_sizes = dense<[1, 1, 8]> : tensor<3xi64>
  } : (tensor<2x4x9xi32>, tensor<1x5x2xi32>) -> tensor<1x5x8xi32>
  func.return %res : tensor<1x5x8xi32>
}

// -----

func.func @gather(%operand : tensor<?x?x2xi32>, %start_indices : tensor<*xi32>) -> tensor<*xi32> {
  // expected-error@+1 {{slice size (-1) is out of bounds for operand dimension (2) at index 2}}
  %res = "mhlo.gather"(%operand, %start_indices) {
    dimension_numbers = #mhlo.gather<
      collapsed_slice_dims = [0, 1],
      index_vector_dim = 2,
      offset_dims = [2],
      start_index_map = [0, 1]
    >,
    indices_are_sorted = false,
    slice_sizes = dense<[1, 1, -1]> : tensor<3xi64>
  } : (tensor<?x?x2xi32>, tensor<*xi32>) -> tensor<*xi32>
  func.return %res : tensor<*xi32>
}

// -----

func.func @gather(%operand : tensor<?x?x2xi32>, %start_indices : tensor<*xi32>) -> tensor<*xi32> {
  // expected-error@+1 {{slice size (8) is out of bounds for operand dimension (2) at index 2}}
  %res = "mhlo.gather"(%operand, %start_indices) {
    dimension_numbers = #mhlo.gather<
      collapsed_slice_dims = [0, 1],
      index_vector_dim = 2,
      offset_dims = [2],
      start_index_map = [0, 1]
    >,
    indices_are_sorted = false,
    slice_sizes = dense<[1, 1, 8]> : tensor<3xi64>
  } : (tensor<?x?x2xi32>, tensor<*xi32>) -> tensor<*xi32>
  func.return %res : tensor<*xi32>
}

// -----

func.func @gather(%operand : tensor<16x11xi32>, %start_indices : tensor<5x2xi32>) -> tensor<5x8x6xi32> {
  // expected-error@+1 {{expects offset_dims to not repeat, got: [2, 2]}}
  %res = "mhlo.gather"(%operand, %start_indices) {
    dimension_numbers = #mhlo.gather<
      collapsed_slice_dims = [],
      index_vector_dim = 1,
      offset_dims = [2, 2],
      start_index_map = [0, 1]
    >,
    indices_are_sorted = false,
    slice_sizes = dense<[8, 6]> : tensor<2xi64>
  } : (tensor<16x11xi32>, tensor<5x2xi32>) -> tensor<5x8x6xi32>
  func.return %res : tensor<5x8x6xi32>
}

// -----

func.func @gather(%operand : tensor<16x11xi32>, %start_indices : tensor<5x2xi32>) -> tensor<5x8x6xi32> {
  // expected-error@+1 {{expects offset_dims to be sorted, got: [2, 1]}}
  %res = "mhlo.gather"(%operand, %start_indices) {
    dimension_numbers = #mhlo.gather<
      collapsed_slice_dims = [],
      index_vector_dim = 1,
      offset_dims = [2, 1],
      start_index_map = [0, 1]
    >,
    indices_are_sorted = false,
    slice_sizes = dense<[8, 6]> : tensor<2xi64>
  } : (tensor<16x11xi32>, tensor<5x2xi32>) -> tensor<5x8x6xi32>
  func.return %res : tensor<5x8x6xi32>
}

// -----

func.func @gather(%operand : tensor<2x4x9xi32>, %start_indices : tensor<1x5x2xi32>) -> tensor<1x5x8xi32> {
  // expected-error@+1 {{expects collapsed_slice_dims to not repeat, got: [1, 1]}}
  %res = "mhlo.gather"(%operand, %start_indices) {
    dimension_numbers = #mhlo.gather<
      collapsed_slice_dims = [1, 1],
      index_vector_dim = 2,
      offset_dims = [2],
      start_index_map = [0, 1]
    >,
    indices_are_sorted = false,
    slice_sizes = dense<[1, 1, 8]> : tensor<3xi64>
  } : (tensor<2x4x9xi32>, tensor<1x5x2xi32>) -> tensor<1x5x8xi32>
  func.return %res : tensor<1x5x8xi32>
}

// -----

func.func @gather(%operand : tensor<2x4x9xi32>, %start_indices : tensor<1x5x2xi32>) -> tensor<1x5x8xi32> {
  // expected-error@+1 {{expects collapsed_slice_dims to be sorted, got: [1, 0]}}
  %res = "mhlo.gather"(%operand, %start_indices) {
    dimension_numbers = #mhlo.gather<
      collapsed_slice_dims = [1, 0],
      index_vector_dim = 2,
      offset_dims = [2],
      start_index_map = [0, 1]
    >,
    indices_are_sorted = false,
    slice_sizes = dense<[1, 1, 8]> : tensor<3xi64>
  } : (tensor<2x4x9xi32>, tensor<1x5x2xi32>) -> tensor<1x5x8xi32>
  func.return %res : tensor<1x5x8xi32>
}

// -----

func.func @gather(%operand : tensor<2x4x9xi32>, %start_indices : tensor<1x5x2xi32>) -> tensor<1x5x8xi32> {
  // expected-error@+1 {{offset_dims[0]: -1 is out of bounds for implied result rank 3}}
  %res = "mhlo.gather"(%operand, %start_indices) {
    dimension_numbers = #mhlo.gather<
      collapsed_slice_dims = [0, 1],
      index_vector_dim = 2,
      offset_dims = [-1],
      start_index_map = [0, 1]
    >,
    indices_are_sorted = false,
    slice_sizes = dense<[1, 1, 8]> : tensor<3xi64>
  } : (tensor<2x4x9xi32>, tensor<1x5x2xi32>) -> tensor<1x5x8xi32>
  func.return %res : tensor<1x5x8xi32>
}

// -----

func.func @gather(%operand : tensor<2x4x9xi32>, %start_indices : tensor<1x5x2xi32>) -> tensor<1x5x8xi32> {
  // expected-error@+1 {{offset_dims[0]: 3 is out of bounds for implied result rank 3}}
  %res = "mhlo.gather"(%operand, %start_indices) {
    dimension_numbers = #mhlo.gather<
      collapsed_slice_dims = [0, 1],
      index_vector_dim = 2,
      offset_dims = [3],
      start_index_map = [0, 1]
    >,
    indices_are_sorted = false,
    slice_sizes = dense<[1, 1, 8]> : tensor<3xi64>
  } : (tensor<2x4x9xi32>, tensor<1x5x2xi32>) -> tensor<1x5x8xi32>
  func.return %res : tensor<1x5x8xi32>
}

// -----

func.func @gather(%operand : tensor<2x4x9xi32>, %start_indices : tensor<1x5x2xi32>) -> tensor<1x5x8xi32> {
  // expected-error@+1 {{start_index_map[0]: -2 is out of bounds for operand rank 3}}
  %res = "mhlo.gather"(%operand, %start_indices) {
    dimension_numbers = #mhlo.gather<
      collapsed_slice_dims = [0, 1],
      index_vector_dim = 2,
      offset_dims = [2],
      start_index_map = [-2, -1]
    >,
    indices_are_sorted = false,
    slice_sizes = dense<[1, 1, 8]> : tensor<3xi64>
  } : (tensor<2x4x9xi32>, tensor<1x5x2xi32>) -> tensor<1x5x8xi32>
  func.return %res : tensor<1x5x8xi32>
}

// -----

func.func @gather(%operand : tensor<2x4x9xi32>, %start_indices : tensor<1x5x2xi32>) -> tensor<1x5x8xi32> {
  // expected-error@+1 {{start_index_map[1]: 3 is out of bounds for operand rank 3}}
  %res = "mhlo.gather"(%operand, %start_indices) {
    dimension_numbers = #mhlo.gather<
      collapsed_slice_dims = [0, 1],
      index_vector_dim = 2,
      offset_dims = [2],
      start_index_map = [0, 3]
    >,
    indices_are_sorted = false,
    slice_sizes = dense<[1, 1, 8]> : tensor<3xi64>
  } : (tensor<2x4x9xi32>, tensor<1x5x2xi32>) -> tensor<1x5x8xi32>
  func.return %res : tensor<1x5x8xi32>
}

// -----

func.func @gather(%operand : tensor<2x4x9xi32>, %start_indices : tensor<1x5x2xi32>) -> tensor<1x5x8xi32> {
  // expected-error@+1 {{expects start_index_map to not repeat, got: [0, 0]}}
  %res = "mhlo.gather"(%operand, %start_indices) {
    dimension_numbers = #mhlo.gather<
      collapsed_slice_dims = [0, 1],
      index_vector_dim = 2,
      offset_dims = [2],
      start_index_map = [0, 0]
    >,
    indices_are_sorted = false,
    slice_sizes = dense<[1, 1, 8]> : tensor<3xi64>
  } : (tensor<2x4x9xi32>, tensor<1x5x2xi32>) -> tensor<1x5x8xi32>
  func.return %res : tensor<1x5x8xi32>
}

// -----

func.func @dynamic_gather(%operand : tensor<2x4x9xi32>, %start_indices : tensor<1x5x2xi32>, %slice_sizes : tensor<3xi32>) -> tensor<1x5x8xi32> {
  %res = "mhlo.dynamic_gather"(%operand, %start_indices, %slice_sizes) {
    dimension_numbers = #mhlo.gather<
      collapsed_slice_dims = [0, 1],
      index_vector_dim = 2,
      offset_dims = [2],
      start_index_map = [0, 1]
    >,
    indices_are_sorted = false
  } : (tensor<2x4x9xi32>, tensor<1x5x2xi32>, tensor<3xi32>) -> tensor<1x5x8xi32>
  func.return %res : tensor<1x5x8xi32>
}

// -----

func.func @dynamic_gather(%operand : tensor<*xi32>, %start_indices : tensor<*xi32>, %slice_sizes : tensor<*xi32>) -> tensor<*xi32> {
  %res = "mhlo.dynamic_gather"(%operand, %start_indices, %slice_sizes) {
    dimension_numbers = #mhlo.gather<
      collapsed_slice_dims = [0, 1],
      index_vector_dim = 2,
      offset_dims = [2],
      start_index_map = [0, 1]
    >,
    indices_are_sorted = false
  } : (tensor<*xi32>, tensor<*xi32>, tensor<*xi32>) -> tensor<*xi32>
  func.return %res : tensor<*xi32>
}

// -----

func.func @dynamic_gather(%operand : tensor<2x4x9xi32>, %start_indices : tensor<*xi32>, %slice_sizes : tensor<*xi32>) -> tensor<*xi32> {
  %res = "mhlo.dynamic_gather"(%operand, %start_indices, %slice_sizes) {
    dimension_numbers = #mhlo.gather<
      collapsed_slice_dims = [0, 1],
      index_vector_dim = 2,
      offset_dims = [2],
      start_index_map = [0, 1]
    >,
    indices_are_sorted = false
  } : (tensor<2x4x9xi32>, tensor<*xi32>, tensor<*xi32>) -> tensor<*xi32>
  func.return %res : tensor<*xi32>
}


// -----

func.func @dynamic_gather(%operand : tensor<*xi32>, %start_indices : tensor<?x?x?xi32>, %slice_sizes : tensor<*xi32>) -> tensor<*xi32> {
  // expected-error@+1 {{index_vector_dim 4 is out of bounds for start indices with rank 3}}
  %res = "mhlo.dynamic_gather"(%operand, %start_indices, %slice_sizes) {
    dimension_numbers = #mhlo.gather<
      collapsed_slice_dims = [0, 1],
      index_vector_dim = 4,
      offset_dims = [2],
      start_index_map = [0, 1]
    >,
    indices_are_sorted = false
  } : (tensor<*xi32>, tensor<?x?x?xi32>, tensor<*xi32>) -> tensor<*xi32>
  func.return %res : tensor<*xi32>
}

// -----

func.func @dynamic_gather(%operand : tensor<?x?x?xi32>, %start_indices : tensor<*xi32>, %slice_sizes : tensor<*xi32>) -> tensor<*xi32> {
  // expected-error@+1 {{offset_dims size (2) plus collapse_slice_dims size (2) is not equal to operand rank (3)}}
  %res = "mhlo.dynamic_gather"(%operand, %start_indices, %slice_sizes) {
    dimension_numbers = #mhlo.gather<
      collapsed_slice_dims = [0, 1],
      index_vector_dim = 2,
      offset_dims = [1, 2],
      start_index_map = [0, 1]
    >,
    indices_are_sorted = false
  } : (tensor<?x?x?xi32>, tensor<*xi32>, tensor<*xi32>) -> tensor<*xi32>
  func.return %res : tensor<*xi32>
}

// -----

func.func @dynamic_gather(%operand : tensor<*xi32>, %start_indices : tensor<?x?x2xi32>, %slice_sizes : tensor<*xi32>) -> tensor<*xi32> {
  // expected-error@+1 {{start_index_map size (1) is not equal to size of index dimension (2) of start_indices (2)}}
  %res = "mhlo.dynamic_gather"(%operand, %start_indices, %slice_sizes) {
    dimension_numbers = #mhlo.gather<
      collapsed_slice_dims = [0, 1],
      index_vector_dim = 2,
      offset_dims = [2],
      start_index_map = [0]
    >,
    indices_are_sorted = false
  } : (tensor<*xi32>, tensor<?x?x2xi32>, tensor<*xi32>) -> tensor<*xi32>
  func.return %res : tensor<*xi32>
}

// -----

func.func @dynamic_gather(%operand : tensor<*xi32>, %start_indices : tensor<*xi32>, %slice_sizes : tensor<?x?xi32>) -> tensor<*xi32> {
  // expected-error@+1 {{slice_sizes.rank != 1}}
  %res = "mhlo.dynamic_gather"(%operand, %start_indices, %slice_sizes) {
    dimension_numbers = #mhlo.gather<
      collapsed_slice_dims = [0, 1],
      index_vector_dim = 2,
      offset_dims = [2],
      start_index_map = [0, 1]
    >,
    indices_are_sorted = false
  } : (tensor<*xi32>, tensor<*xi32>, tensor<?x?xi32>) -> tensor<*xi32>
  func.return %res : tensor<*xi32>
}

// -----

func.func @dynamic_gather(%operand : tensor<?x?x?xi32>, %start_indices : tensor<*xi32>, %slice_sizes : tensor<2xi32>) -> tensor<*xi32> {
  // expected-error@+1 {{slice_sizes size (2) not equal to (implied) operand rank (3)}}
  %res = "mhlo.dynamic_gather"(%operand, %start_indices, %slice_sizes) {
    dimension_numbers = #mhlo.gather<
      collapsed_slice_dims = [0, 1],
      index_vector_dim = 2,
      offset_dims = [2],
      start_index_map = [0, 1]
    >,
    indices_are_sorted = false
  } : (tensor<?x?x?xi32>, tensor<*xi32>, tensor<2xi32>) -> tensor<*xi32>
  func.return %res : tensor<*xi32>
}

// -----

func.func @dynamic_gather(%operand : tensor<2x4x9xi32>, %start_indices : tensor<1x5x2xi32>, %slice_sizes : tensor<3xi32>) -> tensor<3xi32> {
  // expected-error@+1 {{inferred type(s) 'tensor<1x5x?xi32>' are incompatible with return type(s) of operation 'tensor<3xi32>'}}
  %res = "mhlo.dynamic_gather"(%operand, %start_indices, %slice_sizes) {
    dimension_numbers = #mhlo.gather<
      collapsed_slice_dims = [0, 1],
      index_vector_dim = 2,
      offset_dims = [2],
      start_index_map = [0, 1]
    >,
    indices_are_sorted = false
  } : (tensor<2x4x9xi32>, tensor<1x5x2xi32>, tensor<3xi32>) -> tensor<3xi32>
  func.return %res : tensor<3xi32>
}

// -----

func.func @dynamic_gather(%operand : tensor<2x4x9xi32>, %start_indices : tensor<1x5x2xi32>, %slice_sizes : tensor<*xi32>) -> tensor<3xi32> {
  // expected-error@+1 {{inferred type(s) 'tensor<1x5x?xi32>' are incompatible with return type(s) of operation 'tensor<3xi32>'}}
  %res = "mhlo.dynamic_gather"(%operand, %start_indices, %slice_sizes) {
    dimension_numbers = #mhlo.gather<
      collapsed_slice_dims = [0, 1],
      index_vector_dim = 2,
      offset_dims = [2],
      start_index_map = [0, 1]
    >,
    indices_are_sorted = false
  } : (tensor<2x4x9xi32>, tensor<1x5x2xi32>, tensor<*xi32>) -> tensor<3xi32>
  func.return %res : tensor<3xi32>
}

// -----

func.func @dynamic_gather(%operand : tensor<?x?x?xi32>, %start_indices : tensor<?x?x?xi32>, %slice_sizes : tensor<3xi32>) -> tensor<3xi32> {
  // expected-error@+1 {{inferred type(s) 'tensor<?x?x?xi32>' are incompatible with return type(s) of operation 'tensor<3xi32>'}}
  %res = "mhlo.dynamic_gather"(%operand, %start_indices, %slice_sizes) {
    dimension_numbers = #mhlo.gather<
      collapsed_slice_dims = [0, 1],
      index_vector_dim = 2,
      offset_dims = [2],
      start_index_map = [0, 1]
    >,
    indices_are_sorted = false
  } : (tensor<?x?x?xi32>, tensor<?x?x?xi32>, tensor<3xi32>) -> tensor<3xi32>
  func.return %res : tensor<3xi32>
}

// -----

func.func @dynamic_gather(%operand : tensor<*xi32>, %start_indices : tensor<?x?x?xi32>, %slice_sizes : tensor<?xi32>) -> tensor<?xi32> {
  // expected-error@+1 {{inferred type(s) 'tensor<?x?x?xi32>' are incompatible with return type(s) of operation 'tensor<?xi32>'}}
  %res = "mhlo.dynamic_gather"(%operand, %start_indices, %slice_sizes) {
    dimension_numbers = #mhlo.gather<
      collapsed_slice_dims = [0, 1],
      index_vector_dim = 2,
      offset_dims = [2],
      start_index_map = [0, 1]
    >,
    indices_are_sorted = false
  } : (tensor<*xi32>, tensor<?x?x?xi32>, tensor<?xi32>) -> tensor<?xi32>
  func.return %res : tensor<?xi32>
}

// -----

func.func @get_dimension_size(%I: tensor<1x128x512xf32>) -> tensor<i32> {
  // expected-error@+1 {{requires dimension attribute in range [0, 3); found (3)}}
  %size = "mhlo.get_dimension_size"(%I) {dimension = 3 : i64} : (tensor<1x128x512xf32>) -> tensor<i32>
  func.return %size : tensor<i32>
}

// -----

func.func @get_dimension_size(%I: tensor<1x128x512xf32>) -> tensor<i32> {
  %size = "mhlo.get_dimension_size"(%I) {dimension = 2 : i64} : (tensor<1x128x512xf32>) -> tensor<i32>
  func.return %size : tensor<i32>
}

// -----

func.func @get_dimension_size_negative_dimension(%I: tensor<1x128x512xf32>) -> tensor<i32> {
  // expected-error@+1 {{requires dimension attribute in range [0, 3); found (-1)}}
  %size = "mhlo.get_dimension_size"(%I) {dimension = -1 : i64} : (tensor<1x128x512xf32>) -> tensor<i32>
  func.return %size : tensor<i32>
}

// -----

func.func @get_dimension_size_invalid_dimension(%I: tensor<1x128x512xf32>) -> tensor<i32> {
  // expected-error@+1 {{requires dimension attribute in range [0, 3); found (3)}}
  %size = "mhlo.get_dimension_size"(%I) {dimension = 3 : i64} : (tensor<1x128x512xf32>) -> tensor<i32>
  func.return %size : tensor<i32>
}

// -----

func.func @set_dimension_size(%I: tensor<1x128x512xf32>) -> tensor<1x128x512xf32> {
  %dim = mhlo.constant dense<512> : tensor<1xi32>

  // expected-error@+1 {{size operand should be of rank-0}}
  %result = "mhlo.set_dimension_size"(%I, %dim) {dimension = 2 : i64} : (tensor<1x128x512xf32>, tensor<1xi32>) -> tensor<1x128x512xf32>
  func.return %result : tensor<1x128x512xf32>
}

// -----

func.func @set_dimension_size_negative_dimension(%I: tensor<1x128x512xf32>) -> tensor<1x128x512xf32> {
  %dim = mhlo.constant dense<512> : tensor<i32>
  // expected-error@+1 {{requires dimension attribute in range [0, 3); found (-1)}}
  %result = "mhlo.set_dimension_size"(%I, %dim) {dimension =-1 : i64} : (tensor<1x128x512xf32>, tensor<i32>) -> tensor<1x128x512xf32>
  func.return %result : tensor<1x128x512xf32>
}

// -----

func.func @set_dimension_size_invalid_dimension(%I: tensor<1x128x512xf32>) -> tensor<1x128x512xf32> {
  %dim = mhlo.constant dense<512> : tensor<i32>
  // expected-error@+1 {{requires dimension attribute in range [0, 3); found (3)}}
  %result = "mhlo.set_dimension_size"(%I, %dim) {dimension = 3 : i64} : (tensor<1x128x512xf32>, tensor<i32>) -> tensor<1x128x512xf32>
  func.return %result : tensor<1x128x512xf32>
}

// -----

func.func @custom_call_with_dictionary_backend_config() {
  // CHECK: mhlo.custom_call @foo() {api_version = 4 : i32, backend_config = {foo = 42 : i32}}
  "mhlo.custom_call"() {api_version = 4 : i32, backend_config={foo = 42 : i32}, call_target_name = "foo"} : () -> ()
  func.return
}

// -----

func.func @custom_call_with_incompatible_backend_config() {
  // expected-error@+1 {{unsupported user-encoded backend config, backend config must be a dictionary attribute}}
  "mhlo.custom_call"() {api_version = 4 : i32, backend_config="bar=42", call_target_name = "foo"} : () -> ()
  func.return
}

// -----

func.func @custom_call_with_incompatible_backend_config() {
  // expected-error@+1 {{unsupported dictionary attribute backend config, backend config must be a user-encoded string attribute}}
  "mhlo.custom_call"() {api_version = 3 : i32, backend_config={bar = 42 : i32}, call_target_name = "foo"} : () -> ()
  func.return
}

// -----

// CHECK: func @custom_call_multiple_inputs_outputs
func.func @custom_call_multiple_inputs_outputs(%x: tensor<2xf32>, %token: !mhlo.token) -> tensor<2xf32> {
  %0:3 = "mhlo.custom_call"(%x, %token) {backend_config="", call_target_name = "foo", has_side_effect = false, custom_call_schedule = #mhlo<custom_call_schedule NONE>} : (tensor<2xf32>, !mhlo.token) -> (tensor<2xf32>, tensor<2xf32>, !mhlo.token)
  %1 = "mhlo.add"(%0#0, %0#1) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
  func.return %1 : tensor<2xf32>
}

// -----

// CHECK: func @custom_call_multiple_inputs_outputs_with_layout
func.func @custom_call_multiple_inputs_outputs_with_layout(%x: tensor<2xf32>, %token: !mhlo.token) -> tensor<f32> {
  %0:3 = "mhlo.custom_call"(%x, %token) {
    call_target_name = "foo",
    operand_layouts = [dense<[0]> : tensor<1xindex>, dense<> : tensor<0xindex>],
    result_layouts = [dense<> : tensor<0xindex>, dense<[0]> : tensor<1xindex>, dense<> : tensor<0xindex>]
  } : (tensor<2xf32>, !mhlo.token) -> (tensor<f32>, tensor<2xf32>, !mhlo.token)
  func.return %0#0 : tensor<f32>
}

// -----

// CHECK: func @custom_call_tuple_output_with_layout
func.func @custom_call_tuple_output_with_layout(%x: tensor<2xf32>, %token: !mhlo.token) -> tuple<tensor<2xf32>, tensor<2xf32>, !mhlo.token> {
  %0 = "mhlo.custom_call"(%x, %token) {
    call_target_name = "foo",
    operand_layouts = [dense<[0]> : tensor<1xindex>, dense<> : tensor<0xindex>],
    result_layouts = [dense<[0]> : tensor<1xindex>, dense<[0]> : tensor<1xindex>, dense<> : tensor<0xindex>]
  } : (tensor<2xf32>, !mhlo.token) -> tuple<tensor<2xf32>, tensor<2xf32>, !mhlo.token>
  func.return %0 : tuple<tensor<2xf32>, tensor<2xf32>, !mhlo.token>
}

// -----

func.func @custom_call_only_operand_layout_constraints(%x: tensor<2xf32>, %token: !mhlo.token) -> tensor<2xf32> {
  // expected-error@+1 {{Layout attributes should be specified for either both operands and results or none}}
  %0:3 = "mhlo.custom_call"(%x, %token) {
    call_target_name = "foo",
    operand_layouts = [dense<[0]> : tensor<1xindex>, dense<> : tensor<0xindex>]
  } : (tensor<2xf32>, !mhlo.token) -> (tensor<2xf32>, tensor<2xf32>, !mhlo.token)
  func.return %0#0 : tensor<2xf32>
}

// -----

func.func @custom_call_layout_mismatch_num_operands(%x: tensor<2xf32>, %token: !mhlo.token) -> tensor<2xf32> {
  // expected-error@+1 {{Number of operands must match the number of operand layouts, 2 != 1}}
  %0:3 = "mhlo.custom_call"(%x, %token) {
    call_target_name = "foo",
    operand_layouts = [dense<[0]> : tensor<1xindex>],
    result_layouts = [dense<[0]> : tensor<1xindex>, dense<[0]> : tensor<1xindex>, dense<> : tensor<0xindex>]
  } : (tensor<2xf32>, !mhlo.token) -> (tensor<2xf32>, tensor<2xf32>, !mhlo.token)
  func.return %0#0 : tensor<2xf32>
}

// -----

func.func @custom_call_layout_mismatch_num_results() -> tensor<2xf32> {
  // expected-error@+1 {{Number of results must match the number of result layouts, 3 != 2}}
  %0:3 = "mhlo.custom_call"() {
    call_target_name = "foo",
    operand_layouts = [],
    result_layouts = [dense<[0]> : tensor<1xindex>, dense<[0]> : tensor<1xindex>]
  } : () -> (tensor<2xf32>, tensor<2xf32>, !mhlo.token)
  func.return %0#0 : tensor<2xf32>
}

// -----

func.func @custom_call_layout_mismatch_num_results_tuple(%x: tensor<2xf32>, %token: !mhlo.token) -> tuple<tensor<2xf32>, tensor<2xf32>, !mhlo.token> {
  // expected-error@+1 {{Number of results must match the number of result layouts, 3 != 2}}
  %0 = "mhlo.custom_call"(%x, %token) {
    call_target_name = "foo",
    operand_layouts = [dense<[0]> : tensor<1xindex>, dense<> : tensor<0xindex>],
    result_layouts = [dense<[0]> : tensor<1xindex>, dense<[0]> : tensor<1xindex>]
  } : (tensor<2xf32>, !mhlo.token) -> tuple<tensor<2xf32>, tensor<2xf32>, !mhlo.token>
  func.return %0 : tuple<tensor<2xf32>, tensor<2xf32>, !mhlo.token>
}

// -----

func.func @custom_call_tuple_operand_input(%x: tuple<tensor<2xf32>>, %token: !mhlo.token) -> tuple<tensor<2xf32>, tensor<2xf32>, !mhlo.token> {
  // expected-error@+1 {{Tuple types are not fully supported with layout constraints yet}}
  %0 = "mhlo.custom_call"(%x, %token) {
    call_target_name = "foo",
    operand_layouts = [dense<[0]> : tensor<1xindex>, dense<> : tensor<0xindex>],
    result_layouts = [dense<[0]> : tensor<1xindex>, dense<[0]> : tensor<1xindex>, dense<> : tensor<0xindex>]
  } : (tuple<tensor<2xf32>>, !mhlo.token) -> tuple<tensor<2xf32>, tensor<2xf32>, !mhlo.token>
  func.return %0 : tuple<tensor<2xf32>, tensor<2xf32>, !mhlo.token>
}

// -----

func.func @custom_call_token_with_layout(%token: !mhlo.token) {
  // expected-error@+1 {{Only tensor types can have non-empty layout: operand #0 of type '!mhlo.token' has layout dense<[0, 1]> : tensor<2xindex>}}
  "mhlo.custom_call"(%token) {
    call_target_name = "foo",
    operand_layouts = [dense<[0, 1]> : tensor<2xindex>],
    result_layouts = []
  } : (!mhlo.token) -> ()
  func.return
}

// -----

func.func @custom_call_mismatch_tensor_and_layout_rank(%arg: tensor<2x3xf32>) {
  // expected-error@+1 {{incorrect layout dense<[0, 1, 2]> : tensor<3xindex> for type 'tensor<2x3xf32>', layout must be a permutation of [0, 2)}}
  "mhlo.custom_call"(%arg) {
    call_target_name = "foo",
    operand_layouts = [dense<[0, 1, 2]> : tensor<3xindex>],
    result_layouts = []
  } : (tensor<2x3xf32>) -> ()
  func.return
}

// -----

func.func @custom_call_mismatch_tensor_and_layout_permutation(%arg: tensor<1x2x3xf32>) {
  // expected-error@+1 {{incorrect layout dense<[0, 1, 3]> : tensor<3xindex> for type 'tensor<1x2x3xf32>', layout must be a permutation of [0, 3)}}
  "mhlo.custom_call"(%arg) {
    call_target_name = "foo",
    operand_layouts = [dense<[0, 1, 3]> : tensor<3xindex>],
    result_layouts = []
  } : (tensor<1x2x3xf32>) -> ()
  func.return
}

// -----

// CHECK-LABEL: func @custom_call_output_operand_alias
func.func @custom_call_output_operand_alias(%arg0: tuple<tensor<1x1xf32>, tensor<2x3xf32>>, %arg1: tensor<5x5xf32>) {
  // CHECK: mhlo.custom_call
  // CHECK-SAME{LITERAL}: output_operand_aliases = [#mhlo.output_operand_alias<output_tuple_indices = [0], operand_index = 0, operand_tuple_indices = [1]>]}
  %0 = "mhlo.custom_call"(%arg0, %arg1) {
    call_target_name = "foo",
    output_operand_aliases = [
      #mhlo.output_operand_alias<output_tuple_indices = [0],
                                 operand_index = 0,
                                 operand_tuple_indices = [1]>
    ]
  } : (tuple<tensor<1x1xf32>, tensor<2x3xf32>>, tensor<5x5xf32>) -> tuple<tensor<2x3xf32>>
  func.return
}

// -----

// CHECK-LABEL: func @custom_call_output_operand_alias_when_output_not_tuple
func.func @custom_call_output_operand_alias_when_output_not_tuple(%arg0: tuple<tensor<1x1xf32>, tensor<2x3xf32>>, %arg1: tensor<5x5xf32>) {
  %0 = "mhlo.custom_call"(%arg0, %arg1) {
    call_target_name = "foo",
    output_operand_aliases = [
      #mhlo.output_operand_alias<output_tuple_indices = [],
                                 operand_index = 0,
                                 operand_tuple_indices = [1]>
    ]
  } : (tuple<tensor<1x1xf32>, tensor<2x3xf32>>, tensor<5x5xf32>) -> tensor<2x3xf32>
  func.return
}

// -----

// CHECK-LABEL: func @custom_call_output_operand_alias_when_operand_not_tuple
func.func @custom_call_output_operand_alias_when_operand_not_tuple(%arg0: tensor<2x3xf32>, %arg1: tensor<5x5xf32>) {
  %0 = "mhlo.custom_call"(%arg0, %arg1) {
    call_target_name = "foo",
    output_operand_aliases = [
      #mhlo.output_operand_alias<output_tuple_indices = [0],
                                 operand_index = 0,
                                 operand_tuple_indices = []>
    ]
  } : (tensor<2x3xf32>, tensor<5x5xf32>) -> tuple<tensor<2x3xf32>>
  func.return
}

// -----

// CHECK-LABEL: func @custom_call_output_operand_alias_when_no_tuple
func.func @custom_call_output_operand_alias_when_no_tuple(%arg0: tensor<2x3xf32>, %arg1: tensor<5x5xf32>) {
  %0 = "mhlo.custom_call"(%arg0, %arg1) {
    call_target_name = "foo",
    output_operand_aliases = [
      #mhlo.output_operand_alias<output_tuple_indices = [],
                                 operand_index = 0,
                                 operand_tuple_indices = []>
    ]
  } : (tensor<2x3xf32>, tensor<5x5xf32>) -> tensor<2x3xf32>
  func.return
}

// -----

func.func @custom_call_output_operand_alias_mismatch_operand_index(%arg0: tuple<tensor<1x1xf32>, tensor<2x3xf32>>, %arg1: tensor<5x5xf32>) {
  // expected-error@+1 {{expects operandIndex in the output_operand_alias attribute to be in range [0, 2); got: 2}}
  %0 = "mhlo.custom_call"(%arg0, %arg1) {
    call_target_name = "foo",
    output_operand_aliases = [
      #mhlo.output_operand_alias<output_tuple_indices = [0],
                                 operand_index = 2,
                                 operand_tuple_indices = [1]>
    ]
  } : (tuple<tensor<1x1xf32>, tensor<2x3xf32>>, tensor<5x5xf32>) -> tuple<tensor<2x3xf32>>
  func.return
}

// -----

func.func @custom_call_invalid_output_tuple_indices(%arg0: tuple<tensor<1x1xf32>, tensor<2x3xf32>>, %arg1: tensor<5x5xf32>) {
  // expected-error@+1 {{output_tuple_indices in the output_operand_alias attribute out of bounds}}
  %0 = "mhlo.custom_call"(%arg0, %arg1) {
    call_target_name = "foo",
    output_operand_aliases = [
      #mhlo.output_operand_alias<output_tuple_indices = [1],
                                 operand_index = 0,
                                 operand_tuple_indices = [1]>
    ]
  } : (tuple<tensor<1x1xf32>, tensor<2x3xf32>>, tensor<5x5xf32>) -> tuple<tensor<2x3xf32>>
  func.return
}

// -----

func.func @custom_call_invalid_operand_tuple_indices(%arg0: tuple<tensor<1x1xf32>, tensor<2x3xf32>>, %arg1: tensor<5x5xf32>) {
  // expected-error@+1 {{operand_tuple_indices in the output_operand_alias attribute out of bounds}}
  %0 = "mhlo.custom_call"(%arg0, %arg1) {
    call_target_name = "foo",
    output_operand_aliases = [
      #mhlo.output_operand_alias<output_tuple_indices = [0],
                                 operand_index = 0,
                                 operand_tuple_indices = [2]>
    ]
  } : (tuple<tensor<1x1xf32>, tensor<2x3xf32>>, tensor<5x5xf32>) -> tuple<tensor<2x3xf32>>
  func.return
}

// -----

func.func @custom_call_output_operand_alias(%arg0: tuple<tensor<1x1xf32>, tensor<2x3xf32>>, %arg1: tensor<5x5xf32>) {
  // expected-error@+1 {{shapes mismatch in the output_operand_alias attribute: operand part has type 'tensor<2x3xf32>' and output part has type 'tensor<20x30xf32>'}}
  %0 = "mhlo.custom_call"(%arg0, %arg1) {
    call_target_name = "foo",
    output_operand_aliases = [
      #mhlo.output_operand_alias<output_tuple_indices = [0],
                                 operand_index = 0,
                                 operand_tuple_indices = [1]>
    ]
  } : (tuple<tensor<1x1xf32>, tensor<2x3xf32>>, tensor<5x5xf32>) -> tuple<tensor<20x30xf32>>
  func.return
}

// -----
// CHECK: func @conv2d_generic
// CHECK: mhlo.convolution
// CHECK-SAME: dim_numbers = [b, 0, 1, ?, f]x[0, 1, ?, i, o]->[?, b, 0, 1, f]
// CHECK-SAME{LITERAL}: window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
func.func @conv2d_generic(%arg0: tensor<1x8x8x32x207xf32>, %arg1: tensor<3x3x32x207x16xf32>) -> tensor<32x1x8x8x16xf32> {
  %0 = "mhlo.convolution"(%arg0, %arg1) {batch_group_count = 1 : i64,
    dimension_numbers = #mhlo.conv<raw
      input_batch_dimension = 0,
      input_feature_dimension = 4,
      input_spatial_dimensions = [1, 2],
      kernel_input_feature_dimension = 3,
      kernel_output_feature_dimension = 4,
      kernel_spatial_dimensions = [0, 1],
      output_batch_dimension = 1,
      output_feature_dimension = 4,
      output_spatial_dimensions = [2, 3]
    >, feature_group_count = 1 : i64, lhs_dilation = dense<1> : tensor<2xi64>, padding = dense<1> : tensor<2x2xi64>, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>], rhs_dilation = dense<1> : tensor<2xi64>, window_strides = dense<1> : tensor<2xi64>} :
       (tensor<1x8x8x32x207xf32>, tensor<3x3x32x207x16xf32>) -> tensor<32x1x8x8x16xf32>
  func.return %0 : tensor<32x1x8x8x16xf32>
}

// CHECK: func @conv2d
// CHECK: mhlo.convolution
// CHECK-SAME: dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f]
// CHECK-SAME{LITERAL}: window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
func.func @conv2d(%arg0: tensor<1x8x8x207xf32>, %arg1: tensor<3x3x207x16xf32>) -> tensor<1x8x8x16xf32> {
  %0 = mhlo.convolution(%arg0, %arg1)
         dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
         window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
         {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} :
       (tensor<1x8x8x207xf32>, tensor<3x3x207x16xf32>) -> tensor<1x8x8x16xf32>
  func.return %0 : tensor<1x8x8x16xf32>
}

// CHECK: func @conv_empty_spatial_dimensions
// CHECK: mhlo.convolution
// CHECK-SAME: dim_numbers = [b, f]x[i, o]->[b, f]
// CHECK-SAME{LITERAL}: window = {stride = [], pad = [], lhs_dilate = [], rhs_dilate = [], reverse = []}
func.func @conv_empty_spatial_dimensions(%arg0: tensor<3x2xf16>, %arg1: tensor<2x2xf16>) -> tuple<tensor<3x2xf16>> {
  %0 = mhlo.convolution(%arg0, %arg1)
         dim_numbers = [b, f]x[i, o]->[b, f],
         window = {stride = [], pad = [], lhs_dilate = [], rhs_dilate = [], reverse = []}
         {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]}
       : (tensor<3x2xf16>, tensor<2x2xf16>) -> tensor<3x2xf16>
  %1 = "mhlo.tuple"(%0) : (tensor<3x2xf16>) -> tuple<tensor<3x2xf16>>
  func.return %1 : tuple<tensor<3x2xf16>>
}
// -----

func.func @conv2d(%arg0: tensor<1x8x8x207xf32>, %arg1: tensor<3x3x207x16xf32>) -> tensor<1x8x8x16xf32> {
  // expected-error @+3 {{'mhlo.convolution' Expected array with 2 elements, got 3 elements instead}}
  %0 = mhlo.convolution(%arg0, %arg1)
         dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
         window = {stride = [1, 1], pad = [[1, 1, 1], [1, 1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
         {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} :
       (tensor<1x8x8x207xf32>, tensor<3x3x207x16xf32>) -> tensor<1x8x8x16xf32>
  func.return %0 : tensor<1x8x8x16xf32>
}

// -----

// CHECK: module
// CHECK-SAME: mhlo.conv = #mhlo.conv<[b, 0, 1, f]x[0, 1, i, o]->[b, 1, 0, f]>
module attributes { mhlo.conv = #mhlo.conv<raw
      input_batch_dimension = 0,
      input_feature_dimension = 3,
      input_spatial_dimensions = [1, 2],
      kernel_input_feature_dimension = 2,
      kernel_output_feature_dimension = 3,
      kernel_spatial_dimensions = [0, 1],
      output_batch_dimension = 0,
      output_feature_dimension = 3,
      output_spatial_dimensions = [2, 1]>} {}

// -----

// CHECK-LABEL: func @convolution
// CHECK: mhlo.convolution
// CHECK-SAME: dim_numbers = [b, 1, 0, f]x[0, 1, i, o]->[b, 0, 1, f]
// CHECK-SAME{LITERAL}: window = {stride = [2, 1], pad = [[0, 1], [0, 1]], rhs_dilate = [1, 2]}
func.func @convolution(%arg0: tensor<2x2x3x4xf32>, %arg1: tensor<3x2x4x3xf32>) -> tensor<2x1x1x3xf32> {
  %0 = mhlo.convolution(%arg0, %arg1)
     dim_numbers = [b, 1, 0, f]x[0, 1, i, o]->[b, 0, 1, f],
     window = {stride = [2, 1], pad = [[0, 1], [0, 1]], rhs_dilate = [1, 2]}
     { batch_group_count = 1 : i64, feature_group_count = 1 : i64}
  : (tensor<2x2x3x4xf32>, tensor<3x2x4x3xf32>) -> tensor<2x1x1x3xf32>
  func.return %0 : tensor<2x1x1x3xf32>
}

// -----

// CHECK: module
// CHECK: mhlo.conv = #mhlo.conv<[b, 1, 0, f]x[0, 1, i, o]->[b, 0, 1, f]>
module attributes {
  mhlo.conv = #mhlo.conv<[b, 1, 0, f]x[0, 1, i, o]->[b, 0, 1, f]>
} {}

// -----

// CHECK: module
// CHECK: mhlo.conv = #mhlo.conv<[b, 1, 0, ?, f]x[?, 0, 1, i, o]->[b, ?, 0, 1, f]>
module attributes {
  mhlo.conv = #mhlo.conv<[b, 1, 0, ?, f]x[?, 0, 1, i, o]->[b, ?, 0, 1, f]>
} {}

// -----

module attributes {
  // expected-error@+1{{Unexpected dimension c, expecting b, f}}
  mhlo.conv = #mhlo.conv<[c, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f]>
} {}

// -----

module attributes {
  // expected-error@+1{{Unexpected dimension b, expecting i, o}}
  mhlo.conv = #mhlo.conv<[b, 0, 1, f]x[0, 1, b, o]->[b, 0, 1, f]>
} {}

// -----

module attributes {
  // expected-error@+1{{Unexpected dimension i, expecting o}}
  mhlo.conv = #mhlo.conv<[b, 0, 1, f]x[0, 1, i, i]->[b, 0, 1, f]>
} {}

// -----

module attributes {
  // expected-error@+1{{Expected dimensions f not specified}}
  mhlo.conv = #mhlo.conv<[b, 0, 1]x[0, 1, i, o]->[b, 0, 1, f]>
} {}

// -----

module attributes {
  // expected-error@+1{{Unexpected keyword b}}
  mhlo.conv = #mhlo.conv<[b, 0, 1, f]x[0, 1, i, o, b]->[b, 0, 1, f]>
} {}

// -----

module attributes {
  // expected-error@+1{{expected '['}}
  mhlo.conv = #mhlo.conv<{b, 0, 1, f}x[0, 1, i, o]->[b, 0, 1, f]>
} {}

// -----

module attributes {
  // expected-error@+1{{Expected spatial dimensions 0 not specified}}
  mhlo.conv = #mhlo.conv<[b, f, 1]x[o, 0, 1, i]->[f, b, 0, 1]>
} {}

// -----

module attributes {
  // expected-error@+1{{Duplicate entries for spatial dimension 1}}
  mhlo.conv = #mhlo.conv<[b, f, 1, 0, 1]x[o, 0, 1, i]->[f, b, 0, 1]>
} {}

// -----

module attributes {
  // expected-error@+1{{Unexpected dimension -2}}
  mhlo.conv = #mhlo.conv<[b, f, 1, -2]x[o, 0, 1, i]->[f, b, 0, 1]>
} {}

// -----

func.func @convolution(%arg0: tensor<2x2x3x4xf32>, %arg1: tensor<3x5x5x3xf32>) -> tensor<3x5x5x4xf32> {
  // expected-error@+3{{Expected array with 2 elements, got 3 elements instead}}
  %0 = mhlo.convolution(%arg0, %arg1)
     dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
     window = {stride = [2, 1], pad = [[0, 1, 2], [0, 1]], rhs_dilate = [1, 2]}
     { batch_group_count = 1 : i64, feature_group_count = 1 : i64}
  : (tensor<2x2x3x4xf32>, tensor<3x5x5x3xf32>) -> tensor<3x5x5x4xf32>
  func.return %0 : tensor<3x5x5x4xf32>
}

// -----

func.func @convolution(%arg0: tensor<2x2x3x4xf32>, %arg1: tensor<3x5x5x3xf32>) -> tensor<3x5x5x4xf32> {
  // expected-error@+3{{Unexpected keyword stide}}
  %0 = mhlo.convolution(%arg0, %arg1)
     dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
     window = {stide = [2, 1], pad = [[0, 1], [0, 1]], rhs_dilate = [1, 2]}
     { batch_group_count = 1 : i64, feature_group_count = 1 : i64}
  : (tensor<2x2x3x4xf32>, tensor<3x5x5x3xf32>) -> tensor<3x5x5x4xf32>
  func.return %0 : tensor<3x5x5x4xf32>
}
// -----

func.func @convolution(%arg0: tensor<2x2x3x4xf32>, %arg1: tensor<3x5x5x3xf32>) -> tensor<3x5x5x4xf32> {
  // expected-error@+3{{expected integer value}}
  %0 = mhlo.convolution(%arg0, %arg1)
     dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
     window = {stride = [2, b], pad = [[0, 1], [0, 1]], rhs_dilate = [1, 2]}
     { batch_group_count = 1 : i64, feature_group_count = 1 : i64}
  : (tensor<2x2x3x4xf32>, tensor<3x5x5x3xf32>) -> tensor<3x5x5x4xf32>
  func.return %0 : tensor<3x5x5x4xf32>
}
// -----

func.func @convolution(%arg0: tensor<2x2x3x4xf32>, %arg1: tensor<3x5x5x3xf32>) -> tensor<3x5x5x4xf32> {
  // expected-error@+3{{Unexpected keyword stride}}
  %0 = mhlo.convolution(%arg0, %arg1)
     dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
     window = {stride = [2, 1], pad = [[0, 1], [0, 1]], rhs_dilate = [1, 2], stride=[2,1]}
     { batch_group_count = 1 : i64, feature_group_count = 1 : i64}
  : (tensor<2x2x3x4xf32>, tensor<3x5x5x3xf32>) -> tensor<3x5x5x4xf32>
  func.return %0 : tensor<3x5x5x4xf32>
}

// -----

// Test custom attribute printing/parsing.
// We really just need one op as holder, use module: this is the simplest top-level.

// CHECK: module
// CHECK-SAME: mhlo.scatter = #mhlo.scatter<>
module attributes{mhlo.scatter = #mhlo.scatter<>} {}

// -----

// CHECK: module
// CHECK-SAME: mhlo.scatter = #mhlo.scatter<update_window_dims = [1], inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 2], index_vector_dim = 1>
module attributes{
 mhlo.scatter = #mhlo.scatter<
  index_vector_dim = 1,
  scatter_dims_to_operand_dims = [0, 2],
  inserted_window_dims = [0, 1],
  update_window_dims = [1]
 >} {}

// -----

// CHECK: module
// CHECK-SAME: mhlo.scatter = #mhlo.scatter<update_window_dims = [1], inserted_window_dims = [0, 1]>
module attributes{
 mhlo.scatter = #mhlo.scatter<
  inserted_window_dims = [0, 1],
  update_window_dims = [1]
 >} {}

// -----

module attributes{
 mhlo.scatter = #mhlo.scatter<
  index_vector_dim = 1,
  // expected-error@+2 {{duplicated `index_vector_dim` entry}}
  // expected-error@+1 {{failed parsing scatter dimension numbers}}
  index_vector_dim = 1,
 >} {}

// -----

// CHECK: module
// CHECK-SAME: mhlo.gather = #mhlo.gather<>
module attributes{mhlo.gather = #mhlo.gather<>} {}

// -----

// CHECK: module
// CHECK-SAME: mhlo.gather = #mhlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>
module attributes{
 mhlo.gather = #mhlo.gather<
   collapsed_slice_dims = [0],
   index_vector_dim = 1,
   offset_dims = [1],
   start_index_map = [0],
 >} {}

// -----

module attributes{
 mhlo.gather = #mhlo.gather<
   collapsed_slice_dims = [0],
   // expected-error @+2 {{failed parsing gather dimension numbers}}
   // expected-error @+1 {{duplicated `collapsed_slice_dims` entry}}
   collapsed_slice_dims = [0],
 >} {}

// -----

// CHECK: module
// CHECK-SAME: mhlo.dot = #mhlo.dot<
// CHECK-SAME:       lhs_batching_dimensions = [0],
// CHECK-SAME:       rhs_batching_dimensions = [1],
// CHECK-SAME:       lhs_contracting_dimensions = [2],
// CHECK-SAME:       rhs_contracting_dimensions = [3]
// CHECK-SAME:     >
module attributes {
  mhlo.dot = #mhlo.dot<
      lhs_batching_dimensions = [0],
      rhs_batching_dimensions = [1],
      lhs_contracting_dimensions = [2],
      rhs_contracting_dimensions = [3]
  >} {}

// -----

// CHECK: module
// CHECK-SAME: mhlo.dot = #mhlo.dot<
// CHECK-SAME:       lhs_batching_dimensions = [0],
// CHECK-SAME:       rhs_batching_dimensions = [1],
// CHECK-SAME:       lhs_contracting_dimensions = [2],
// CHECK-SAME:       rhs_contracting_dimensions = [3]
// CHECK-SAME:     >
module attributes {
  mhlo.dot = #mhlo.dot<
      lhs_batching_dimensions = [0],
      rhs_batching_dimensions = [1],
      lhs_contracting_dimensions = [2],
      rhs_contracting_dimensions = [3],
  >} {}

// -----

// CHECK: module
// CHECK-SAME: mhlo.dot = #mhlo.dot<
// CHECK-SAME:       lhs_batching_dimensions = [0],
// CHECK-SAME:       rhs_batching_dimensions = [1],
// CHECK-SAME:       lhs_contracting_dimensions = [2],
// CHECK-SAME:       rhs_contracting_dimensions = [3]
// CHECK-SAME:     >
module attributes {
  mhlo.dot = #mhlo.dot<
      rhs_batching_dimensions = [1],
      rhs_contracting_dimensions = [3],
      lhs_contracting_dimensions = [2],
      lhs_batching_dimensions = [0],
  >} {}

// -----

module attributes {
  mhlo.dot = #mhlo.dot<
      rhs_batching_dimensions = [1],
      // expected-error@+2 {{duplicated `rhs_batching_dimensions` entry}}
      // expected-error@+1 {{failed parsing dot dimension numbers}}
      rhs_batching_dimensions = [3],
      lhs_contracting_dimensions = [2],
      lhs_batching_dimensions = [0],
  >} {}

// -----

module attributes {
  // expected-error@+3 {{expected '>'}}
  // expected-error@+3 {{failed parsing dot dimension numbers}}
  mhlo.dot = #mhlo.dot<
      rhs_batching_dimensions = [1]
      rhs_contracting_dimensions = [3]
      lhs_contracting_dimensions = [2]
      lhs_batching_dimensions = [0]
  >} {}


// -----

module attributes {
  // expected-error@+2 {{expected one of: `lhs_batching_dimensions`, `rhs_batching_dimensions`, `lhs_contracting_dimensions`, `rhs_contracting_dimensions`}}
  // expected-error@+1 {{failed parsing dot dimension numbers}}
  mhlo.dot = #mhlo.dot<foo = [0]>
} {}

// -----

module attributes {
  mhlo.dot = #mhlo.dot<
      rhs_batching_dimensions = [1],
      rhs_contracting_dimensions = [3],
      lhs_contracting_dimensions = [2],
      lhs_batching_dimensions = [0],
      // expected-error@+2 {{expected one of: `lhs_batching_dimensions`, `rhs_batching_dimensions`, `lhs_contracting_dimensions`, `rhs_contracting_dimensions`}}
      // expected-error@+1 {{failed parsing dot dimension numbers}}
      foo = [0]
  >} {}

// -----

// CHECK-LABEL: func @test_alias_attribute
// CHECK-SAME:  mhlo.result_alias = #mhlo.result_alias<
// CHECK-SAME:       tuple_indices = [1, 1],
// CHECK-SAME:       result_index = [2, 0, 1],
// CHECK-SAME:       must_alias>
func.func @test_alias_attribute (%arg0: tuple<i32, tuple<i32, tensor<3xf32>>> {mhlo.result_alias = #mhlo.result_alias<
      tuple_indices = [1, 1],
      result_index = [2, 0, 1],
      must_alias>}
    ) -> (i32, i32, tuple<tuple<i32, tensor<3xf32>>>) {
  %0:3 = "Test.Op"() : () -> (i32, i32, tuple<tuple<i32, tensor<3xf32>>>)
  func.return %0#0, %0#1, %0#2 : i32, i32, tuple<tuple<i32, tensor<3xf32>>>
}

// -----

// CHECK-LABEL: func @test_alias_dynamic_dimension
// CHECK-SAME:  mhlo.result_alias = #mhlo.result_alias<result_index = [2]>
func.func @test_alias_dynamic_dimension (%arg0: tensor<?xf32> {mhlo.result_alias = #mhlo.result_alias<result_index = [2]>}
    ) -> (i32, i32, tensor<2xf32>) {
  %0:3 = "Test.Op"() : () -> (i32, i32, tensor<2xf32>)
  func.return %0#0, %0#1, %0#2 : i32, i32, tensor<2xf32>
}

// -----

// CHECK-LABEL: func @test_may_alias_no_tuple
// CHECK-SAME:  mhlo.result_alias = #mhlo.result_alias<result_index = [2]>
func.func @test_may_alias_no_tuple (%arg0: tensor<2xf32> {mhlo.result_alias = #mhlo.result_alias<result_index = [2]>}
    ) -> (i32, i32, tensor<2xf32>) {
  %0:3 = "Test.Op"() : () -> (i32, i32, tensor<2xf32>)
  func.return %0#0, %0#1, %0#2 : i32, i32, tensor<2xf32>
}

// -----

// CHECK-LABEL: func @test_may_alias_arg_tuple
// CHECK-SAME:  mhlo.result_alias = #mhlo.result_alias<tuple_indices = [2, 0], result_index = [2]>
func.func @test_may_alias_arg_tuple (%arg0: tuple<i32, i32, tuple<tensor<2xf32>, i32>> {mhlo.result_alias = #mhlo.result_alias<tuple_indices = [2, 0], result_index = [2]>}
    ) -> (i32, i32, tensor<2xf32>) {
  %0:3 = "Test.Op"() : () -> (i32, i32, tensor<2xf32>)
  func.return %0#0, %0#1, %0#2 : i32, i32, tensor<2xf32>
}

// -----

// CHECK-LABEL: func @test_may_alias_result_tuple
// CHECK-SAME:  mhlo.result_alias = #mhlo.result_alias<result_index = [2, 1, 2]>
func.func @test_may_alias_result_tuple (%arg0: tensor<2xf32> {mhlo.result_alias = #mhlo.result_alias<result_index = [2, 1, 2]>}
    ) -> (i32, i32, tuple<i32, tuple<i32, i32, tensor<2xf32>>>, i32) {
  %0:4 = "Test.Op"() : () -> (i32, i32, tuple<i32, tuple<i32, i32, tensor<2xf32>>>, i32)
  func.return %0#0, %0#1, %0#2, %0#3 : i32, i32, tuple<i32, tuple<i32, i32, tensor<2xf32>>>, i32
}

// -----

// expected-error@+1 {{attribute "mhlo.result_alias" can only be used on function-like operations}}
module attributes {mhlo.result_alias = #mhlo.result_alias<result_index = [2, 3]>} {}

// -----

// expected-error @+2 {{expected at least 1 element(s), found 0}}
// expected-error@+1 {{failed parsing argument-result alias attribute}}
func.func @error_empty_result_index (%arg0: tensor<2xf32> {mhlo.result_alias = #mhlo.result_alias<result_index = []>}
    ) -> (tensor<2xf32>) {
  func.return %arg0 : tensor<2xf32>
}

// -----

// expected-error@+1 {{attribute "mhlo.result_alias" expects all argument and result indices to be >= 0}}
func.func @error_negative_arg_tuple_index (%arg0: tensor<2xf32> {mhlo.result_alias = #mhlo.result_alias<tuple_indices = [0, -1], result_index = [0]>}
    ) -> (tensor<2xf32>) {
  func.return %arg0 : tensor<2xf32>
}

// -----

// expected-error@+1 {{attribute "mhlo.result_alias" expects all argument and result indices to be >= 0}}
func.func @error_negative_result_index (%arg0: tensor<2xf32> {mhlo.result_alias = #mhlo.result_alias<result_index = [-1]>}
    ) -> (tensor<2xf32>) {
  func.return %arg0 : tensor<2xf32>
}

// -----

// expected-error@+1 {{attribute "mhlo.result_alias" expects all argument and result indices to be >= 0}}
func.func @error_negative_result_tuple_index (%arg0: tensor<2xf32> {mhlo.result_alias = #mhlo.result_alias<result_index = [0, -1]>}
    ) -> (tensor<2xf32>) {
  func.return %arg0 : tensor<2xf32>
}

// -----

// expected-error@+1 {{attribute "mhlo.result_alias" result index is out of range, must be <1}}
func.func @error_result_index_out_of_range (%arg0: tensor<2xf32> {mhlo.result_alias = #mhlo.result_alias<result_index = [1]>}
    ) -> (tensor<2xf32>) {
  func.return %arg0 : tensor<2xf32>
}

// -----

// expected-error@+1 {{attribute "mhlo.result_alias" argument tuple indices are invalid}}
func.func @error_invalid_argument_tuple_indices (%arg0: tuple<i32, tensor<2xf32>> {mhlo.result_alias = #mhlo.result_alias<tuple_indices = [2], result_index = [0]>}
    ) -> (tensor<2xf32>) {
  %0 = "Test.Op"() : () -> (tensor<2xf32>)
  func.return %0 : tensor<2xf32>
}

// -----

// expected-error@+1 {{attribute "mhlo.result_alias" aliases do not have compatible types, 'tensor<2xf32>' vs. 'tensor<1xf32>'}}
func.func @error_incompatible_alias_shapes (%arg0: tensor<2xf32> {mhlo.result_alias = #mhlo.result_alias<result_index = [0, 1]>}
    ) -> (tuple<i32, tensor<1xf32>>) {
  %0 = "Test.Op"() : () -> (tuple<i32, tensor<1xf32>>)
  func.return %0 : tuple<i32, tensor<1xf32>>
}

// -----

// expected-error@+1 {{attribute "mhlo.result_alias" aliases do not have compatible types, 'tensor<2xf32>' vs. 'tensor<2xi32>'}}
func.func @error_incompatible_alias_element_types (%arg0: tensor<2xf32> {mhlo.result_alias = #mhlo.result_alias<result_index = [0, 1]>}
    ) -> (tuple<i32, tensor<2xi32>>) {
  %0 = "Test.Op"() : () -> (tuple<i32, tensor<2xi32>>)
  func.return %0 : tuple<i32, tensor<2xi32>>
}

// -----

// mhlo.batch_norm_training

// CHECK-LABEL: @batch_norm_train
func.func @batch_norm_train(%input: tensor<2x2x2x2xf32>, %scale: tensor<2xf32>, %offset: tensor<2xf32>) -> tensor<2x2x2x2xf32> {
  %0:3 = "mhlo.batch_norm_training" (%input, %scale, %offset) {epsilon = 0.001 : f32, feature_index = 1 : i64} : (tensor<2x2x2x2xf32>, tensor<2xf32>, tensor<2xf32>) -> (tensor<2x2x2x2xf32>, tensor<2xf32>, tensor<2xf32>)
  func.return %0#0 : tensor<2x2x2x2xf32>
}

// -----

// CHECK-LABEL: @batch_norm_train_dynamic
func.func @batch_norm_train_dynamic(%input: tensor<?x?x2x2xf32>, %scale: tensor<2xf32>, %offset: tensor<2xf32>) -> tensor<?x?x2x2xf32> {
  %0:3 = "mhlo.batch_norm_training" (%input, %scale, %offset) {
    epsilon = 0.001 : f32, feature_index = 1 : i64
  } : (tensor<?x?x2x2xf32>, tensor<2xf32>, tensor<2xf32>) -> (tensor<?x?x2x2xf32>, tensor<2xf32>, tensor<2xf32>)
  func.return %0#0 : tensor<?x?x2x2xf32>
}

// -----

func.func @error_batch_norm_train(%input: tensor<2x2x2x2xf32>, %scale: tensor<2xf32>, %offset: tensor<2xf32>) -> tensor<2x2x2x2xf32> {
  // expected-error@+1 {{expects featureIndex to be smaller than the rank of multi-dimensional operands; got featureIndex 4, and rank 4.}}
  %0:3 = "mhlo.batch_norm_training" (%input, %scale, %offset) {epsilon = 0.001 : f32, feature_index = 4 : i64} : (tensor<2x2x2x2xf32>, tensor<2xf32>, tensor<2xf32>) -> (tensor<2x2x2x2xf32>, tensor<2xf32>, tensor<2xf32>)
  func.return %0#0 : tensor<2x2x2x2xf32>
}

// -----

func.func @error_batch_norm_train(%input: tensor<2x2x2x2xf32>, %scale: tensor<2xf32>, %offset: tensor<2xf32>) -> tensor<2x2x2x2xf32> {
  // expected-error@+1 {{expects featureIndex to be a non-negative number, got -1.}}
  %0:3 = "mhlo.batch_norm_training" (%input, %scale, %offset) {epsilon = 0.001 : f32, feature_index = -1 : i64} : (tensor<2x2x2x2xf32>, tensor<2xf32>, tensor<2xf32>) -> (tensor<2x2x2x2xf32>, tensor<2xf32>, tensor<2xf32>)
  func.return %0#0 : tensor<2x2x2x2xf32>
}

// -----

func.func @error_batch_norm_train(%input: tensor<2x2x2x2xf32>, %scale: tensor<3xf32>, %offset: tensor<3xf32>) -> tensor<2x2x2x2xf32> {
  // expected-error@+1 {{expects the size of single-dimensional operands to be compatible with feature count, but the size of single-dimensional operands is 3 and the feature count is 2.}}
  %0:3 = "mhlo.batch_norm_training" (%input, %scale, %offset) {epsilon = 0.001 : f32, feature_index = 3 : i64} : (tensor<2x2x2x2xf32>, tensor<3xf32>, tensor<3xf32>) -> (tensor<2x2x2x2xf32>, tensor<3xf32>, tensor<3xf32>)
  func.return %0#0 : tensor<2x2x2x2xf32>
}

// -----

// mhlo.batch_norm_inference

// CHECK-LABEL: @batch_norm_inference
func.func @batch_norm_inference(%input: tensor<4x256xf32>, %scale: tensor<256xf32>, %offset: tensor<256xf32>, %mean: tensor<256xf32>, %variance: tensor<256xf32>) -> (tensor<4x256xf32>) {
  %0 = "mhlo.batch_norm_inference" (%input, %scale, %offset, %mean, %variance) {epsilon = 1.001000e-05 : f32, feature_index = 1 : i64} :
      (tensor<4x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>,
        tensor<256xf32>) -> tensor<4x256xf32>
  func.return %0 : tensor<4x256xf32>
}

// -----

// CHECK-LABEL: @batch_norm_inference_dynamic
func.func @batch_norm_inference_dynamic(%input: tensor<4x?xf32>, %scale: tensor<?xf32>, %offset: tensor<?xf32>, %mean: tensor<?xf32>, %variance: tensor<?xf32>) -> (tensor<4x?xf32>) {
  %0 = "mhlo.batch_norm_inference" (%input, %scale, %offset, %mean, %variance) {
    epsilon = 1.001000e-05 : f32, feature_index = 1 : i64
  } : (tensor<4x?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>) -> tensor<4x?xf32>
  func.return %0 : tensor<4x?xf32>
}

// -----

func.func @error_batch_norm_inference(%input: tensor<4x256xf32>, %scale: tensor<256xf32>, %offset: tensor<256xf32>, %mean: tensor<256xf32>, %variance: tensor<256xf32>) -> (tensor<4x256xf32>) {
  // expected-error@+1 {{expects featureIndex to be smaller than the rank of multi-dimensional operands; got featureIndex 2, and rank 2.}}
  %0 = "mhlo.batch_norm_inference" (%input, %scale, %offset, %mean, %variance) {epsilon = 1.001000e-05 : f32, feature_index = 2 : i64} :
      (tensor<4x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>,
        tensor<256xf32>) -> tensor<4x256xf32>
  func.return %0 : tensor<4x256xf32>
}

// -----

func.func @error_batch_norm_inference(%input: tensor<4x256xf32>, %scale: tensor<256xf32>, %offset: tensor<256xf32>, %mean: tensor<256xf32>, %variance: tensor<256xf32>) -> (tensor<4x256xf32>) {
  // expected-error@+1 {{expects featureIndex to be a non-negative number, got -1.}}
  %0 = "mhlo.batch_norm_inference" (%input, %scale, %offset, %mean, %variance) {epsilon = 1.001000e-05 : f32, feature_index = -1 : i64} :
      (tensor<4x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>,
        tensor<256xf32>) -> tensor<4x256xf32>
  func.return %0 : tensor<4x256xf32>
}

// -----

func.func @error_batch_norm_inference(%input: tensor<4x256xf32>, %scale: tensor<25xf32>, %offset: tensor<25xf32>, %mean: tensor<25xf32>, %variance: tensor<25xf32>) -> (tensor<4x256xf32>) {
  // expected-error@+1 {{expects the size of single-dimensional operands to be compatible with feature count, but the size of single-dimensional operands is 25 and the feature count is 256.}}
  %0 = "mhlo.batch_norm_inference" (%input, %scale, %offset, %mean, %variance) {epsilon = 1.001000e-05 : f32, feature_index = 1 : i64} :
      (tensor<4x256xf32>, tensor<25xf32>, tensor<25xf32>, tensor<25xf32>,
        tensor<25xf32>) -> tensor<4x256xf32>
  func.return %0 : tensor<4x256xf32>
}

// -----

// mhlo.batch_norm_grad

// CHECK-LABEL: @batch_norm_grad
func.func @batch_norm_grad(%input: tensor<2x2x2x2xf32>, %scale: tensor<2xf32>, %mean: tensor<2xf32>, %variance: tensor<2xf32>, %grad_output: tensor<2x2x2x2xf32>) -> tensor<2x2x2x2xf32> {
  %0:3 = "mhlo.batch_norm_grad" (%input, %scale, %mean, %variance, %grad_output) {epsilon = 0.001 : f32, feature_index = 0 : i64} : (tensor<2x2x2x2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2x2x2x2xf32>) -> (tensor<2x2x2x2xf32>, tensor<2xf32>, tensor<2xf32>)
  func.return %0#0 : tensor<2x2x2x2xf32>
}

// -----

func.func @batch_norm_grad_dynamic(%input: tensor<?x2x2x2xf32>, %scale: tensor<?xf32>, %mean: tensor<?xf32>, %variance: tensor<?xf32>, %grad_output: tensor<?x2x2x2xf32>) -> tensor<?x2x2x2xf32> {
  %0:3 = "mhlo.batch_norm_grad" (%input, %scale, %mean, %variance, %grad_output) {
    epsilon = 0.001 : f32, feature_index = 0 : i64
  } : (tensor<?x2x2x2xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?xf32>, tensor<?x2x2x2xf32>) -> (tensor<?x2x2x2xf32>, tensor<?xf32>, tensor<?xf32>)
  func.return %0#0 : tensor<?x2x2x2xf32>
}

// -----

func.func @error_batch_norm_grad(%input: tensor<2x2x2x2xf32>, %scale: tensor<2xf32>, %mean: tensor<2xf32>, %variance: tensor<2xf32>, %grad_output: tensor<2x2x2x2xf32>) -> tensor<2x2x2x2xf32> {
  // expected-error@+1 {{expects featureIndex to be smaller than the rank of multi-dimensional operands; got featureIndex 4, and rank 4.}}
  %0:3 = "mhlo.batch_norm_grad" (%input, %scale, %mean, %variance, %grad_output) {epsilon = 0.001 : f32, feature_index = 4 : i64} : (tensor<2x2x2x2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2x2x2x2xf32>) -> (tensor<2x2x2x2xf32>, tensor<2xf32>, tensor<2xf32>)
  func.return %0#0 : tensor<2x2x2x2xf32>
}

// -----

func.func @error_batch_norm_grad(%input: tensor<2x2x2x2xf32>, %scale: tensor<2xf32>, %mean: tensor<2xf32>, %variance: tensor<2xf32>, %grad_output: tensor<2x2x2x2xf32>) -> tensor<2x2x2x2xf32> {
  // expected-error@+1 {{expects featureIndex to be a non-negative number, got -1.}}
  %0:3 = "mhlo.batch_norm_grad" (%input, %scale, %mean, %variance, %grad_output) {epsilon = 0.001 : f32, feature_index = -1 : i64} : (tensor<2x2x2x2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2x2x2x2xf32>) -> (tensor<2x2x2x2xf32>, tensor<2xf32>, tensor<2xf32>)
  func.return %0#0 : tensor<2x2x2x2xf32>
}

// -----

func.func @error_batch_norm_grad(%input: tensor<2x2x2x2xf32>, %scale: tensor<4xf32>, %mean: tensor<4xf32>, %variance: tensor<4xf32>, %grad_output: tensor<2x2x2x2xf32>) -> tensor<2x2x2x2xf32> {
  // expected-error@+1 {{expects the size of single-dimensional operands to be compatible with feature count, but the size of single-dimensional operands is 4 and the feature count is 2.}}
  %0:3 = "mhlo.batch_norm_grad" (%input, %scale, %mean, %variance, %grad_output) {epsilon = 0.001 : f32, feature_index = 0 : i64} : (tensor<2x2x2x2xf32>, tensor<4xf32>, tensor<4xf32>, tensor<4xf32>, tensor<2x2x2x2xf32>) -> (tensor<2x2x2x2xf32>, tensor<4xf32>, tensor<4xf32>)
  func.return %0#0 : tensor<2x2x2x2xf32>
}

// -----

func.func @error_batch_norm_grad(%input: tensor<2x2x2x2xf32>, %scale: tensor<4xf32>, %mean: tensor<2xf32>, %variance: tensor<2xf32>, %grad_output: tensor<2x2x2x2xf32>) -> tensor<2x2x2x2xf32> {
  // expected-error@+1 {{expects single-dimensional operands to have compatible shapes}}
  %0:3 = "mhlo.batch_norm_grad" (%input, %scale, %mean, %variance, %grad_output) {epsilon = 0.001 : f32, feature_index = 0 : i64} : (tensor<2x2x2x2xf32>, tensor<4xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2x2x2x2xf32>) -> (tensor<2x2x2x2xf32>, tensor<2xf32>, tensor<2xf32>)
  func.return %0#0 : tensor<2x2x2x2xf32>
}

// -----

func.func @error_batch_norm_grad(%input: tensor<2x2x2x2xi32>, %scale: tensor<2xf32>, %mean: tensor<2xf32>, %variance: tensor<2xf32>, %grad_output: tensor<2x2x2x2xf32>) -> tensor<2x2x2x2xf32> {
  // expected-error@+1 {{operand #0 must be ranked tensor of f8E4M3FN type or f8E5M2 type or 16-bit float or 32-bit float or 64-bit float or bfloat16 type values, but got 'tensor<2x2x2x2xi32>'}}
  %0:3 = "mhlo.batch_norm_grad" (%input, %scale, %mean, %variance, %grad_output) {epsilon = 0.001 : f32, feature_index = 0 : i64} : (tensor<2x2x2x2xi32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2x2x2x2xf32>) -> (tensor<2x2x2x2xf32>, tensor<2xf32>, tensor<2xf32>)
  func.return %0#0 : tensor<2x2x2x2xf32>
}

// -----

func.func @error_batch_norm_grad(%input: tensor<2x2x2x2xf32>, %scale: tensor<2xf32>, %mean: tensor<2xf32>, %variance: tensor<2xf32>, %grad_output: tensor<2x2x2xf32>) -> tensor<2x2x2x2xf32> {
  // expected-error@+1 {{expects multi-dimensional operands to have compatible shapes}}
  %0:3 = "mhlo.batch_norm_grad" (%input, %scale, %mean, %variance, %grad_output) {epsilon = 0.001 : f32, feature_index = 0 : i64} : (tensor<2x2x2x2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2x2x2xf32>) -> (tensor<2x2x2x2xf32>, tensor<2xf32>, tensor<2xf32>)
  func.return %0#0 : tensor<2x2x2x2xf32>
}

// -----

func.func @error_batch_norm_grad(%input: tensor<2x2x2x2xf32>, %scale: tensor<2xf32>, %mean: tensor<2xf32>, %variance: tensor<2xf32>, %grad_output: tensor<2x2x2x2xf32>) -> tensor<2x2x2x2xf32> {
  // expected-error@+1 {{failed to verify that all of {operand, grad_operand, grad_scale, grad_offset} have same element type}}
  %0:3 = "mhlo.batch_norm_grad" (%input, %scale, %mean, %variance, %grad_output) {epsilon = 0.001 : f32, feature_index = 0 : i64} : (tensor<2x2x2x2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2x2x2x2xf32>) -> (tensor<2x2x2x2xf32>, tensor<2xf64>, tensor<2xf64>)
  func.return %0#0 : tensor<2x2x2x2xf32>
}

// -----

func.func @error_batch_norm_grad(%input: tensor<2x2x2x2xf32>, %scale: tensor<2xf32>, %mean: tensor<2xf32>, %variance: tensor<2xf32>, %grad_output: tensor<2x2x2x2xf32>) -> tensor<2x2x2x2xf64> {
  // expected-error@+1 {{failed to verify that all of {operand, grad_operand, grad_scale, grad_offset} have same element type}}
  %0:3 = "mhlo.batch_norm_grad" (%input, %scale, %mean, %variance, %grad_output) {epsilon = 0.001 : f32, feature_index = 0 : i64} : (tensor<2x2x2x2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2x2x2x2xf32>) -> (tensor<2x2x2x2xf64>, tensor<2xf32>, tensor<2xf32>)
  func.return %0#0 : tensor<2x2x2x2xf64>
}

// -----

func.func @error_batch_norm_grad(%input: tensor<2x2x2x2xf32>, %scale: tensor<2xf32>, %mean: tensor<2xf32>, %variance: tensor<2xf32>, %grad_output: tensor<2x2x2x2xf32>) -> tensor<2x2x2x2xf32> {
  // expected-error@+1 {{result #1 must be 1D tensor of f8E4M3FN type or f8E5M2 type or 16-bit float or 32-bit float or 64-bit float or bfloat16 type values, but got 'tensor<2x2xf32>'}}
  %0:3 = "mhlo.batch_norm_grad" (%input, %scale, %mean, %variance, %grad_output) {epsilon = 0.001 : f32, feature_index = 0 : i64} : (tensor<2x2x2x2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2x2x2x2xf32>) -> (tensor<2x2x2x2xf32>, tensor<2x2xf32>, tensor<2xf32>)
  func.return %0#0 : tensor<2x2x2x2xf32>
}

// -----

func.func @error_batch_norm_grad(%input: tensor<*xf32>, %scale: tensor<2xf32>, %mean: tensor<2xf32>, %variance: tensor<2xf32>, %grad_output: tensor<2x2x2x2xf32>) -> tensor<*xf32> {
  // expected-error@+1 {{operand #0 must be ranked tensor of f8E4M3FN type or f8E5M2 type or 16-bit float or 32-bit float or 64-bit float or bfloat16 type values, but got 'tensor<*xf32>'}}
  %0:3 = "mhlo.batch_norm_grad" (%input, %scale, %mean, %variance, %grad_output) {epsilon = 0.001 : f32, feature_index = 0 : i64} : (tensor<*xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2x2x2x2xf32>) -> (tensor<*xf32>, tensor<2xf32>, tensor<2xf32>)
  func.return %0#0 : tensor<*xf32>
}

// -----

// Test rng_get_and_update_state_op
// CHECK-LABEL: xla.rng_get_and_update_state
func.func @xla.rng_get_and_update_state() -> tensor<2xui64> {
  %result = mhlo.xla.rng_get_and_update_state {delta = 1 : i64}
  func.return %result : tensor<2xui64>
}
// CHECK: mhlo.xla.rng_get_and_update_state

// -----

// CHECK-LABEL: @fft
func.func @fft(%arg0: tensor<3x9xcomplex<f32>>) -> tensor<3x9xcomplex<f32>> {
  %0 = "mhlo.fft"(%arg0) { fft_length = dense<9> : tensor<1xi64>, fft_type = #mhlo<fft_type FFT> } : (tensor<3x9xcomplex<f32>>) -> tensor<3x9xcomplex<f32>>
  func.return %0 : tensor<3x9xcomplex<f32>>
}

// -----

// CHECK-LABEL: @ifft
func.func @ifft(%arg0: tensor<3x9xcomplex<f32>>) -> tensor<3x9xcomplex<f32>> {
  %0 = "mhlo.fft"(%arg0) { fft_length = dense<9> : tensor<1xi64>, fft_type = #mhlo<fft_type IFFT> } : (tensor<3x9xcomplex<f32>>) -> tensor<3x9xcomplex<f32>>
  func.return %0 : tensor<3x9xcomplex<f32>>
}

// -----

// CHECK-LABEL: @rfft
func.func @rfft(%arg0: tensor<3x9xf32>) -> tensor<3x5xcomplex<f32>> {
  %0 = "mhlo.fft"(%arg0) { fft_length = dense<9> : tensor<1xi64>, fft_type = #mhlo<fft_type RFFT> } : (tensor<3x9xf32>) -> tensor<3x5xcomplex<f32>>
  func.return %0 : tensor<3x5xcomplex<f32>>
}

// -----

// CHECK-LABEL: @irfft
func.func @irfft(%arg0: tensor<3x9xcomplex<f32>>) -> tensor<3x16xf32> {
  %0 = "mhlo.fft"(%arg0) { fft_length = dense<16> : tensor<1xi64>, fft_type = #mhlo<fft_type IRFFT> } : (tensor<3x9xcomplex<f32>>) -> tensor<3x16xf32>
  func.return %0 : tensor<3x16xf32>
}

// -----

// CHECK-LABEL: @rfft_unranked
func.func @rfft_unranked(%arg0: tensor<*xf32>) -> tensor<*xcomplex<f32>> {
  %0 = "mhlo.fft"(%arg0) { fft_length = dense<9> : tensor<1xi64>, fft_type = #mhlo<fft_type RFFT> } : (tensor<*xf32>) -> tensor<*xcomplex<f32>>
  func.return %0 : tensor<*xcomplex<f32>>
}

// -----

func.func @rfft_not_float32or64(%arg0: tensor<3x9xf16>) -> tensor<3x5xcomplex<f32>> {
  // expected-error@+1 {{RFFT requires f32 or f64 input type, but is given 'f16'.}}
  %0 = "mhlo.fft"(%arg0) { fft_length = dense<9> : tensor<1xi64>, fft_type = #mhlo<fft_type RFFT> } : (tensor<3x9xf16>) -> tensor<3x5xcomplex<f32>>
  func.return %0 : tensor<3x5xcomplex<f32>>
}

// -----

func.func @fft_invalid_rank(%arg0: tensor<3x9xf32>) -> tensor<3x9xcomplex<f32>> {
  // expected-error@+1 {{rank must be between 1 and 3, but got 4.}}
  %0 = "mhlo.fft"(%arg0) { fft_length = dense<9> : tensor<4xi64>, fft_type = #mhlo<fft_type RFFT> } : (tensor<3x9xf32>) -> tensor<3x9xcomplex<f32>>
  func.return %0 : tensor<3x9xcomplex<f32>>
}

// -----

func.func @fft_rank_mismatch(%arg0: tensor<3x9xf32>) -> tensor<3x9xcomplex<f32>> {
  // expected-error@+1 {{operand rank must not be less than fft rank of 3 for operand of type 'tensor<3x9xf32>'}}
  %0 = "mhlo.fft"(%arg0) { fft_length = dense<9> : tensor<3xi64>, fft_type = #mhlo<fft_type RFFT> } : (tensor<3x9xf32>) -> tensor<3x9xcomplex<f32>>
  func.return %0 : tensor<3x9xcomplex<f32>>
}

// -----

func.func @rfft_invalid_dim(%arg0: tensor<3x9xf32>) -> tensor<3x9xcomplex<f32>> {
  // expected-error@+1 {{RFFT requires innermost dimensions to be compatible with fft_length. Got: 3, 9 but wanted 9, 9.}}
  %0 = "mhlo.fft"(%arg0) { fft_length = dense<9> : tensor<2xi64>, fft_type = #mhlo<fft_type RFFT> } : (tensor<3x9xf32>) -> tensor<3x9xcomplex<f32>>
  func.return %0 : tensor<3x9xcomplex<f32>>
}

// -----

func.func @irfft_invalid_dim(%arg0: tensor<3x9xcomplex<f32>>) -> tensor<3x9xf32> {
  // expected-error@+1 {{IRFFT requires non-final dimensions to be compatible with fft_length. Got: 3, 9 but wanted 9, 9, and 3 != 9.}}
  %0 = "mhlo.fft"(%arg0) { fft_length = dense<9> : tensor<2xi64>, fft_type = #mhlo<fft_type IRFFT> } : (tensor<3x9xcomplex<f32>>) -> tensor<3x9xf32>
  func.return %0 : tensor<3x9xf32>
}

// -----

func.func @irfft_invalid_dim(%arg0: tensor<3x9xcomplex<f32>>) -> tensor<3x9xf32> {
  // expected-error@+1 {{IRFFT requires innermost dimension to be compatible with fft_length[-1]/2+1. Got: 9 but fft_length is 9.}}
  %0 = "mhlo.fft"(%arg0) { fft_length = dense<9> : tensor<1xi64>, fft_type = #mhlo<fft_type IRFFT> } : (tensor<3x9xcomplex<f32>>) -> tensor<3x9xf32>
  func.return %0 : tensor<3x9xf32>
}

// -----

func.func @irfft_invalid_elt(%arg0: tensor<3x9xf32>) -> tensor<3x9xcomplex<f32>> {
  // expected-error@+1 {{FFT/IFFT/IRFFT take a complex tensor as input, but is given 'tensor<3x9xf32>'}}
  %0 = "mhlo.fft"(%arg0) { fft_length = dense<16> : tensor<1xi64>, fft_type = #mhlo<fft_type IRFFT> } : (tensor<3x9xf32>) -> tensor<3x9xcomplex<f32>>
  func.return %0 : tensor<3x9xcomplex<f32>>
}

// -----

func.func @irfft_invalid_ret_elt(%arg0: tensor<3x9xcomplex<f32>>) -> tensor<3x16xcomplex<f32>> {
  // expected-error@+1 {{inferred type(s) 'tensor<3x16xf32>' are incompatible with return type(s) of operation 'tensor<3x16xcomplex<f32>>'}}
  %0 = "mhlo.fft"(%arg0) { fft_length = dense<16> : tensor<1xi64>, fft_type = #mhlo<fft_type IRFFT> } : (tensor<3x9xcomplex<f32>>) -> tensor<3x16xcomplex<f32>>
  func.return %0 : tensor<3x16xcomplex<f32>>
}

// -----

func.func @rfft_invalid_ret_elt(%arg0: tensor<3x9xf32>) -> tensor<3x9xf32> {
  // expected-error@+1 {{inferred type(s) 'tensor<3x5xcomplex<f32>>' are incompatible with return type(s) of operation 'tensor<3x9xf32>'}}
  %0 = "mhlo.fft"(%arg0) { fft_length = dense<9> : tensor<1xi64>, fft_type = #mhlo<fft_type RFFT> } : (tensor<3x9xf32>) -> tensor<3x9xf32>
  func.return %0 : tensor<3x9xf32>
}

// -----

// CHECK-LABEL: @rfft_dynamic
func.func @rfft_dynamic(%arg0: tensor<?x?xf32>) -> tensor<?x?xcomplex<f32>> {
  %0 = "mhlo.fft"(%arg0) { fft_length = dense<9> : tensor<1xi64>, fft_type = #mhlo<fft_type RFFT> } : (tensor<?x?xf32>) -> tensor<?x?xcomplex<f32>>
  func.return %0 : tensor<?x?xcomplex<f32>>
}

// -----

func.func @rfft_dynamic_incompatible_dims(%arg0: tensor<3x10xf32>) -> tensor<?x?xcomplex<f32>> {
  // expected-error@+1{{RFFT requires innermost dimensions to be compatible with fft_length. Got: 3, 10 but wanted 9.}}
  %0 = "mhlo.fft"(%arg0) { fft_length = dense<9> : tensor<1xi64>, fft_type = #mhlo<fft_type RFFT> } : (tensor<3x10xf32>) -> tensor<?x?xcomplex<f32>>
  func.return %0 : tensor<?x?xcomplex<f32>>
}

// -----

// CHECK-LABEL: @irfft_dynamic
func.func @irfft_dynamic(%arg0: tensor<?x?xcomplex<f32>>) -> tensor<?x?xf32> {
  %0 = "mhlo.fft"(%arg0) { fft_length = dense<16> : tensor<1xi64>, fft_type = #mhlo<fft_type IRFFT> } : (tensor<?x?xcomplex<f32>>) -> tensor<?x?xf32>
  func.return %0 : tensor<?x?xf32>
}

// -----

func.func @irfft_dynamic_incompatible_non_final_dims(%arg0: tensor<?x3x15xcomplex<f32>>) -> tensor<?x?x?xf32> {
  // expected-error@+1{{IRFFT requires non-final dimensions to be compatible with fft_length. Got: -9223372036854775808, 3, 15 but wanted 4, 16, and 3 != 4}}
  %0 = "mhlo.fft"(%arg0) { fft_length = dense<[4, 16]> : tensor<2xi64>, fft_type = #mhlo<fft_type IRFFT> } : (tensor<?x3x15xcomplex<f32>>) -> tensor<?x?x?xf32>
  func.return %0 : tensor<?x?x?xf32>
}

// -----

func.func @irfft_dynamic_incompatible_final_dim(%arg0: tensor<?x8xcomplex<f32>>) -> tensor<?x?xf32> {
  // expected-error@+1{{IRFFT requires innermost dimension to be compatible with fft_length[-1]/2+1. Got: 8 but fft_length is 16.}}
  %0 = "mhlo.fft"(%arg0) { fft_length = dense<16> : tensor<1xi64>, fft_type = #mhlo<fft_type IRFFT> } : (tensor<?x8xcomplex<f32>>) -> tensor<?x?xf32>
  func.return %0 : tensor<?x?xf32>
}

// -----

// CHECK-LABEL: @irfft_dynamic
func.func @irfft_dynamic(%arg0: tensor<?x?xcomplex<f32>>) -> tensor<?x?xf32> {
  %0 = "mhlo.fft"(%arg0) { fft_length = dense<16> : tensor<1xi64>, fft_type = #mhlo<fft_type IRFFT> } : (tensor<?x?xcomplex<f32>>) -> tensor<?x?xf32>
  func.return %0 : tensor<?x?xf32>
}

// -----

// CHECK-LABEL: @eltwise_static_and_dynamic_type(
//  CHECK-SAME: %[[A:.*]]: tensor<10x10xf32>, %[[B:.*]]: tensor<?x?xf32>) -> tensor<10x10xf32>
//       CHECK: %[[R:.*]] = mhlo.add %[[A]], %[[B]] : (tensor<10x10xf32>, tensor<?x?xf32>) -> tensor<10x10xf32>
//       CHECK: return %[[R]] : tensor<10x10xf32>
func.func @eltwise_static_and_dynamic_type(%arg0: tensor<10x10xf32>, %arg1: tensor<?x?xf32>) -> tensor<10x10xf32> {
  %0 = mhlo.add %arg0, %arg1 : (tensor<10x10xf32>, tensor<?x?xf32>) -> tensor<10x10xf32>
  func.return %0 : tensor<10x10xf32>
}

// -----

// CHECK: func @quantized_conv2d
// CHECK: mhlo.convolution
// CHECK-SAME: dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f]
// CHECK-SAME{LITERAL}: window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
func.func @quantized_conv2d(%arg0: tensor<1x8x8x207x!quant.uniform<i8:f32, 2.0:15>>, %arg1: tensor<3x3x207x16x!quant.uniform<i8:f32, 5.0:20>>) -> tensor<1x8x8x16x!quant.uniform<i8:f32, 10.0:50>> {
  %0 = mhlo.convolution(%arg0, %arg1)
         dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
         window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
         {batch_group_count = 1 : i64, feature_group_count = 1 : i64, precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]} :
       (tensor<1x8x8x207x!quant.uniform<i8:f32, 2.0:15>>, tensor<3x3x207x16x!quant.uniform<i8:f32, 5.0:20>>) -> tensor<1x8x8x16x!quant.uniform<i8:f32, 10.0:50>>
  func.return %0 : tensor<1x8x8x16x!quant.uniform<i8:f32, 10.0:50>>
}

// -----

// CHECK-LABEL: func @quantized_clamp
func.func @quantized_clamp(%arg0: tensor<1x!quant.uniform<ui8:f32, 34.0:16>>) -> tensor<1x!quant.uniform<ui8:f32, 34.0:16>> {
  %0 = "mhlo.clamp"(%arg0, %arg0, %arg0) : (tensor<1x!quant.uniform<ui8:f32, 34.0:16>>, tensor<1x!quant.uniform<ui8:f32, 34.0:16>>, tensor<1x!quant.uniform<ui8:f32, 34.0:16>>) -> tensor<1x!quant.uniform<ui8:f32, 34.0:16>>
  func.return %0: tensor<1x!quant.uniform<ui8:f32, 34.0:16>>
}

// -----

// CHECK-LABEL: func @quantized_dot_i8
func.func @quantized_dot_i8(%arg0: tensor<2x2x!quant.uniform<i8:f32, 2.0:15>>, %arg1: tensor<2x2x!quant.uniform<i8:f32, 5.0:20>>) -> tensor<2x2x!quant.uniform<i8:f32, 10.0:50>> {
  %0 = "mhlo.dot"(%arg0, %arg1) : (tensor<2x2x!quant.uniform<i8:f32, 2.0:15>>, tensor<2x2x!quant.uniform<i8:f32, 5.0:20>>) -> tensor<2x2x!quant.uniform<i8:f32, 10.0:50>>
  func.return %0: tensor<2x2x!quant.uniform<i8:f32, 10.0:50>>
}

// -----

// CHECK-LABEL: func @quantized_dot_i8_per_axis
func.func @quantized_dot_i8_per_axis(%arg0: tensor<2x2x!quant.uniform<i8:f32, 2.0:15>>, %arg1: tensor<2x2x!quant.uniform<i8<-127:127>:f32:0, {0.072314441204071045,0.050758145749568939}>>) -> tensor<2x2x!quant.uniform<i8:f32, 10.0:50>> {
  %0 = "mhlo.dot"(%arg0, %arg1) : (tensor<2x2x!quant.uniform<i8:f32, 2.0:15>>, tensor<2x2x!quant.uniform<i8<-127:127>:f32:0, {0.072314441204071045,0.050758145749568939}>>) -> tensor<2x2x!quant.uniform<i8:f32, 10.0:50>>
  func.return %0: tensor<2x2x!quant.uniform<i8:f32, 10.0:50>>
}

// -----

// CHECK-LABEL: func @quantized_dot_i4
func.func @quantized_dot_i4(%arg0: tensor<2x2x!quant.uniform<i4:f32, 2.0:15>>, %arg1: tensor<2x2x!quant.uniform<i4:f32, 5.0:20>>) -> tensor<2x2x!quant.uniform<i4:f32, 10.0:50>> {
  %0 = "mhlo.dot"(%arg0, %arg1) : (tensor<2x2x!quant.uniform<i4:f32, 2.0:15>>, tensor<2x2x!quant.uniform<i4:f32, 5.0:20>>) -> tensor<2x2x!quant.uniform<i4:f32, 10.0:50>>
  func.return %0: tensor<2x2x!quant.uniform<i4:f32, 10.0:50>>
}

// -----

// CHECK-LABEL: func @quantized_dot_general
func.func @quantized_dot_general(%arg0: tensor<2x16x32x!quant.uniform<i8:f32, 2.0:15>>, %arg1: tensor<2x32x32x!quant.uniform<i8:f32, 5.0:20>>) -> tensor<2x16x32x!quant.uniform<i8:f32, 10.0:50>> {
  %0 = "mhlo.dot_general"(%arg0, %arg1) {
    dot_dimension_numbers = #mhlo.dot<
      lhs_batching_dimensions = [0],
      lhs_contracting_dimensions = [2],
      rhs_batching_dimensions = [0],
      rhs_contracting_dimensions = [1]
    >,
    precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]}
    : (tensor<2x16x32x!quant.uniform<i8:f32, 2.0:15>>, tensor<2x32x32x!quant.uniform<i8:f32, 5.0:20>>) -> tensor<2x16x32x!quant.uniform<i8:f32, 10.0:50>>
  func.return %0 : tensor<2x16x32x!quant.uniform<i8:f32, 10.0:50>>
}

// -----

// CHECK-LABEL: func @add_dependency
func.func @add_dependency(%data: tensor<4x16xf32>) -> tensor<4x16xf32> {
  %token = "mhlo.create_token"() : () -> !mhlo.token
  %0 = "mhlo.add_dependency"(%data, %token) : (tensor<4x16xf32>, !mhlo.token) -> tensor<4x16xf32>
  func.return %0 : tensor<4x16xf32>
}
// -----

// CHECK-LABEL: func @add_dependency_token
func.func @add_dependency_token(%data: tensor<4x16xf32>) -> !mhlo.token {
  %token = "mhlo.create_token"() : () -> !mhlo.token
  %token2 = "mhlo.create_token"() : () -> !mhlo.token
  %0 = "mhlo.add_dependency"(%token2, %token) : (!mhlo.token, !mhlo.token) -> !mhlo.token
  func.return %0 : !mhlo.token
}

// -----

func.func @add_dependency(%data: tensor<4x16xf32>) -> !mhlo.token {
  // expected-error@+2 {{'mhlo.add_dependency' op inferred type(s) 'tensor<4x16xf32>' are incompatible with return type(s) of operation '!mhlo.token'}}
  %token = "mhlo.create_token"() : () -> !mhlo.token
  %0 = "mhlo.add_dependency"(%data, %token) : (tensor<4x16xf32>, !mhlo.token) -> !mhlo.token
  func.return %0 : !mhlo.token
}

// -----

func.func @add_dependency(%data: tensor<4x16xf32>) -> tensor<4x16xf32> {
  // expected-error@+3 {{inferred type(s) '!mhlo.token' are incompatible with return type(s) of operation 'tensor<4x16xf32>'}}
  %token = "mhlo.create_token"() : () -> !mhlo.token
  %token2 = "mhlo.create_token"() : () -> !mhlo.token
  %0 = "mhlo.add_dependency"(%token2, %token) : (!mhlo.token, !mhlo.token) -> tensor<4x16xf32>
  func.return %0 : tensor<4x16xf32>
}

// -----

// CHECK: func @uniform_quantize
func.func @uniform_quantize(%arg: tensor<16x16xf32>) -> tensor<16x16x!quant.uniform<ui8:f32, 34.0:16>> {
  %0 = mhlo.uniform_quantize %arg : (tensor<16x16xf32>) -> tensor<16x16x!quant.uniform<ui8:f32, 34.0:16>>
  func.return %0 : tensor<16x16x!quant.uniform<ui8:f32, 34.0:16>>
}

// -----

// CHECK: func @uniform_requantize
func.func @uniform_requantize(%arg: tensor<16x16x!quant.uniform<i8:f32, 5.0:20>>) -> tensor<16x16x!quant.uniform<i8:f32, 34.0:16>> {
  %0 = mhlo.uniform_quantize %arg : (tensor<16x16x!quant.uniform<i8:f32, 5.0:20>>) -> tensor<16x16x!quant.uniform<i8:f32, 34.0:16>>
  func.return %0 : tensor<16x16x!quant.uniform<i8:f32, 34.0:16>>
}

// -----

// CHECK: func @uniform_dequantize
func.func @uniform_dequantize(%arg: tensor<16x16x!quant.uniform<i8:f32, 34.0:16>>) -> tensor<16x16xf32> {
  %0 = mhlo.uniform_dequantize %arg : (tensor<16x16x!quant.uniform<i8:f32, 34.0:16>>) -> tensor<16x16xf32>
  func.return %0 : tensor<16x16xf32>
}

// -----

// CHECK: func @uniform_dequantize_unranked
func.func @uniform_dequantize_unranked(%arg: tensor<*x!quant.uniform<i8:f32, 34.0:16>>) -> tensor<*xf32> {
  %0 = mhlo.uniform_dequantize %arg : (tensor<*x!quant.uniform<i8:f32, 34.0:16>>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// -----

func.func @uniform_dequantize_not_quantize(%arg: tensor<16x16xf32>) -> tensor<16x16xf32> {
  // expected-error@+1 {{operand #0 must be tensor of 4/8/16/32-bit uniform quantized signed integer or 4/8/16/32-bit uniform quantized unsigned integer values, but got 'tensor<16x16xf32>'}}
  %0 = mhlo.uniform_dequantize %arg : (tensor<16x16xf32>) -> tensor<16x16xf32>
  func.return %0 : tensor<16x16xf32>
}

// -----

// CHECK-LABEL: func @quantized_constants
func.func @quantized_constants() -> (tensor<2x!quant.uniform<i8:f32, 2.0:15>>, tensor<2x!quant.uniform<ui8:f32, 34.0:16>>, tensor<2x!quant.uniform<i8:f32, 2.0:15>>) {
  %0 = mhlo.constant() {value = dense<[1, 2]> : tensor<2xi8>} : () -> tensor<2x!quant.uniform<i8:f32, 2.000000e+00:15>>
  %1 = mhlo.constant dense<[10.0, 12.0]> : tensor<2xf32>
  %2 = mhlo.constant dense<[3.0, 100.0]> : tensor<2xf32>
  %3 = mhlo.uniform_quantize %2 : (tensor<2xf32>) -> tensor<2x!quant.uniform<i8:f32, 2.0:15>>
  %4 = mhlo.uniform_quantize %1 : (tensor<2xf32>) -> tensor<2x!quant.uniform<ui8:f32, 34.0:16>>
  func.return %0, %4, %3 : tensor<2x!quant.uniform<i8:f32, 2.0:15>>, tensor<2x!quant.uniform<ui8:f32, 34.0:16>>, tensor<2x!quant.uniform<i8:f32, 2.0:15>>
  // CHECK: mhlo.constant() {value = dense<[1, 2]> : tensor<2xi8>} : () -> tensor<2x!quant.uniform<i8:f32, 2.000000e+00:15>>
  // CHECK-NEXT: mhlo.constant dense<[1.000000e+01, 1.200000e+01]> : tensor<2xf32>
  // CHECK-NEXT: mhlo.constant dense<[3.000000e+00, 1.000000e+02]> : tensor<2xf32>
}

// -----

func.func @quantized_constants_invalid_storage_type() -> () {
  // expected-error@+1 {{'mhlo.constant' op inferred type(s) 'tensor<2xui8>' are incompatible with return type(s) of operation 'tensor<2x!quant.uniform<i8:f32, 2.000000e+00:15>>}}
  %0 = "mhlo.constant"() {value = dense<[1, 2]> : tensor<2xui8>} : () -> tensor<2x!quant.uniform<i8:f32, 2.0:15>>
  func.return
}

// -----

func.func @dot_i4xi4_i8(%arg0: tensor<1x2xi4>, %arg1: tensor<2x1xi4>) -> tensor<1x1xi8> {
  %0 = "mhlo.dot"(%arg0, %arg1) : (tensor<1x2xi4>, tensor<2x1xi4>) -> tensor<1x1xi8>
  func.return %0: tensor<1x1xi8>
}

// -----

// CHECK-LABEL: func @dot_i8xi8_i16
func.func @dot_i8xi8_i16(%arg0: tensor<1x2xi8>, %arg1: tensor<2x1xi8>) -> tensor<1x1xi16> {
  %0 = "mhlo.dot"(%arg0, %arg1) : (tensor<1x2xi8>, tensor<2x1xi8>) -> tensor<1x1xi16>
  func.return %0: tensor<1x1xi16>
}

// -----

// CHECK-LABEL: func @einsum_i4xi4_i8
func.func @einsum_i4xi4_i8(%arg0: tensor<1x2xi4>, %arg1: tensor<2x1xi4>) -> tensor<1x1xi8> {
  %0 = "mhlo.einsum"(%arg0, %arg1) {einsum_config = "ab,bc->ac"} : (tensor<1x2xi4>, tensor<2x1xi4>) -> tensor<1x1xi8>
  func.return %0: tensor<1x1xi8>
}

// -----

// CHECK-LABEL: func @einsum_i8xi8_i16
func.func @einsum_i8xi8_i16(%arg0: tensor<1x2xi8>, %arg1: tensor<2x1xi8>) -> tensor<1x1xi16> {
  %0 = "mhlo.einsum"(%arg0, %arg1) {einsum_config = "ab,bc->ac"} : (tensor<1x2xi8>, tensor<2x1xi8>) -> tensor<1x1xi16>
  func.return %0: tensor<1x1xi16>
}

// -----

// CHECK-LABEL: func @part_id
func.func @part_id() -> tensor<ui32> {
  %1 = "mhlo.partition_id"() : () -> tensor<ui32>
  return %1 : tensor<ui32>
}

// -----

// CHECK-LABEL: func @domain
func.func @domain(%arg0: tensor<ui32>) -> tensor<ui32> {
  %1 = "mhlo.domain"(%arg0) {kind = #mhlo<kind sharding>, entry_metadata = "", exit_metadata = ""} : (tensor<ui32>) -> tensor<ui32>
  return %1 : tensor<ui32>
}

// -----

// CHECK-LABEL: func @conv_i4
func.func @conv_i4(%arg0: tensor<64x8x8x8xi4>, %arg1: tensor<4x4x8x32xi4>) -> tensor<64x3x3x32xi8> {
  // Note: This has been lowered and adapted from:
  // %0 = "tf.Conv2D"(%arg0, %arg1) {
  //        data_format = "NHWC",
  //        dilations = [1, 2, 2, 1],
  //        explicit_paddings = [0, 0, 0, 1, 0, 1, 0, 0],
  //        padding = "EXPLICIT",
  //        strides = [1, 1, 1, 1]} :
  //      (tensor<64x8x8x8xf32>, tensor<4x4x8x32xf32>) -> tensor<64x3x3x32xf32>
  %0 = mhlo.convolution(%arg0, %arg1)
         dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
         window = {stride = [1, 1], pad = [[0, 1], [0, 1]], rhs_dilate = [2, 2]}
         {batch_group_count = 1 : i64, feature_group_count = 1 : i64} :
       (tensor<64x8x8x8xi4>, tensor<4x4x8x32xi4>) -> tensor<64x3x3x32xi8>
  func.return %0 : tensor<64x3x3x32xi8>
}

// -----

// CHECK-LABEL: func @pad
func.func @pad(%arg0: tensor<1x2x3xf16>, %arg1: tensor<f16>) -> tensor<2x4x7xf16> {
  %0 = "mhlo.pad"(%arg0, %arg1) {
    edge_padding_low = dense<[0, 1, 2]> : tensor<3xi64>,
    edge_padding_high = dense<[1, 1, 0]> : tensor<3xi64>,
    interior_padding = dense<[0, 0, 1]> : tensor<3xi64>
  } : (tensor<1x2x3xf16>, tensor<f16>) -> tensor<2x4x7xf16>
  func.return %0 : tensor<2x4x7xf16>
}


// -----

func.func @pad_c2(%arg0: tensor<1x2x3xf16>, %arg1: tensor<f16>) -> tensor<2x4x7xf16> {
  // expected-error@+1 {{edge_padding_low length (2) must match operand rank (3)}}
  %0 = "mhlo.pad"(%arg0, %arg1) {
    edge_padding_low = dense<[0, 1]> : tensor<2xi64>,
    edge_padding_high = dense<[1, 1]> : tensor<2xi64>,
    interior_padding = dense<[0, 0]> : tensor<2xi64>
  } : (tensor<1x2x3xf16>, tensor<f16>) -> tensor<2x4x7xf16>
  func.return %0 : tensor<2x4x7xf16>
}

// -----

func.func @pad_c3(%arg0: tensor<1x2x3xf16>, %arg1: tensor<f16>) -> tensor<2x4x3xf16> {
  // expected-error@+1 {{Interior padding cannot be negative: -1}}
  %0 = "mhlo.pad"(%arg0, %arg1) {
    edge_padding_low = dense<[0, 1, 2]> : tensor<3xi64>,
    edge_padding_high = dense<[1, 1, 0]> : tensor<3xi64>,
    interior_padding = dense<[0, 0, -1]> : tensor<3xi64>
  } : (tensor<1x2x3xf16>, tensor<f16>) -> tensor<2x4x3xf16>
  func.return %0 : tensor<2x4x3xf16>
}

// -----

func.func @pad_c4(%arg0: tensor<1x2x3xf16>, %arg1: tensor<f16>) -> tensor<2x4x7xf16> {
  // expected-error@+1 {{Padding result in negative size for dimension 2}}
  %0 = "mhlo.pad"(%arg0, %arg1) {
    edge_padding_low = dense<[0, 1, -4]> : tensor<3xi64>,
    edge_padding_high = dense<[1, 1, 0]> : tensor<3xi64>,
    interior_padding = dense<[0, 0, 0]> : tensor<3xi64>
  } : (tensor<1x2x3xf16>, tensor<f16>) -> tensor<2x4x7xf16>
  func.return %0 : tensor<2x4x7xf16>
}

// -----

func.func @pad_c4(%arg0: tensor<1x2x3xf16>, %arg1: tensor<f16>) -> tensor<8x8x8xf16> {
  // expected-error@+1 {{'mhlo.pad' op inferred type(s) 'tensor<2x4x7xf16>' are incompatible with return type(s) of operation 'tensor<8x8x8xf16>'}}
  %0 = "mhlo.pad"(%arg0, %arg1) {
    edge_padding_low = dense<[0, 1, 2]> : tensor<3xi64>,
    edge_padding_high = dense<[1, 1, 0]> : tensor<3xi64>,
    interior_padding = dense<[0, 0, 1]> : tensor<3xi64>
  } : (tensor<1x2x3xf16>, tensor<f16>) -> tensor<8x8x8xf16>
  func.return %0 : tensor<8x8x8xf16>
}

// -----

// CHECK-LABEL: func @pad_dynamic
func.func @pad_dynamic(%arg0: tensor<?x48x48x32xf32>) -> tensor<?x48x48x48xf32> {
  %0 = "mhlo.constant"() {value = dense<0.000000e+00> : tensor<f32>} : () -> tensor<f32>
  %1 = "mhlo.pad"(%arg0, %0) {
    edge_padding_low = dense<0> : tensor<4xi64>,
    edge_padding_high = dense<[0, 0, 0, 16]> : tensor<4xi64>,
    interior_padding = dense<0> : tensor<4xi64>
  } : (tensor<?x48x48x32xf32>, tensor<f32>) -> tensor<?x48x48x48xf32>
  func.return %1 : tensor<?x48x48x48xf32>
}

// -----

func.func @pad_i2(%arg0: tensor<1x2x3xf16>, %arg1: tensor<2xf16>) -> tensor<2x4x7xf16> {
  // expected-error@+1 {{padding value type should be a rank-0 tensor, is rank 1}}
  %0 = "mhlo.pad"(%arg0, %arg1) {
    edge_padding_low = dense<[0, 1, 2]> : tensor<3xi64>,
    edge_padding_high = dense<[1, 1, 0]> : tensor<3xi64>,
    interior_padding = dense<[0, 0, 1]> : tensor<3xi64>
  } : (tensor<1x2x3xf16>, tensor<2xf16>) -> tensor<2x4x7xf16>
  func.return %0 : tensor<2x4x7xf16>
}

// -----

func.func @pad_i3(%arg0: tensor<1x2x3xf16>, %arg1: tensor<f16>) -> tensor<2x4x7xf16> {
  // expected-error@+1 {{edge_padding_low has rank 0 instead of required rank 1}}
  %0 = "mhlo.pad"(%arg0, %arg1) {
    edge_padding_low = dense<1> : tensor<i64>,
    edge_padding_high = dense<1> : tensor<i64>,
    interior_padding = dense<1> : tensor<i64>
  } : (tensor<1x2x3xf16>, tensor<f16>) -> tensor<2x4x7xf16>
  func.return %0 : tensor<2x4x7xf16>
}

// -----

func.func @is_compatible_dynamism_mix(%arg0: tensor<?xf32>, %arg1: tensor<1xf32>) {
  %0 = "mhlo.add"(%arg0, %arg0) : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  %1 = "mhlo.add"(%arg0, %arg0) : (tensor<?xf32>, tensor<?xf32>) -> tensor<1xf32>
  %2 = "mhlo.add"(%arg0, %arg1) : (tensor<?xf32>, tensor<1xf32>) -> tensor<?xf32>
  %3 = "mhlo.add"(%arg0, %arg1) : (tensor<?xf32>, tensor<1xf32>) -> tensor<1xf32>
  %4 = "mhlo.add"(%arg1, %arg0) : (tensor<1xf32>, tensor<?xf32>) -> tensor<?xf32>
  %5 = "mhlo.add"(%arg1, %arg0) : (tensor<1xf32>, tensor<?xf32>) -> tensor<1xf32>
  %6 = "mhlo.add"(%arg1, %arg1) : (tensor<1xf32>, tensor<1xf32>) -> tensor<?xf32>
  %7 = "mhlo.add"(%arg1, %arg1) : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
  func.return
}

// TODO(b/231448733): verifyCompatibleShape allows rankedness mismatches but Elemementwise doesn't.
// Sort this out while refactoring uses of SameOperandsAndResultType and friends.
// func.func @is_compatible_dynamism_mix(%arg0: tensor<*xf32>, %arg1: tensor<?xf32>, %arg2: tensor<1xf32>) {
//   %0 = "mhlo.add"(%arg0, %arg0) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
//   %1 = "mhlo.add"(%arg0, %arg0) : (tensor<*xf32>, tensor<*xf32>) -> tensor<?xf32>
//   %2 = "mhlo.add"(%arg0, %arg0) : (tensor<*xf32>, tensor<*xf32>) -> tensor<1xf32>
//   %3 = "mhlo.add"(%arg0, %arg1) : (tensor<*xf32>, tensor<?xf32>) -> tensor<*xf32>
//   %4 = "mhlo.add"(%arg0, %arg1) : (tensor<*xf32>, tensor<?xf32>) -> tensor<?xf32>
//   %5 = "mhlo.add"(%arg0, %arg1) : (tensor<*xf32>, tensor<?xf32>) -> tensor<1xf32>
//   %6 = "mhlo.add"(%arg0, %arg2) : (tensor<*xf32>, tensor<1xf32>) -> tensor<*xf32>
//   %7 = "mhlo.add"(%arg0, %arg2) : (tensor<*xf32>, tensor<1xf32>) -> tensor<?xf32>
//   %8 = "mhlo.add"(%arg0, %arg2) : (tensor<*xf32>, tensor<1xf32>) -> tensor<1xf32>
//   %9 = "mhlo.add"(%arg1, %arg0) : (tensor<?xf32>, tensor<*xf32>) -> tensor<*xf32>
//   %10 = "mhlo.add"(%arg1, %arg0) : (tensor<?xf32>, tensor<*xf32>) -> tensor<?xf32>
//   %11 = "mhlo.add"(%arg1, %arg0) : (tensor<?xf32>, tensor<*xf32>) -> tensor<1xf32>
//   %12 = "mhlo.add"(%arg1, %arg1) : (tensor<?xf32>, tensor<?xf32>) -> tensor<*xf32>
//   %13 = "mhlo.add"(%arg1, %arg1) : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
//   %14 = "mhlo.add"(%arg1, %arg1) : (tensor<?xf32>, tensor<?xf32>) -> tensor<1xf32>
//   %15 = "mhlo.add"(%arg1, %arg2) : (tensor<?xf32>, tensor<1xf32>) -> tensor<*xf32>
//   %16 = "mhlo.add"(%arg1, %arg2) : (tensor<?xf32>, tensor<1xf32>) -> tensor<?xf32>
//   %17 = "mhlo.add"(%arg1, %arg2) : (tensor<?xf32>, tensor<1xf32>) -> tensor<1xf32>
//   %18 = "mhlo.add"(%arg2, %arg0) : (tensor<1xf32>, tensor<*xf32>) -> tensor<*xf32>
//   %19 = "mhlo.add"(%arg2, %arg0) : (tensor<1xf32>, tensor<*xf32>) -> tensor<?xf32>
//   %20 = "mhlo.add"(%arg2, %arg0) : (tensor<1xf32>, tensor<*xf32>) -> tensor<1xf32>
//   %21 = "mhlo.add"(%arg2, %arg1) : (tensor<1xf32>, tensor<?xf32>) -> tensor<*xf32>
//   %22 = "mhlo.add"(%arg2, %arg1) : (tensor<1xf32>, tensor<?xf32>) -> tensor<?xf32>
//   %23 = "mhlo.add"(%arg2, %arg1) : (tensor<1xf32>, tensor<?xf32>) -> tensor<1xf32>
//   %24 = "mhlo.add"(%arg2, %arg2) : (tensor<1xf32>, tensor<1xf32>) -> tensor<*xf32>
//   %25 = "mhlo.add"(%arg2, %arg2) : (tensor<1xf32>, tensor<1xf32>) -> tensor<?xf32>
//   %26 = "mhlo.add"(%arg2, %arg2) : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
//   func.return
// }

// -----

func.func @is_compatible_dynamism_rankedness_mismatch(%arg0: tensor<*xf32>) {
  // expected-error@+1 {{all non-scalar operands/results must have the same shape and base type}}
  %0 = "mhlo.add"(%arg0, %arg0) : (tensor<*xf32>, tensor<*xf32>) -> tensor<1xf32>
  func.return
}

// -----

func.func @is_compatible_dynamism_ranked_mismatch(%arg0: tensor<?xf32>) {
  // expected-error@+1 {{op requires compatible types for all operands and results}}
  %0 = "mhlo.add"(%arg0, %arg0) : (tensor<?xf32>, tensor<?xf32>) -> tensor<?x?xf32>
  func.return
}

// -----

func.func @is_compatible_dynamism_dim_mismatch(%arg0: tensor<1x?xf32>) {
  // expected-error@+1 {{op requires compatible types for all operands and results}}
  %0 = "mhlo.add"(%arg0, %arg0) : (tensor<1x?xf32>, tensor<1x?xf32>) -> tensor<2x2xf32>
  func.return
}

// -----

// TODO(b/230263270): For mhlo.add, the plan is to only allow fp+fp=fp, q+q=q and q+q=fp.
func.func @is_compatible_quant_mix_non_quant(%arg0: tensor<1xf32>, %arg1: tensor<1x!quant.uniform<i8:f32, 1.0:17>>) {
  %0 = "mhlo.add"(%arg0, %arg0) : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
  %1 = "mhlo.add"(%arg0, %arg0) : (tensor<1xf32>, tensor<1xf32>) -> tensor<1x!quant.uniform<i8:f32, 1.0:17>>
  %2 = "mhlo.add"(%arg0, %arg1) : (tensor<1xf32>, tensor<1x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<1x!quant.uniform<i8:f32, 1.0:17>>
  %3 = "mhlo.add"(%arg0, %arg1) : (tensor<1xf32>, tensor<1x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<1x!quant.uniform<i8:f32, 1.0:17>>
  %4 = "mhlo.add"(%arg1, %arg0) : (tensor<1x!quant.uniform<i8:f32, 1.0:17>>, tensor<1xf32>) -> tensor<1xf32>
  %5 = "mhlo.add"(%arg1, %arg0) : (tensor<1x!quant.uniform<i8:f32, 1.0:17>>, tensor<1xf32>) -> tensor<1xf32>
  %6 = "mhlo.add"(%arg1, %arg1) : (tensor<1x!quant.uniform<i8:f32, 1.0:17>>, tensor<1x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<1x!quant.uniform<i8:f32, 1.0:17>>
  %7 = "mhlo.add"(%arg1, %arg1) : (tensor<1x!quant.uniform<i8:f32, 1.0:17>>, tensor<1x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<1x!quant.uniform<i8:f32, 1.0:17>>
  func.return
}

// -----

func.func @is_compatible_quant_mix_scale(%arg0: tensor<1x!quant.uniform<i8:f32, 1.0:17>>) {
  %0 = "mhlo.add"(%arg0, %arg0) : (tensor<1x!quant.uniform<i8:f32, 1.0:17>>, tensor<1x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<1x!quant.uniform<i8:f32, 2.0:17>>
  func.return
}

// -----

func.func @is_compatible_quant_mix_zero_point(%arg0: tensor<1x!quant.uniform<i8:f32, 1.0:17>>) {
  %0 = "mhlo.add"(%arg0, %arg0) : (tensor<1x!quant.uniform<i8:f32, 1.0:17>>, tensor<1x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<1x!quant.uniform<i8:f32, 1.0:18>>
  func.return
}

// -----

func.func @is_compatible_quant_expressed_mismatch(%arg0: tensor<1x!quant.uniform<i8:f32, 1.0:17>>) {
  // expected-error@+1 {{op requires compatible types for all operands and results}}
  %0 = "mhlo.add"(%arg0, %arg0) : (tensor<1x!quant.uniform<i8:f32, 1.0:17>>, tensor<1x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<1x!quant.uniform<i8:bf16, 1.0:17>>
  func.return
}

// -----

func.func @is_compatible_quant_storage_mismatch(%arg0: tensor<1x!quant.uniform<i8:f32, 1.0:17>>) {
  // expected-error@+1 {{op requires compatible types for all operands and results}}
  %0 = "mhlo.add"(%arg0, %arg0) : (tensor<1x!quant.uniform<i8:f32, 1.0:17>>, tensor<1x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<1x!quant.uniform<i4:f32, 1.0:17>>
  func.return
}

// -----

func.func @is_compatible_quant_signedness_mismatch(%arg0: tensor<1x!quant.uniform<i8:f32, 1.0:17>>) {
  // expected-error@+1 {{op requires compatible types for all operands and results}}
  %0 = "mhlo.add"(%arg0, %arg0) : (tensor<1x!quant.uniform<i8:f32, 1.0:17>>, tensor<1x!quant.uniform<i8:f32, 1.0:17>>) -> tensor<1x!quant.uniform<u8:f32, 1.0:17>>
  func.return
}

// -----

// CHECK-LABEL: is_compatible_dynamism_bounds
func.func @is_compatible_dynamism_bounds_mismatch(
  %arg0: tensor<?xf32, #mhlo.type_extensions<bounds = [4]>>,
  %arg1: tensor<?xf32, #mhlo.type_extensions<bounds = [4]>>) {
  %0 = "mhlo.add"(%arg0, %arg1) : (
    tensor<?xf32, #mhlo.type_extensions<bounds = [4]>>,
    tensor<?xf32, #mhlo.type_extensions<bounds = [4]>>) -> tensor<3xf32>
  func.return
}

// -----

func.func @is_compatible_dynamism_bounds_mismatch(
  %arg0: tensor<?xf32, #mhlo.type_extensions<bounds = [4]>>,
  %arg1: tensor<?xf32, #mhlo.type_extensions<bounds = [4]>>) {
  // expected-error@+1 {{requires compatible types for all operands and results}}
  %0 = "mhlo.add"(%arg0, %arg1) : (
    tensor<?xf32, #mhlo.type_extensions<bounds = [4]>>,
    tensor<?xf32, #mhlo.type_extensions<bounds = [4]>>) -> tensor<5xf32>
  func.return
}

// -----

// CHECK-LABEL: scatter_update_scalar
func.func @scatter_update_scalar(%arg0: tensor<3xi32>, %arg1: tensor<1x1xi32>,
                            %arg2: tensor<1xi32>) -> tensor<3xi32> {
  %0 = "mhlo.scatter"(%arg0, %arg1, %arg2) ({
  ^bb0(%arg3: tensor<i32>, %arg4: tensor<i32>):
    "mhlo.return"(%arg4) : (tensor<i32>) -> ()
  }) {
    indices_are_sorted = false,
    scatter_dimension_numbers = #mhlo.scatter<
      update_window_dims = [],
      inserted_window_dims = [0],
      scatter_dims_to_operand_dims = [0],
      index_vector_dim = 1,
    >,
    unique_indices = false
  } : (tensor<3xi32>, tensor<1x1xi32>, tensor<1xi32>) -> tensor<3xi32>
  func.return %0 : tensor<3xi32>
}

// -----

// CHECK-LABEL: scatter_variadic
func.func @scatter_variadic(%arg0: tensor<3xi32>, %arg1: tensor<1x1xi32>,
                            %arg2: tensor<1xi32>) -> tensor<3xi32> {
  %0, %1 = "mhlo.scatter"(%arg0, %arg0, %arg1, %arg2, %arg2) ({
  ^bb0(%arg3: tensor<i32>, %arg4: tensor<i32>, %arg5: tensor<i32>, %arg6: tensor<i32>):
    "mhlo.return"(%arg3, %arg5) : (tensor<i32>, tensor<i32>) -> ()
  }) {
    indices_are_sorted = false,
    scatter_dimension_numbers = #mhlo.scatter<
      update_window_dims = [],
      inserted_window_dims = [0],
      scatter_dims_to_operand_dims = [0],
      index_vector_dim = 1,
    >,
    unique_indices = false
  } : (tensor<3xi32>, tensor<3xi32>, tensor<1x1xi32>, tensor<1xi32>, tensor<1xi32>) -> (tensor<3xi32>, tensor<3xi32>)
  func.return %0 : tensor<3xi32>
}

// -----


#SV = #sparse_tensor.encoding<{
  dimLevelType = ["compressed"]
}>

func.func @is_compatible_sparse_mix_non_sparse(%arg0: tensor<1xf32>, %arg1: tensor<1xf32, #SV>) {
  %0 = "mhlo.add"(%arg0, %arg0) : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
  %1 = "mhlo.add"(%arg0, %arg0) : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32, #SV>
  %2 = "mhlo.add"(%arg0, %arg1) : (tensor<1xf32>, tensor<1xf32, #SV>) -> tensor<1xf32, #SV>
  %3 = "mhlo.add"(%arg0, %arg1) : (tensor<1xf32>, tensor<1xf32, #SV>) -> tensor<1xf32, #SV>
  %4 = "mhlo.add"(%arg1, %arg0) : (tensor<1xf32, #SV>, tensor<1xf32>) -> tensor<1xf32>
  %5 = "mhlo.add"(%arg1, %arg0) : (tensor<1xf32, #SV>, tensor<1xf32>) -> tensor<1xf32>
  %6 = "mhlo.add"(%arg1, %arg1) : (tensor<1xf32, #SV>, tensor<1xf32, #SV>) -> tensor<1xf32, #SV>
  %7 = "mhlo.add"(%arg1, %arg1) : (tensor<1xf32, #SV>, tensor<1xf32, #SV>) -> tensor<1xf32, #SV>
  func.return
}

// CHECK-LABEL: func @abs
func.func @abs(%arg0: tensor<1x2xf32>) -> tensor<1x2xf32> {
  %0 = "mhlo.abs"(%arg0) {} : (tensor<1x2xf32>) -> tensor<1x2xf32>
  func.return %0 : tensor<1x2xf32>
}

// -----

// CHECK-LABEL: func @abs_complex
func.func @abs_complex(%arg0: tensor<1x2xcomplex<f32>>) -> tensor<1x2xf32> {
  %0 = "mhlo.abs"(%arg0) {} : (tensor<1x2xcomplex<f32>>) -> tensor<1x2xf32>
  func.return %0 : tensor<1x2xf32>
}

// -----

func.func @abs_mismatch_element_type(%arg0: tensor<1x2xcomplex<f32>>) -> tensor<1x2xf64> {
// expected-error@+1 {{'mhlo.abs' op inferred type(s) 'tensor<1x2xf32>' are incompatible with return type(s) of operation 'tensor<1x2xf64>'}}
  %0 = "mhlo.abs"(%arg0) {} : (tensor<1x2xcomplex<f32>>) -> tensor<1x2xf64>
  func.return %0 : tensor<1x2xf64>
}

// -----

func.func @abs_complex_mismatch_element_type(%arg0: tensor<1x2xcomplex<f32>>) -> tensor<1x2xf64> {
// expected-error@+1 {{'stablehlo.abs' op inferred type(s) 'tensor<1x2xf32>' are incompatible with return type(s) of operation 'tensor<1x2xf64>'}}
  %0 = "stablehlo.abs"(%arg0) {} : (tensor<1x2xcomplex<f32>>) -> tensor<1x2xf64>
  func.return %0 : tensor<1x2xf64>
}

// -----

// CHECK-LABEL: func @round_even
func.func @round_even(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  %0 = "mhlo.round_nearest_even"(%arg0) {} : (tensor<2xf32>) -> tensor<2xf32>
  func.return %0 : tensor<2xf32>
}

// -----

// CHECK-LABEL: func @complex
func.func @complex(%arg0: tensor<10x10xf32>, %arg1: tensor<10x10xf32>) -> tensor<10x10xcomplex<f32>> {
  %0 = "mhlo.complex"(%arg0, %arg1) {} : (tensor<10x10xf32>, tensor<10x10xf32>) -> tensor<10x10xcomplex<f32>>
  func.return %0 : tensor<10x10xcomplex<f32>>
}

// -----

func.func @complex_int_input(%arg0: tensor<10x10xi32>, %arg1: tensor<10x10xi32>) -> tensor<10x10xcomplex<i32>> {
  // expected-error@+1 {{operand #0 must be tensor of 32-bit float or 64-bit float values, but got 'tensor<10x10xi32>'}}
  %0 = "mhlo.complex"(%arg0, %arg1) {} : (tensor<10x10xi32>, tensor<10x10xi32>) -> tensor<10x10xcomplex<i32>>
  func.return %0 : tensor<10x10xcomplex<i32>>
}

// -----

func.func @complex_f32_f64_mix_input(%arg0: tensor<10x10xf32>, %arg1: tensor<10x10xf64>) -> tensor<10x10xcomplex<f64>> {
  // expected-error@+1 {{requires the same element type for all operands}}
  %0 = "mhlo.complex"(%arg0, %arg1) {} : (tensor<10x10xf32>, tensor<10x10xf64>) -> tensor<10x10xcomplex<f64>>
  func.return %0 : tensor<10x10xcomplex<f64>>
}

// -----

func.func @complex_f16_input(%arg0: tensor<10x10xf16>, %arg1: tensor<10x10xf16>) -> tensor<10x10xcomplex<f16>> {
  // expected-error@+1 {{operand #0 must be tensor of 32-bit float or 64-bit float values, but got 'tensor<10x10xf16>'}}
  %0 = "mhlo.complex"(%arg0, %arg1) {} : (tensor<10x10xf16>, tensor<10x10xf16>) -> tensor<10x10xcomplex<f16>>
  func.return %0 : tensor<10x10xcomplex<f16>>
}

// -----

func.func @complex_mismatch_return_element_type(%arg0: tensor<10x10xf32>, %arg1: tensor<10x10xf32>) -> tensor<10x10xcomplex<f64>> {
  // expected-error@+1 {{inferred type(s) 'tensor<10x10xcomplex<f32>>' are incompatible with return type(s) of operation 'tensor<10x10xcomplex<f64>>'}}
  %0 = "mhlo.complex"(%arg0, %arg1) {} : (tensor<10x10xf32>, tensor<10x10xf32>) -> tensor<10x10xcomplex<f64>>
  func.return %0 : tensor<10x10xcomplex<f64>>
}

// -----

func.func @complex_mismatch_return_shape(%arg0: tensor<10x10xf32>, %arg1: tensor<10x10xf32>) -> tensor<5x5xcomplex<f32>> {
  // expected-error@+1 {{requires the same shape for all operands and results}}
  %0 = "mhlo.complex"(%arg0, %arg1) {} : (tensor<10x10xf32>, tensor<10x10xf32>) -> tensor<5x5xcomplex<f32>>
  func.return %0 : tensor<5x5xcomplex<f32>>
}

// -----

// async positive test

// CHECK-LABEL: func @async_op
// CHECK-LABEL: func @async
func.func @async_op(%arg0: tensor<10x10xf32>) -> tensor<32xf32>
  attributes {execution_thread = "thread"} {
  %1 = mhlo.constant dense<2.0> : tensor<32xf32>
  func.return %1 : tensor<32xf32>
}

func.func @async(%arg0: tensor<10x10xf32>) -> tensor<32xf32> {
  %0 = "mhlo.async_start"(%arg0) {called_computation=@async_op, execution_thread="thread"} : (tensor<10x10xf32>) -> !mhlo.async_bundle<tensor<10x10xf32>, tensor<32xf32>, tensor<i32>>
  %1 = "mhlo.async_update"(%0) {called_computation=@async_op, execution_thread="thread"} : (!mhlo.async_bundle<tensor<10x10xf32>, tensor<32xf32>, tensor<i32>>) -> !mhlo.async_bundle<tensor<10x10xf32>, tensor<32xf32>, tensor<i32>>
  %2 = "mhlo.async_done"(%1) {called_computation=@async_op, execution_thread="thread"} : (!mhlo.async_bundle<tensor<10x10xf32>, tensor<32xf32>, tensor<i32>>) -> tensor<32xf32>
  func.return %2 : tensor<32xf32>
}

// -----

func.func @async_op(%arg0: tensor<10x10xf32>) -> tensor<32xf32>
  attributes {execution_thread = "thread"} {
  %1 = mhlo.constant dense<2.0> : tensor<32xf32>
  func.return %1 : tensor<32xf32>
}

func.func @async(%arg0: tensor<10x10xf32>) -> tensor<32xf32> {
  // expected-error@+1 {{component #0 of return type doesn't match callee input types}}
  %0 = "mhlo.async_start"(%arg0) {called_computation=@async_op, execution_thread="thread"} : (tensor<10x10xf32>) -> !mhlo.async_bundle<tensor<10xf32>, tensor<32xf32>, tensor<i32>>
  %1 = "mhlo.async_update"(%0) {called_computation=@async_op, execution_thread="thread"} : (!mhlo.async_bundle<tensor<10xf32>, tensor<32xf32>, tensor<i32>>) -> !mhlo.async_bundle<tensor<10xf32>, tensor<32xf32>, tensor<i32>>
  %2 = "mhlo.async_done"(%1) {called_computation=@async_op, execution_thread="thread"} : (!mhlo.async_bundle<tensor<10xf32>, tensor<32xf32>, tensor<i32>>) -> tensor<32xf32>
  func.return %2 : tensor<32xf32>
}

// -----

func.func @async_op(%arg0: tensor<10x10xf32>) -> tensor<32xf32>
  attributes {execution_thread = "thread"} {
  %1 = mhlo.constant dense<2.0> : tensor<32xf32>
  func.return %1 : tensor<32xf32>
}

func.func @async(%arg0: tensor<10x10xf32>) -> tensor<32xf32> {
  // expected-error@+1 {{component #1 of return type doesn't match callee result types}}
  %0 = "mhlo.async_start"(%arg0) {called_computation=@async_op, execution_thread="thread"} : (tensor<10x10xf32>) -> !mhlo.async_bundle<tensor<10x10xf32>, tensor<f32>, tensor<i32>>
  %1 = "mhlo.async_update"(%0) {called_computation=@async_op, execution_thread="thread"} : (!mhlo.async_bundle<tensor<10x10xf32>, tensor<f32>, tensor<i32>>) -> !mhlo.async_bundle<tensor<10x10xf32>, tensor<f32>, tensor<i32>>
  %2 = "mhlo.async_done"(%1) {called_computation=@async_op, execution_thread="thread"} : (!mhlo.async_bundle<tensor<10x10xf32>, tensor<f32>, tensor<i32>>) -> tensor<32xf32>
  func.return %2 : tensor<32xf32>
}

// -----

/////
// async_start negative tests
/////

func.func @async(%arg0: tensor<10x10xf32>) -> tensor<32xf32> {
  // expected-error@+1 {{can't find function: async_op}}
  %0 = "mhlo.async_start"(%arg0) {called_computation=@async_op, execution_thread="thread"} : (tensor<10x10xf32>) -> !mhlo.async_bundle<tensor<64xf32>, tensor<32xf32>, tensor<i32>>
  %1 = "mhlo.async_update"(%0) {called_computation=@async_op, execution_thread="thread"} : (!mhlo.async_bundle<tensor<64xf32>, tensor<32xf32>, tensor<i32>>) -> !mhlo.async_bundle<tensor<64xf32>, tensor<32xf32>, tensor<i32>>
  %2 = "mhlo.async_done"(%1) {called_computation=@async_op, execution_thread="thread"} : (!mhlo.async_bundle<tensor<64xf32>, tensor<32xf32>, tensor<i32>>) -> tensor<32xf32>
  func.return %2 : tensor<32xf32>
}

// -----

func.func @async_op(%arg0: tensor<10x10xf32>) -> tensor<32xf32> {
  %1 = mhlo.constant dense<2.0> : tensor<32xf32>
  func.return %1 : tensor<32xf32>
}

func.func @async(%arg0: tensor<10x10xf32>) -> tensor<32xf32> {
  // expected-error@+1 {{callee must have execution_thread attribute}}
  %0 = "mhlo.async_start"(%arg0) {called_computation=@async_op, execution_thread="thread"} : (tensor<10x10xf32>) -> !mhlo.async_bundle<tensor<10x10xf32>, tensor<32xf32>, tensor<i32>>
  %1 = "mhlo.async_update"(%0) {called_computation=@async_op, execution_thread="thread"} : (!mhlo.async_bundle<tensor<10x10xf32>, tensor<32xf32>, tensor<i32>>) -> !mhlo.async_bundle<tensor<10x10xf32>, tensor<32xf32>, tensor<i32>>
  %2 = "mhlo.async_done"(%1) {called_computation=@async_op, execution_thread="thread"} : (!mhlo.async_bundle<tensor<10x10xf32>, tensor<32xf32>, tensor<i32>>) -> tensor<32xf32>
  func.return %2 : tensor<32xf32>
}

// -----

func.func @async_op(%arg0: tensor<10x10xf32>) -> tensor<32xf32>
  attributes {execution_thread = "thread"} {
  %1 = mhlo.constant dense<2.0> : tensor<32xf32>
  func.return %1 : tensor<32xf32>
}

func.func @async(%arg0: tensor<10x10xf32>) -> tensor<10x10xf32> {
  // expected-error@+1 {{result is expected to be a bundle of at least 2 components, but got 1}}
  %0 = "mhlo.async_start"(%arg0) {called_computation=@async_op, execution_thread="thread"} : (tensor<10x10xf32>) -> !mhlo.async_bundle<tensor<10x10xf32>>
  func.return %arg0 : tensor<10x10xf32>
}

// -----

func.func @async_op(%arg0: tensor<10x10xf32>) -> tensor<32xf32>
  attributes {execution_thread = "thread2"} {
  %1 = mhlo.constant dense<2.0> : tensor<32xf32>
  func.return %1 : tensor<32xf32>
}

func.func @async(%arg0: tensor<10x10xf32>) -> tensor<32xf32> {
  // expected-error@+1 {{op execution_thread does not match the execution_thread of async_op.  Got: "thread", but expected "thread2".}}
  %0 = "mhlo.async_start"(%arg0) {called_computation=@async_op, execution_thread="thread"} : (tensor<10x10xf32>) -> !mhlo.async_bundle<tensor<10x10xf32>, tensor<32xf32>, tensor<i32>>
  %1 = "mhlo.async_update"(%0) {called_computation=@async_op, execution_thread="thread"} : (!mhlo.async_bundle<tensor<10x10xf32>, tensor<32xf32>, tensor<i32>>) -> !mhlo.async_bundle<tensor<10x10xf32>, tensor<32xf32>, tensor<i32>>
  %2 = "mhlo.async_done"(%1) {called_computation=@async_op, execution_thread="thread"} : (!mhlo.async_bundle<tensor<10x10xf32>, tensor<32xf32>, tensor<i32>>) -> tensor<32xf32>
  func.return %2 : tensor<32xf32>
}

// -----

func.func @async_op(%arg0: tensor<10x10xf32>) -> tensor<32xf32>
  attributes {execution_thread = "thread"} {
  %1 = mhlo.constant dense<2.0> : tensor<32xf32>
  func.return %1 : tensor<32xf32>
}

func.func @async(%arg0: tensor<10x10xf32>) -> tensor<32xf32> {
  // expected-error@+1 {{number of operands doesn't match operands for async_op. Got: 0, but expected: 1.}}
  %0 = "mhlo.async_start"() {called_computation=@async_op, execution_thread="thread"} : () -> !mhlo.async_bundle<tensor<64xf32>, tensor<32xf32>, tensor<i32>>
  %1 = "mhlo.async_update"(%0) {called_computation=@async_op, execution_thread="thread"} : (!mhlo.async_bundle<tensor<64xf32>, tensor<32xf32>, tensor<i32>>) -> !mhlo.async_bundle<tensor<64xf32>, tensor<32xf32>, tensor<i32>>
  %2 = "mhlo.async_done"(%1) {called_computation=@async_op, execution_thread="thread"} : (!mhlo.async_bundle<tensor<64xf32>, tensor<32xf32>, tensor<i32>>) -> tensor<32xf32>
  func.return %2 : tensor<32xf32>
}

// -----

func.func @async_op(%arg0: tensor<10x10xf32>) -> tensor<32xf32>
  attributes {execution_thread = "thread"} {
  %1 = mhlo.constant dense<2.0> : tensor<32xf32>
  func.return %1 : tensor<32xf32>
}

func.func @async(%arg0: tensor<f32>) -> tensor<32xf32> {
  // expected-error@+1 {{type mismatch on argument #0 of async_op. Got: 'tensor<f32>', but expected: 'tensor<10x10xf32>'}}
  %0 = "mhlo.async_start"(%arg0) {called_computation=@async_op, execution_thread="thread"} : (tensor<f32>) -> !mhlo.async_bundle<tensor<64xf32>, tensor<32xf32>, tensor<i32>>
  %1 = "mhlo.async_update"(%0) {called_computation=@async_op, execution_thread="thread"} : (!mhlo.async_bundle<tensor<64xf32>, tensor<32xf32>, tensor<i32>>) -> !mhlo.async_bundle<tensor<64xf32>, tensor<32xf32>, tensor<i32>>
  %2 = "mhlo.async_done"(%1) {called_computation=@async_op, execution_thread="thread"} : (!mhlo.async_bundle<tensor<64xf32>, tensor<32xf32>, tensor<i32>>) -> tensor<32xf32>
  func.return %2 : tensor<32xf32>
}

// -----
/////
// async_update negative tests
/////

func.func @async_op(%arg0: tensor<10x10xf32>) -> tensor<32xf32>
  attributes {execution_thread = "thread"} {
  %1 = mhlo.constant dense<2.0> : tensor<32xf32>
  func.return %1 : tensor<32xf32>
}

func.func @async(%arg0: tensor<10x10xf32>) -> tensor<32xf32> {
  %0 = "mhlo.async_start"(%arg0) {called_computation=@async_op, execution_thread="thread"} : (tensor<10x10xf32>) -> !mhlo.async_bundle<tensor<10x10xf32>, tensor<32xf32>, tensor<i32>>
  // expected-error@+1 {{op execution_thread does not match name of async_op.  Got: "thread2", but expected "thread".}}
  %1 = "mhlo.async_update"(%0) {called_computation=@async_op, execution_thread="thread2"} : (!mhlo.async_bundle<tensor<10x10xf32>, tensor<32xf32>, tensor<i32>>) -> !mhlo.async_bundle<tensor<10x10xf32>, tensor<32xf32>, tensor<i32>>
  %2 = "mhlo.async_done"(%1) {called_computation=@async_op, execution_thread="thread"} : (!mhlo.async_bundle<tensor<10x10xf32>, tensor<32xf32>, tensor<i32>>) -> tensor<32xf32>
  func.return %2 : tensor<32xf32>
}

// -----
/////
// async_update negative tests
/////

func.func @async_op(%arg0: tensor<10x10xf32>) -> tensor<32xf32>
  attributes {execution_thread = "thread"} {
  %1 = mhlo.constant dense<2.0> : tensor<32xf32>
  func.return %1 : tensor<32xf32>
}

func.func @async(%arg0: tensor<10x10xf32>) -> tensor<32xf32> {
  %0 = "mhlo.async_start"(%arg0) {called_computation=@async_op, execution_thread="thread"} : (tensor<10x10xf32>) -> !mhlo.async_bundle<tensor<10x10xf32>, tensor<32xf32>, tensor<i32>>
  %1 = "mhlo.async_update"(%0) {called_computation=@async_op, execution_thread="thread"} : (!mhlo.async_bundle<tensor<10x10xf32>, tensor<32xf32>, tensor<i32>>) -> !mhlo.async_bundle<tensor<10x10xf32>, tensor<32xf32>, tensor<i32>>
  // expected-error@+1 {{op execution_thread does not match name of async_op.  Got: "thread2", but expected "thread".}}
  %2 = "mhlo.async_done"(%1) {called_computation=@async_op, execution_thread="thread2"} : (!mhlo.async_bundle<tensor<10x10xf32>, tensor<32xf32>, tensor<i32>>) -> tensor<32xf32>
  func.return %2 : tensor<32xf32>
}
// -----

func.func @async_op(%arg0: tensor<10x10xf32>) -> tensor<32xf32>
  attributes {execution_thread = "thread"} {
  %1 = mhlo.constant dense<2.0> : tensor<32xf32>
  func.return %1 : tensor<32xf32>
}

func.func @async(%arg0: tensor<10x10xf32>) -> tensor<f32> {
  %0 = "mhlo.async_start"(%arg0) {called_computation=@async_op, execution_thread="thread"} : (tensor<10x10xf32>) -> !mhlo.async_bundle<tensor<10x10xf32>, tensor<32xf32>, tensor<i32>>
  %1 = "mhlo.async_update"(%0) {called_computation=@async_op, execution_thread="thread"} : (!mhlo.async_bundle<tensor<10x10xf32>, tensor<32xf32>, tensor<i32>>) -> !mhlo.async_bundle<tensor<10x10xf32>, tensor<32xf32>, tensor<i32>>
// expected-error@+1 {{inferred type(s) 'tensor<32xf32>' are incompatible with return type(s) of operation 'tensor<f32>'}}
  %2 = "mhlo.async_done"(%1) {called_computation=@async_op, execution_thread="thread"} : (!mhlo.async_bundle<tensor<10x10xf32>, tensor<32xf32>, tensor<i32>>) -> tensor<f32>
  func.return %2 : tensor<f32>
}

// -----

// CHECK-LABEL: func @is_finite
func.func @is_finite(%arg0: tensor<3xf32>) -> tensor<3xi1> {
  %0 = "mhlo.is_finite"(%arg0) {} : (tensor<3xf32>) -> tensor<3xi1>
  func.return %0 : tensor<3xi1>
}

// -----

func.func @is_finite_int_input(%arg0: tensor<3xi32>) -> tensor<3xi1> {
  // expected-error@+1 {{operand #0 must be tensor of f8E4M3FN type or f8E5M2 type or 16-bit float or 32-bit float or 64-bit float or bfloat16 type values, but got 'tensor<3xi32>'}}
  %0 = "mhlo.is_finite"(%arg0) {} : (tensor<3xi32>) -> tensor<3xi1>
  func.return %0 : tensor<3xi1>
}

// -----

func.func @is_finite_mismatch_return_element_type(%arg0: tensor<3xf32>) -> tensor<3xi10> {
  // expected-error@+1 {{result #0 must be tensor of pred (AKA boolean or 1-bit integer) values, but got 'tensor<3xi10>'}}
  %0 = "mhlo.is_finite"(%arg0) {} : (tensor<3xf32>) -> tensor<3xi10>
  func.return %0 : tensor<3xi10>
}

// -----

func.func @is_finite_mismatch_return_shape(%arg0: tensor<3xf32>) -> tensor<4xi1> {
  // expected-error@+1 {{all non-scalar operands/results must have the same shape and base type}}
  %0 = "mhlo.is_finite"(%arg0) {} : (tensor<3xf32>) -> tensor<4xi1>
  func.return %0 : tensor<4xi1>
}

// -----

func.func @negative_dimension_attr(%arg0: tensor<?x?xf32, #mhlo.type_extensions<bounds = [3, -1]>>, %arg1: tensor<i32>) -> tensor<*xf32> {
  // expected-error@+1 {{requires dimension attribute in range [0, 2); found (-1)}}
  %result = "mhlo.set_dimension_size"(%arg0, %arg1) {dimension = -1 : i64} : (tensor<?x?xf32, #mhlo.type_extensions<bounds = [3, -1]>>, tensor<i32>) -> tensor<*xf32>
  func.return %result : tensor<*xf32>
}

// -----

func.func @invalid_dimension_attr(%arg0: tensor<?x?xf32, #mhlo.type_extensions<bounds = [3, -1]>>, %arg1: tensor<i32>) -> tensor<*xf32> {
  // expected-error@+1 {{requires dimension attribute in range [0, 2); found (2)}}
  %result = "mhlo.set_dimension_size"(%arg0, %arg1) {dimension = 2 : i64} : (tensor<?x?xf32, #mhlo.type_extensions<bounds = [3, -1]>>, tensor<i32>) -> tensor<*xf32>
  func.return %result : tensor<*xf32>
}

// -----

func.func @f8e4m3fn(%arg0: tensor<f16>) -> tensor<f8E4M3FN> {
  %0 = "mhlo.convert"(%arg0) : (tensor<f16>) -> tensor<f8E4M3FN>
  func.return %0 : tensor<f8E4M3FN>
}

// -----

func.func @f8e5m2(%arg0: tensor<f16>) -> tensor<f8E5M2> {
  %0 = "mhlo.convert"(%arg0) : (tensor<f16>) -> tensor<f8E5M2>
  func.return %0 : tensor<f8E5M2>
}
