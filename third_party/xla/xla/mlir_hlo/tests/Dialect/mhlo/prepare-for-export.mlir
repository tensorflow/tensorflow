// RUN: mlir-hlo-opt -xla-prepare-for-export %s | FileCheck %s

// CHECK-LABEL: func @splat_constants
func.func @splat_constants() -> tensor<1x64x224x224xf32> {
  %cst = mhlo.constant dense<0.000000e+00> : tensor<1x64x224x224xf32>
  func.return %cst : tensor<1x64x224x224xf32>
  // CHECK: %[[CST:.*]] = mhlo.constant dense<0.000000e+00> : tensor<f32>
  // CHECK: "mhlo.broadcast_in_dim"(%[[CST]])
  // CHECK-SAME: (tensor<f32>) -> tensor<1x64x224x224xf32>
}

// -----

// CHECK-LABEL: @splat_constant_complex_float
func.func @splat_constant_complex_float() -> tensor<128x1014x508xcomplex<f64>> {
// CHECK: %[[CST:.*]] = mhlo.constant dense<(1.000000e+00,2.000000e+00)> : tensor<complex<f64>>
// CHECK: %[[BCAST:.*]] = "mhlo.broadcast_in_dim"(%[[CST]]
// CHECK: return %[[BCAST]]
  %0 = mhlo.constant dense<(1.000000e+00,2.000000e+00)> : tensor<128x1014x508xcomplex<f64>>
  func.return %0 : tensor<128x1014x508xcomplex<f64>>
}

// -----

// CHECK-LABEL: @broadcast_in_dim_dimension_unsorted
func.func @broadcast_in_dim_dimension_unsorted(%arg0: tensor<1x2xi32>) -> tensor<1x2x3xi32> {
// Unfuse the transpose from the broadcastInDim before export.
// CHECK: %[[TRANSPOSE:.*]] = "mhlo.transpose"(%arg0){{.*}}permutation = dense<[1, 0]>{{.*}} -> tensor<2x1xi32>
// CHECK: mhlo.broadcast_in_dim"(%[[TRANSPOSE]]){{.*}}broadcast_dimensions = dense<[1, 2]>
  %0 = "mhlo.broadcast_in_dim"(%arg0) <{broadcast_dimensions = dense<[2, 1]> : tensor<2xi64>}> : (tensor<1x2xi32>) -> tensor<1x2x3xi32>
  func.return %0 : tensor<1x2x3xi32>
}

// -----

// CHECK-LABEL: @reduce_with_multiple_implicit_captures
func.func @reduce_with_multiple_implicit_captures(%arg0: tensor<2x2xf32>) -> tuple<tensor<i1>> {
  %0 = mhlo.constant dense<1.000000e+00> : tensor<f32>
  %1 = mhlo.constant dense<0.000000e+00> : tensor<f32>
  // CHECK: mhlo.reduce
  %2 = mhlo.reduce(%arg0 init: %1) across dimensions = [0, 1] : (tensor<2x2xf32>, tensor<f32>) -> tensor<f32>
   reducer(%arg1: tensor<f32>, %arg2: tensor<f32>)  {
    // CHECK-DAG: mhlo.constant dense<0.000000e+00> : tensor<f32>
    // CHECK-DAG: mhlo.constant dense<1.000000e+00> : tensor<f32>
    // CHECK: mhlo.compare
    %5 = mhlo.compare  NE, %arg1, %1 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %6 = mhlo.compare  NE, %arg2, %1 : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %7 = mhlo.or %5, %6 : tensor<i1>
    %8 = mhlo.select %7, %0, %1 : tensor<i1>, tensor<f32>
    mhlo.return %8 : tensor<f32>
  }
  %3 = mhlo.compare  NE, %2, %1 : (tensor<f32>, tensor<f32>) -> tensor<i1>
  %4 = mhlo.tuple %3 {xla_shape = "(pred[])"} : tuple<tensor<i1>>
  return %4 : tuple<tensor<i1>>
}

// -----

// CHECK-LABEL: @all_reduce_with_implicit_capture
func.func @all_reduce_with_implicit_capture(%arg0: tensor<f32>) -> tensor<f32> {
    %c = mhlo.constant dense<0.0> : tensor<f32>
    // CHECK: mhlo.all_reduce
    // CHECK-NEXT: ^[[BB:bb.*]](%[[ARG1:arg.*]]: tensor<f32>, %[[ARG2:arg.*]]: tensor<f32>):
    %0 = "mhlo.all_reduce"(%arg0) ({
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
    // CHECK: %[[VAL1:.*]] = mhlo.constant dense<0.000000e+00> : tensor<f32>
    // CHECK: mhlo.add
    // CHECK-SAME: %[[ARG1]], %[[VAL1]]
      %1 = mhlo.add %arg1, %c : tensor<f32>
      mhlo.return %1 : tensor<f32>
    }) {replica_groups = dense<[[0], [1]]> : tensor<2x1xi64>} : (tensor<f32>) -> tensor<f32>
    return %0 : tensor<f32>
  }

// -----

// CHECK-LABEL: @reduce_scatter_with_implicit_capture
func.func @reduce_scatter_with_implicit_capture(%data: tensor<4x16xf32>) -> tensor<4x4xf32> {
  %c = mhlo.constant dense<0.0> : tensor<f32>
  // CHECK: mhlo.reduce_scatter
  // CHECK-NEXT: ^[[BB:bb.*]](%[[ARG1:arg.*]]: tensor<f32>, %[[ARG2:arg.*]]: tensor<f32>):
  %0 = "mhlo.reduce_scatter"(%data) ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
    // CHECK: %[[VAL1:.*]] = mhlo.constant dense<0.000000e+00> : tensor<f32>
    // CHECK: mhlo.add
    // CHECK-SAME: %[[ARG1]], %[[VAL1]]
    %1 = mhlo.add %arg2, %c : tensor<f32>
    "mhlo.return"(%1) : (tensor<f32>) -> ()
  }) {replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>,
      scatter_dimension = 1 : i64,
      channel_handle = #mhlo.channel_handle<handle = 1, type = 0>,
      use_global_device_ids} : (tensor<4x16xf32>) -> tensor<4x4xf32>
  func.return %0 : tensor<4x4xf32>
}

// -----

// CHECK-LABEL: @reduce_window_with_implicit_capture
func.func @reduce_window_with_implicit_capture(%arg0: tensor<2x17x31x7xf32>, %arg1: tensor<f32>) -> tensor<2x16x30x7xf32> {
    %c = mhlo.constant dense<0.0> : tensor<f32>
    // CHECK: mhlo.reduce_window
    // CHECK-NEXT: ^[[BB:bb.*]](%[[ARG2:arg.*]]: tensor<f32>, %[[ARG3:arg.*]]: tensor<f32>):
    %0 = "mhlo.reduce_window"(%arg0, %arg1) ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      // CHECK: %[[VAL1:.*]] = mhlo.constant dense<0.000000e+00> : tensor<f32>
      // CHECK: mhlo.maximum
      // CHECK-SAME: %[[ARG2]], %[[VAL1]]
      %1 = mhlo.maximum %arg2, %c : tensor<f32>
      mhlo.return %1 : tensor<f32>
    }) {window_dimensions = dense<[1, 2, 2, 1]> : tensor<4xi64>} : (tensor<2x17x31x7xf32>, tensor<f32>) -> tensor<2x16x30x7xf32>
    return %0 : tensor<2x16x30x7xf32>
  }

// -----

// CHECK-LABEL: @scatter_with_implicit_capture
func.func @scatter_with_implicit_capture(%arg0: tensor<3xi32>, %arg1: tensor<1x1xi32>,
                            %arg2: tensor<1xi32>) -> tensor<3xi32> {
 %c = mhlo.constant dense<0> : tensor<i32>
 // CHECK: mhlo.scatter
 // CHECK-NEXT: ^[[BB:bb.*]](%[[ARG3:arg.*]]: tensor<i32>, %[[ARG4:arg.*]]: tensor<i32>):
  %0 = "mhlo.scatter"(%arg0, %arg1, %arg2) ({
  ^bb0(%arg3: tensor<i32>, %arg4: tensor<i32>):
    // CHECK: %[[VAL1:.*]] = mhlo.constant dense<0> : tensor<i32>
    // CHECK: mhlo.add
    // CHECK-SAME: %[[ARG4]], %[[VAL1]]
    %x = mhlo.add %arg4, %c : tensor<i32>
    "mhlo.return"(%x) : (tensor<i32>) -> ()
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

// CHECK-LABEL: @select_and_scatter_with_implicit_capture
func.func @select_and_scatter_with_implicit_capture(%arg0: tensor<10x24x24x64xf32>, %arg1: tensor<10x23x23x64xf32>, %arg2: tensor<f32>) -> tensor<10x24x24x64xf32> {
    %c1 = mhlo.constant dense<0.0> : tensor<f32>
    %c2 = mhlo.constant dense<0.0> : tensor<f32>
    // CHECK: mhlo.select_and_scatter
    // CHECK-NEXT: ^[[BB:bb.*]](%[[ARG3:arg.*]]: tensor<f32>, %[[ARG4:arg.*]]: tensor<f32>):
    %0 = "mhlo.select_and_scatter"(%arg0, %arg1, %arg2) ({
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      // CHECK: %[[VAL1:.*]] = mhlo.constant dense<0.000000e+00> : tensor<f32>
      // CHECK: mhlo.compare
      // CHECK-SAME: %[[ARG3]], %[[VAL1]]
      %1 = mhlo.compare  GE, %arg3, %c1,  TOTALORDER : (tensor<f32>, tensor<f32>) -> tensor<i1>
      mhlo.return %1 : tensor<i1>
    }, {
    // CHECK: ^[[BB:bb.*]](%[[ARG3:arg.*]]: tensor<f32>, %[[ARG4:arg.*]]: tensor<f32>):
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      // CHECK: %[[VAL2:.*]] = mhlo.constant dense<0.000000e+00> : tensor<f32>
      // CHECK: mhlo.add
      // CHECK-SAME: %[[ARG4]], %[[VAL2]]
      %1 = mhlo.add %arg4, %c2 : tensor<f32>
      mhlo.return %1 : tensor<f32>
    }) {window_dimensions = dense<[1, 2, 2, 1]> : tensor<4xi64>} : (tensor<10x24x24x64xf32>, tensor<10x23x23x64xf32>, tensor<f32>) -> tensor<10x24x24x64xf32>
    return %0 : tensor<10x24x24x64xf32>
  }

// -----

// CHECK-LABEL: @sort_with_implicit_capture
func.func @sort_with_implicit_capture(%input0: tensor<16x16xf32>, %input1: tensor<16x16xi32>) {
  %c = mhlo.constant dense<0.0> : tensor<f32>
  // CHECK: mhlo.sort
  // CHECK-NEXT: ^[[BB:bb.*]](%[[ARG0:arg.*]]: tensor<f32>, %[[ARG1:arg.*]]: tensor<f32>, %[[ARG2:arg.*]]: tensor<i32>, %[[ARG3:arg.*]]: tensor<i32>):
  %0:2 = "mhlo.sort"(%input0, %input1) ({
  ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>, %arg2: tensor<i32>, %arg3: tensor<i32>):
    // CHECK: %[[VAL1:.*]] = mhlo.constant dense<0.000000e+00> : tensor<f32>
    // CHECK: mhlo.compare
    // CHECK-SAME: %[[ARG0]], %[[VAL1]]
    %7 = "mhlo.compare"(%arg0, %c) {comparison_direction = #mhlo<comparison_direction GT>} : (tensor<f32>, tensor<f32>) -> tensor<i1>
    "mhlo.return"(%7) : (tensor<i1>) -> ()
  }) {dimension = 1 : i64, is_stable = true} : (tensor<16x16xf32>, tensor<16x16xi32>) -> (tensor<16x16xf32>, tensor<16x16xi32>)
  func.return
}
