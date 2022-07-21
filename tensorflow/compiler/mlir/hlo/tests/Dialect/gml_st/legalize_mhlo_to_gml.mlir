// RUN: mlir-hlo-opt %s --legalize-mhlo-to-gml | FileCheck %s

// CHECK-LABEL: @dynamic_broadcast_in_dim
// CHECK-SAME:  %[[ARG:.*]]: tensor<?x?xf32>, %[[SHAPE:.*]]: tensor<3xindex>
func.func @dynamic_broadcast_in_dim(%arg : tensor<?x?xf32>, %shape : tensor<3xindex>) -> tensor<?x?x?xf32> {
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0
  // CHECK-DAG: %[[C1:.*]] = arith.constant 1
  // CHECK-DAG: %[[C2:.*]] = arith.constant 2
  // CHECK-DAG: %[[SHAPE_D0:.*]] = tensor.extract %[[SHAPE]][%[[C0]]]
  // CHECK-DAG: %[[SHAPE_D1:.*]] = tensor.extract %[[SHAPE]][%[[C1]]]
  // CHECK-DAG: %[[SHAPE_D2:.*]] = tensor.extract %[[SHAPE]][%[[C2]]]
  // CHECK-DAG: %[[INIT:.*]] = linalg.init_tensor [%[[SHAPE_D0]], %[[SHAPE_D1]], %[[SHAPE_D2]]] : tensor<?x?x?xf32>
  // CHECK-NEXT: %[[BCAST:.*]] = gml_st.dynamic_broadcast_in_dim
  // CHECK-SAME: ins(%[[ARG]] : tensor<?x?xf32>)
  // CHECK-SAME: outs(%[[INIT]] : tensor<?x?x?xf32>)
  // CHECK-SAME: {broadcast_dimensions = [:i64 0, 2]}
  // CHECK:     return %[[BCAST]]
  %0 = "mhlo.dynamic_broadcast_in_dim"(%arg, %shape)
      { broadcast_dimensions = dense<[0, 2]> : tensor<2xi64> }
      : (tensor<?x?xf32>, tensor<3xindex>) -> tensor<?x?x?xf32>
  func.return %0 : tensor<?x?x?xf32>
}

func.func @simple_gather(%operand : tensor<3x3xf32>,
                         %indices: tensor<3x2xi64>) -> tensor<3xf32> {
  %0 = "mhlo.gather"(%operand, %indices) {
    dimension_numbers = #mhlo.gather<
      collapsed_slice_dims = [0, 1],
      index_vector_dim = 1,
      offset_dims = [],
      start_index_map = [0, 1]
    >,
    indices_are_sorted = false,
    slice_sizes = dense<[1, 1]> : tensor<2xi64>
  } : (tensor<3x3xf32>, tensor<3x2xi64>) -> tensor<3xf32>
  func.return %0 : tensor<3xf32>
}

// CHECK-LABEL: @simple_gather
//       CHECK: %[[INIT:.*]] = linalg.init_tensor [3] : tensor<3xf32>
//       CHECK: %[[GATHER:.*]] = gml_st.gather
//  CHECK-SAME:   ins(%arg0 : tensor<3x3xf32>, %arg1 : tensor<3x2xi64>)
//  CHECK-SAME:   outs(%[[INIT]] : tensor<3xf32>)
//       CHECK: return %[[GATHER]]

func.func @unsupported_gather(%operand : tensor<3x3xf32>,
                              %indices: tensor<3x2xi64>) -> tensor<3xf32> {
  %0 = "mhlo.gather"(%operand, %indices) {
    dimension_numbers = #mhlo.gather<
      collapsed_slice_dims = [0, 1],
      index_vector_dim = 1,
      offset_dims = [],
      start_index_map = [1, 0]
    >,
    indices_are_sorted = false,
    slice_sizes = dense<[1, 1]> : tensor<2xi64>
  } : (tensor<3x3xf32>, tensor<3x2xi64>) -> tensor<3xf32>
  func.return %0 : tensor<3xf32>
}

// CHECK-LABEL: @unsupported_gather
//       CHECK: mhlo.gather

func.func @simple_scatter(%arg0: tensor<3xi32>, %arg1: tensor<1x1xi32>,
                          %arg2: tensor<1xi32>) -> tensor<3xi32> {
  %0 = "mhlo.scatter"(%arg0, %arg1, %arg2) ({
  ^bb0(%arg3: tensor<i32>, %arg4: tensor<i32>):
    %sum = mhlo.add %arg3, %arg4 : tensor<i32>
    "mhlo.return"(%sum) : (tensor<i32>) -> ()
  }) {
    scatter_dimension_numbers = #mhlo.scatter<
      update_window_dims = [],
      inserted_window_dims = [0],
      scatter_dims_to_operand_dims = [0],
      index_vector_dim = 1,
    >,
    unique_indices = false,
    indices_are_sorted = false
  } : (tensor<3xi32>, tensor<1x1xi32>, tensor<1xi32>) -> tensor<3xi32>
  func.return %0 : tensor<3xi32>
}

// CHECK-LABEL: @simple_scatter
//       CHECK: %[[INIT:.*]] = linalg.init_tensor [3] : tensor<3xi32>
//       CHECK: gml_st.scatter ins(%{{.*}} : tensor<3xi32>,
//  CHECK-SAME:                    %{{.*}} : tensor<1x1xi32>,
//  CHECK-SAME:                    %{{.*}} : tensor<1xi32>)
//  CHECK-SAME:                outs(%0 : tensor<3xi32>)

func.func @scatter_wrong_update(%arg0: tensor<3xi32>, %arg1: tensor<1x1xi32>,
                          %arg2: tensor<1xi32>) -> tensor<3xi32> {
  %0 = "mhlo.scatter"(%arg0, %arg1, %arg2) ({
  ^bb0(%arg3: tensor<i32>, %arg4: tensor<i32>):
    %sum = mhlo.subtract %arg3, %arg4 : tensor<i32>
    "mhlo.return"(%sum) : (tensor<i32>) -> ()
  }) {
    scatter_dimension_numbers = #mhlo.scatter<
      update_window_dims = [],
      inserted_window_dims = [0],
      scatter_dims_to_operand_dims = [0],
      index_vector_dim = 1,
    >,
    unique_indices = false,
    indices_are_sorted = false
  } : (tensor<3xi32>, tensor<1x1xi32>, tensor<1xi32>) -> tensor<3xi32>
  func.return %0 : tensor<3xi32>
}

// CHECK-LABEL: @scatter_wrong_update
//       CHECK: mhlo.scatter

func.func @scatter_wrong_index_vector_dim(
              %arg0: tensor<3xi32>,
              %arg1: tensor<1xi32>,
              %arg2: tensor<1xi32>) -> tensor<3xi32> {
  %0 = "mhlo.scatter"(%arg0, %arg1, %arg2) ({
  ^bb0(%arg3: tensor<i32>, %arg4: tensor<i32>):
    %sum = mhlo.add %arg3, %arg4 : tensor<i32>
    "mhlo.return"(%sum) : (tensor<i32>) -> ()
  }) {
    scatter_dimension_numbers = #mhlo.scatter<
      update_window_dims = [],
      inserted_window_dims = [0],
      scatter_dims_to_operand_dims = [0],
      index_vector_dim = 1,
    >,
    unique_indices = false,
    indices_are_sorted = false
  } : (tensor<3xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3xi32>
  func.return %0 : tensor<3xi32>
}

// CHECK-LABEL: @scatter_wrong_index_vector_dim
//       CHECK: mhlo.scatter

func.func @scatter_wrong_dims(%arg0: tensor<3xi32>, %arg1: tensor<1x1xi32>,
                          %arg2: tensor<3x1xi32>) -> tensor<3xi32> {
  %0 = "mhlo.scatter"(%arg0, %arg1, %arg2) ({
  ^bb0(%arg3: tensor<i32>, %arg4: tensor<i32>):
    %sum = mhlo.add %arg3, %arg4 : tensor<i32>
    "mhlo.return"(%sum) : (tensor<i32>) -> ()
  }) {
    scatter_dimension_numbers = #mhlo.scatter<
      update_window_dims = [0],
      inserted_window_dims = [],
      scatter_dims_to_operand_dims = [0],
      index_vector_dim = 1,
    >,
    unique_indices = false,
    indices_are_sorted = false
  } : (tensor<3xi32>, tensor<1x1xi32>, tensor<3x1xi32>) -> tensor<3xi32>
  func.return %0 : tensor<3xi32>
}

// CHECK-LABEL: @scatter_wrong_dims
//       CHECK: mhlo.scatter
