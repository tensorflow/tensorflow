// RUN: mlir-hlo-opt %s --legalize-mhlo-to-thlo=enable-experimental=true | FileCheck %s

// CHECK-LABEL: @dynamic_broadcast_in_dim
// CHECK-SAME:  %[[ARG:.*]]: tensor<?x?xf32>, %[[SHAPE:.*]]: tensor<3xindex>
func.func @dynamic_broadcast_in_dim(%arg : tensor<?x?xf32>, %shape : tensor<3xindex>) -> tensor<?x?x?xf32> {
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0
  // CHECK-DAG: %[[C1:.*]] = arith.constant 1
  // CHECK-DAG: %[[C2:.*]] = arith.constant 2
  // CHECK-DAG: %[[SHAPE_D0:.*]] = tensor.extract %[[SHAPE]][%[[C0]]]
  // CHECK-DAG: %[[SHAPE_D1:.*]] = tensor.extract %[[SHAPE]][%[[C1]]]
  // CHECK-DAG: %[[SHAPE_D2:.*]] = tensor.extract %[[SHAPE]][%[[C2]]]
  // CHECK-DAG: %[[INIT:.*]] = tensor.empty(%[[SHAPE_D0]], %[[SHAPE_D1]], %[[SHAPE_D2]]) : tensor<?x?x?xf32>
  // CHECK-NEXT: %[[BCAST:.*]] = thlo.dynamic_broadcast_in_dim
  // CHECK-NEXT: ins(%[[ARG]] : tensor<?x?xf32>)
  // CHECK-NEXT: outs(%[[INIT]] : tensor<?x?x?xf32>)
  // CHECK-NEXT: broadcast_dimensions = [0, 2]
  // CHECK:     return %[[BCAST]]
  %0 = "mhlo.dynamic_broadcast_in_dim"(%arg, %shape)
      { broadcast_dimensions = dense<[0, 2]> : tensor<2xi64> }
      : (tensor<?x?xf32>, tensor<3xindex>) -> tensor<?x?x?xf32>
  func.return %0 : tensor<?x?x?xf32>
}

// CHECK-LABEL: @dynamic_broadcast_in_dim_expansion_behavior_known
// CHECK-SAME:  %[[ARG:.*]]: tensor<?x?xf32>, %[[SHAPE:.*]]: tensor<3xindex>
func.func @dynamic_broadcast_in_dim_expansion_behavior_known(
    %arg : tensor<?x?xf32>, %shape : tensor<3xindex>) -> tensor<?x?x?xf32> {
  // CHECK:       %[[BCAST:.*]] = "mhlo.dynamic_broadcast_in_dim"(%[[ARG]], %[[SHAPE]])
  // CHECK:       return %[[BCAST]]
  %0 = "mhlo.dynamic_broadcast_in_dim"(%arg, %shape) {
      broadcast_dimensions = dense<[0, 2]> : tensor<2xi64>,
      known_expanding_dimensions = dense<[0]> : tensor<1xi64>,
      known_nonexpanding_dimensions = dense<[1]> : tensor<1xi64> }
      : (tensor<?x?xf32>, tensor<3xindex>) -> tensor<?x?x?xf32>
  func.return %0 : tensor<?x?x?xf32>
}

// CHECK-LABEL: @dynamic_broadcast_in_dim_with_known_expanding
// CHECK-SAME:  %[[ARG:.*]]: tensor<?x?x?xf32>, %[[SHAPE:.*]]: tensor<4xindex>
func.func @dynamic_broadcast_in_dim_with_known_expanding(%arg : tensor<?x?x?xf32>, %shape : tensor<4xindex>) -> tensor<?x?x?x?xf32> {
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0
  // CHECK-DAG: %[[C1:.*]] = arith.constant 1
  // CHECK-DAG: %[[C2:.*]] = arith.constant 2
  // CHECK-DAG: %[[C3:.*]] = arith.constant 3
  // CHECK-DAG: %[[SHAPE_D0:.*]] = tensor.extract %[[SHAPE]][%[[C0]]]
  // CHECK-DAG: %[[SHAPE_D1:.*]] = tensor.extract %[[SHAPE]][%[[C1]]]
  // CHECK-DAG: %[[SHAPE_D2:.*]] = tensor.extract %[[SHAPE]][%[[C2]]]
  // CHECK-DAG: %[[SHAPE_D3:.*]] = tensor.extract %[[SHAPE]][%[[C3]]]
  // CHECK-DAG: %[[INIT:.*]] = tensor.empty(%[[SHAPE_D0]], %[[SHAPE_D1]], %[[SHAPE_D2]], %[[SHAPE_D3]]) : tensor<?x?x?x?xf32>
  // CHECK-NEXT: %[[BCAST:.*]] = thlo.dynamic_broadcast_in_dim
  // CHECK-NEXT: ins(%[[ARG]] : tensor<?x?x?xf32>)
  // CHECK-NEXT: outs(%[[INIT]] : tensor<?x?x?x?xf32>)
  // CHECK-NEXT: broadcast_dimensions = [0, 2, 3]
  // CHECK-SAME: {known_expanding_dimensions = array<i64: 0>, known_nonexpanding_dimensions = array<i64: 2>}
  // CHECK:     return %[[BCAST]]
  %0 = "mhlo.dynamic_broadcast_in_dim"(%arg, %shape) {
      broadcast_dimensions = dense<[0, 2, 3]> : tensor<3xi64>,
      known_expanding_dimensions = dense<[0]> : tensor<1xi64>,
      known_nonexpanding_dimensions = dense<[2]> : tensor<1xi64> }
      : (tensor<?x?x?xf32>, tensor<4xindex>) -> tensor<?x?x?x?xf32>
  func.return %0 : tensor<?x?x?x?xf32>
}

// CHECK-LABEL: @concatenate
// CHECK-SAME:  %[[A:.*]]: tensor<?x?xi32>, %[[B:.*]]: tensor<?x?xi32>, %[[C:.*]]: tensor<?x?xi32>
func.func @concatenate(%a: tensor<?x?xi32>, %b: tensor<?x?xi32>, %c: tensor<?x?xi32>) -> tensor<?x?xi32> {
  // CHECK-DAG:  %[[C0:.*]] = arith.constant 0
  // CHECK-DAG:  %[[C1:.*]] = arith.constant 1
  // CHECK-DAG:  %[[D0:.*]] = tensor.dim %[[A]], %[[C0]]
  // CHECK-DAG:  %[[CONCAT_DIM_A:.*]] = tensor.dim %[[A]], %[[C1]]
  // CHECK-DAG:  %[[CONCAT_DIM_B:.*]] = tensor.dim %[[B]], %[[C1]]
  // CHECK-DAG:  %[[CONCAT_DIM_C:.*]] = tensor.dim %[[C]], %[[C1]]
  // CHECK-DAG:  %[[CONCAT_DIM_AB:.*]] = arith.addi %[[CONCAT_DIM_A]], %[[CONCAT_DIM_B]]
  // CHECK-DAG:  %[[CONCAT_DIM_ABC:.*]] = arith.addi %[[CONCAT_DIM_AB]], %[[CONCAT_DIM_C]]
  // CHECK-DAG:  %[[INIT:.*]] = tensor.empty(%[[D0]], %[[CONCAT_DIM_ABC]])
  // CHECK:      %[[CONCATENATE:.*]] = thlo.concatenate
  // CHECK-NEXT:     ins(%[[A]] : tensor<?x?xi32>, %[[B]] : tensor<?x?xi32>, %[[C]] : tensor<?x?xi32>)
  // CHECK-NEXT:     outs(%[[INIT]] : tensor<?x?xi32>)
  // CHECK-NEXT:     dimension = 1
  // CHECK:      return %[[CONCATENATE]]
  %concat = "mhlo.concatenate"(%a, %b, %c) { dimension = 1 } : (tensor<?x?xi32>, tensor<?x?xi32>, tensor<?x?xi32>) -> tensor<?x?xi32>
  func.return %concat : tensor<?x?xi32>
}

// CHECK-LABEL: @concatenate_with_static_info
// CHECK-SAME:  %[[A:.*]]: tensor<64x32xi32>, %[[B:.*]]: tensor<64x16xi32>, %[[C:.*]]: tensor<64x?xi32>
func.func @concatenate_with_static_info(%a: tensor<64x32xi32>, %b: tensor<64x16xi32>, %c: tensor<64x?xi32>) -> tensor<64x?xi32> {
  // CHECK-DAG:  %[[C1:.*]] = arith.constant 1
  // CHECK-DAG:  %[[C48:.*]] = arith.constant 48
  // CHECK-DAG:  %[[CONCAT_DIM_C:.*]] = tensor.dim %[[C]], %[[C1]]
  // CHECK-DAG:  %[[CONCAT_DIM_SUM:.*]] = arith.addi %[[CONCAT_DIM_C]], %[[C48]]
  // CHECK-DAG:  %[[INIT:.*]] = tensor.empty(%[[CONCAT_DIM_SUM]])
  // CHECK:      %[[CONCAT:.*]] = thlo.concatenate
  // CHECK-NEXT:     ins(%[[A]] : tensor<64x32xi32>, %[[B]] : tensor<64x16xi32>, %[[C]] : tensor<64x?xi32>)
  // CHECK-NEXT:     outs(%[[INIT]] : tensor<64x?xi32>)
  // CHECK-NEXT:     dimension = 1
  // CHECK:      return %[[CONCAT]]
  %concat = "mhlo.concatenate"(%a, %b, %c) { dimension = 1 } : (tensor<64x32xi32>, tensor<64x16xi32>, tensor<64x?xi32>) -> tensor<64x?xi32>
  func.return %concat : tensor<64x?xi32>
}

func.func @simple_gather(%operand : tensor<3x3xf32>,
                         %indices: tensor<3x2xi64>) -> tensor<3x1x1xf32> {
  %0 = "mhlo.gather"(%operand, %indices) {
    dimension_numbers = #mhlo.gather<
      collapsed_slice_dims = [],
      index_vector_dim = 1,
      offset_dims = [1, 2],
      start_index_map = [0, 1]
    >,
    indices_are_sorted = false,
    slice_sizes = dense<[1, 1]> : tensor<2xi64>
  } : (tensor<3x3xf32>, tensor<3x2xi64>) -> tensor<3x1x1xf32>
  func.return %0 : tensor<3x1x1xf32>
}

// CHECK-LABEL: @simple_gather
//   CHECK-DAG: %[[CAST:.*]] = arith.index_cast {{.*}} : tensor<3x2xi64> to tensor<3x2xindex>
//   CHECK-DAG: %[[INIT:.*]] = tensor.empty() : tensor<3x1x1xf32>
//       CHECK: %[[GATHER:.*]] = thlo.gather
//  CHECK-NEXT:   ins(%{{.*}} : tensor<3x3xf32>, %[[CAST]] : tensor<3x2xindex>)
//  CHECK-NEXT:   outs(%[[INIT]] : tensor<3x1x1xf32>)
//       CHECK: return %[[GATHER]]

func.func @simple_gather_unsigned(
    %operand : tensor<3x3xui32>, %indices: tensor<3x2xui64>) -> tensor<3x1x1xui32> {
  %0 = "mhlo.gather"(%operand, %indices) {
    dimension_numbers = #mhlo.gather<
      collapsed_slice_dims = [],
      index_vector_dim = 1,
      offset_dims = [1, 2],
      start_index_map = [0, 1]
    >,
    indices_are_sorted = false,
    slice_sizes = dense<[1, 1]> : tensor<2xi64>
  } : (tensor<3x3xui32>, tensor<3x2xui64>) -> tensor<3x1x1xui32>
  func.return %0 : tensor<3x1x1xui32>
}
// CHECK-LABEL: @simple_gather_unsigned
//   CHECK-DAG: %[[CAST:.*]] = builtin.unrealized_conversion_cast {{.*}} : tensor<3x3xui32> to tensor<3x3xi32>
//   CHECK-DAG: %[[INIT:.*]] = tensor.empty() : tensor<3x1x1xi32>
//   CHECK-DAG: %[[INDEX_CAST:.*]] = arith.index_castui {{.*}} to tensor<3x2xindex>
//       CHECK: %[[GATHER:.*]] = thlo.gather
//  CHECK-NEXT:   ins(%[[CAST]] : tensor<3x3xi32>, %[[INDEX_CAST]] : tensor<3x2xindex>)
//  CHECK-NEXT:   outs(%[[INIT]] : tensor<3x1x1xi32>)
//       CHECK: %[[CAST2:.*]] = builtin.unrealized_conversion_cast %[[GATHER]] : tensor<3x1x1xi32> to tensor<3x1x1xui32>
//       CHECK: return %[[CAST2]]

func.func @gather_with_slices(
    %operand : tensor<300x300xi32>, %indices: tensor<3x2xi64>) -> tensor<3x101x102xi32> {
  %0 = "mhlo.gather"(%operand, %indices) {
    dimension_numbers = #mhlo.gather<
      collapsed_slice_dims = [],
      index_vector_dim = 1,
      offset_dims = [1, 2],
      start_index_map = [0, 1]
    >,
    indices_are_sorted = false,
    slice_sizes = dense<[101, 102]> : tensor<2xi64>
  } : (tensor<300x300xi32>, tensor<3x2xi64>) -> tensor<3x101x102xi32>
  func.return %0 : tensor<3x101x102xi32>
}
// CHECK-LABEL: @gather_with_slices
//       CHECK: %[[INIT:.*]] = tensor.empty() : tensor<3x101x102xi32>
//       CHECK: thlo.gather
//       CHECK:   outs(%[[INIT]] : tensor<3x101x102xi32>)

func.func @gather_dynamic(
    %operand : tensor<300xi32>, %indices: tensor<?x1xi64>) -> tensor<?x42xi32> {
  %0 = "mhlo.gather"(%operand, %indices) {
    dimension_numbers = #mhlo.gather<
      collapsed_slice_dims = [],
      index_vector_dim = 1,
      offset_dims = [1],
      start_index_map = [0]
    >,
    indices_are_sorted = false,
    slice_sizes = dense<[42]> : tensor<1xi64>
  } : (tensor<300xi32>, tensor<?x1xi64>) -> tensor<?x42xi32>
  func.return %0 : tensor<?x42xi32>
}
// CHECK-LABEL: @gather_dynamic
//   CHECK-DAG: %[[C0:.*]] = arith.constant 0
//       CHECK: %[[DIM:.*]] = tensor.dim {{.*}} %[[C0]] : tensor<?x1xi64>
//       CHECK: %[[INIT:.*]] = tensor.empty(%dim) : tensor<?x42xi32>
//       CHECK: thlo.gather
//       CHECK:   outs(%[[INIT]] : tensor<?x42xi32>)

func.func @unsupported_gather(%operand: tensor<3x3xf32>,
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

func.func @simple_scatter(%dst: tensor<3x3xf32>, %indices: tensor<2x2xi32>,
                          %update: tensor<2x1x3xf32>) -> tensor<3x3xf32> {
  %0 = "mhlo.scatter"(%dst, %indices, %update) ({
  ^bb0(%in: tensor<f32>, %out: tensor<f32>):
    %sum = mhlo.add %in, %out : tensor<f32>
    "mhlo.return"(%sum) : (tensor<f32>) -> ()
  }) {
    scatter_dimension_numbers = #mhlo.scatter<
      update_window_dims = [1, 2],
      inserted_window_dims = [],
      scatter_dims_to_operand_dims = [0, 1],
      index_vector_dim = 1,
    >,
    unique_indices = false,
    indices_are_sorted = false
  } : (tensor<3x3xf32>, tensor<2x2xi32>, tensor<2x1x3xf32>) -> tensor<3x3xf32>
  func.return %0 : tensor<3x3xf32>
}

// CHECK-LABEL: @simple_scatter
// CHECK-SAME: (%[[DST:.*]]: tensor<3x3xf32>, %[[INDICES:.*]]: tensor<2x2xi32>,
// CHECK-SAME:  %[[UPDATE:.*]]: tensor<2x1x3xf32>)
//      CHECK:   %[[CAST:.*]] = arith.index_cast %[[INDICES]] {{.*}} to tensor<2x2xindex>
//      CHECK:   thlo.scatter 
// CHECK-NEXT:     ins(%[[CAST]] : tensor<2x2xindex>,
// CHECK-SAME:        %[[UPDATE]] : tensor<2x1x3xf32>)
// CHECK-NEXT:     outs(%[[DST]] : tensor<3x3xf32>)
// CHECK-NEXT:     (%[[UPD:.*]]: f32, %[[CUR:.*]]: f32) {
// CHECK-NEXT:    %[[CUR_T:.*]] = tensor.from_elements %[[CUR]] : tensor<f32>
// CHECK-NEXT:    %[[UPD_T:.*]] = tensor.from_elements %[[UPD]] : tensor<f32>
// CHECK-NEXT:    %[[CUR:.*]] = tensor.extract %[[CUR_T]][] : tensor<f32>
// CHECK-NEXT:    %[[UPD:.*]] = tensor.extract %[[UPD_T]][] : tensor<f32>
// CHECK-NEXT:    arith.addf %[[CUR]], %[[UPD]] : f32
// CHECK-NEXT:    tensor.from_elements
// CHECK-NEXT:    tensor.extract

// -----

// CHECK-LABEL: func @sort
// CHECK-SAME:  (%[[IN0:.*]]: tensor<16x16xf32>, %[[IN1:.*]]: tensor<16x16xi32>)
func.func @sort(%input0: tensor<16x16xf32>, %input1: tensor<16x16xi32>) {
  %0:2 = "mhlo.sort"(%input0, %input1) ({
  ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>,
       %arg2: tensor<i32>, %arg3: tensor<i32>):
    %7 = "mhlo.compare"(%arg0, %arg1)
      {comparison_direction = #mhlo<comparison_direction GT>}
        : (tensor<f32>, tensor<f32>) -> tensor<i1>
    "mhlo.return"(%7) : (tensor<i1>) -> ()
  }) {dimension = -1 : i64, is_stable = true}
     : (tensor<16x16xf32>, tensor<16x16xi32>)
    -> (tensor<16x16xf32>, tensor<16x16xi32>)
  func.return
}
// CHECK-DAG:   %[[INIT0:.*]] = tensor.empty() : tensor<16x16xf32>
// CHECK-DAG:   %[[INIT1:.*]] = tensor.empty() : tensor<16x16xi32>
// CHECK:       thlo.sort
// CHECK-NEXT:  ins(%[[IN0]] : tensor<16x16xf32>, %[[IN1]] : tensor<16x16xi32>)
// CHECK-NEXT:  outs(%[[INIT0]] : tensor<16x16xf32>, %[[INIT1]] : tensor<16x16xi32>)
// CHECK-NEXT:  dimension = 1
// CHECK-NEXT:  is_stable = true
// CHECK:       (%[[FLOAT0:.*]]: f32, %[[FLOAT1:.*]]: f32, %[[INT0:.*]]: i32, %[[INT1:.*]]: i32)
// CHECK-DAG:     %[[TENSOR0:.*]] = tensor.from_elements %[[FLOAT0]] : tensor<f32>
// CHECK-DAG:     %[[TENSOR1:.*]] = tensor.from_elements %[[FLOAT1]] : tensor<f32>
// CHECK-DAG:     %[[EXTRACTED0:.*]] = tensor.extract %[[TENSOR0]][] : tensor<f32>
// CHECK-DAG:     %[[EXTRACTED1:.*]] = tensor.extract %[[TENSOR1]][] : tensor<f32>
// CHECK:         %[[CMPRESULT:.*]] = arith.cmpf ogt, %[[EXTRACTED0]], %[[EXTRACTED1]] : f32
// CHECK-NEXT:    %[[RESULT:.*]] = tensor.from_elements %[[CMPRESULT]] : tensor<i1>
// CHECK-NEXT:    %[[EXTRACTED_RESULT:.*]] = tensor.extract %[[RESULT]][] : tensor<i1>
// CHECK-NEXT:    thlo.yield %[[EXTRACTED_RESULT]] : i1
