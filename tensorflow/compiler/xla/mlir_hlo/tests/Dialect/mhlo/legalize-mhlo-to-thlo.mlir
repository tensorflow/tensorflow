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
  // CHECK-SAME: ins(%[[ARG]] : tensor<?x?xf32>)
  // CHECK-SAME: outs(%[[INIT]] : tensor<?x?x?xf32>)
  // CHECK-SAME: broadcast_dimensions = [0, 2]
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
  // CHECK-SAME: ins(%[[ARG]] : tensor<?x?x?xf32>)
  // CHECK-SAME: outs(%[[INIT]] : tensor<?x?x?x?xf32>)
  // CHECK-SAME: broadcast_dimensions = [0, 2, 3]
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
  // CHECK-SAME:     ins(%[[A]] : tensor<?x?xi32>, %[[B]] : tensor<?x?xi32>, %[[C]] : tensor<?x?xi32>)
  // CHECK-SAME:     outs(%[[INIT]] : tensor<?x?xi32>)
  // CHECK-SAME:     {dimension = 1 : i64}
  // CHECK:      return %[[CONCATENATE]]
  %concat = "mhlo.concatenate"(%a, %b, %c) { dimension = 1 } : (tensor<?x?xi32>, tensor<?x?xi32>, tensor<?x?xi32>) -> tensor<?x?xi32>
  func.return %concat : tensor<?x?xi32>
}

// CHECK-LABEL: @concatenate_with_static_info
// CHECK-SAME:  %[[A:.*]]: tensor<?x32xi32>, %[[B:.*]]: tensor<64x16xi32>, %[[C:.*]]: tensor<?x?xi32>
func.func @concatenate_with_static_info(%a: tensor<?x32xi32>, %b: tensor<64x16xi32>, %c: tensor<?x?xi32>) -> tensor<64x?xi32> {
  // CHECK-DAG:  %[[C1:.*]] = arith.constant 1
  // CHECK-DAG:  %[[C48:.*]] = arith.constant 48
  // CHECK-DAG:  %[[CONCAT_DIM_C:.*]] = tensor.dim %[[C]], %[[C1]]
  // CHECK-DAG:  %[[CONCAT_DIM_SUM:.*]] = arith.addi %[[CONCAT_DIM_C]], %[[C48]]
  // CHECK-DAG:  %[[INIT:.*]] = tensor.empty(%[[CONCAT_DIM_SUM]])
  // CHECK:      %[[CONCAT:.*]] = thlo.concatenate
  // CHECK-SAME:     ins(%[[A]] : tensor<?x32xi32>, %[[B]] : tensor<64x16xi32>, %[[C]] : tensor<?x?xi32>)
  // CHECK-SAME:     outs(%[[INIT]] : tensor<64x?xi32>)
  // CHECK-SAME:     {dimension = 1 : i64}
  // CHECK:      return %[[CONCAT]]
  %concat = "mhlo.concatenate"(%a, %b, %c) { dimension = 1 } : (tensor<?x32xi32>, tensor<64x16xi32>, tensor<?x?xi32>) -> tensor<64x?xi32>
  func.return %concat : tensor<64x?xi32>
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
//       CHECK: %[[INIT:.*]] = tensor.empty() : tensor<3xf32>
//       CHECK: %[[GATHER:.*]] = thlo.gather
//  CHECK-SAME:   ins(%arg0 : tensor<3x3xf32>, %arg1 : tensor<3x2xi64>)
//  CHECK-SAME:   outs(%[[INIT]] : tensor<3xf32>)
//       CHECK: return %[[GATHER]]

func.func @simple_gather_unsigned(
    %operand : tensor<3x3xui32>, %indices: tensor<3x2xi64>) -> tensor<3xui32> {
  %0 = "mhlo.gather"(%operand, %indices) {
    dimension_numbers = #mhlo.gather<
      collapsed_slice_dims = [0, 1],
      index_vector_dim = 1,
      offset_dims = [],
      start_index_map = [0, 1]
    >,
    indices_are_sorted = false,
    slice_sizes = dense<[1, 1]> : tensor<2xi64>
  } : (tensor<3x3xui32>, tensor<3x2xi64>) -> tensor<3xui32>
  func.return %0 : tensor<3xui32>
}
// CHECK-LABEL: @simple_gather_unsigned
//       CHECK: %[[CAST:.*]] = builtin.unrealized_conversion_cast %arg0 : tensor<3x3xui32> to tensor<3x3xi32>
//       CHECK: %[[INIT:.*]] = tensor.empty() : tensor<3xi32>
//       CHECK: %[[GATHER:.*]] = thlo.gather
//  CHECK-SAME:   ins(%[[CAST]] : tensor<3x3xi32>, %arg1 : tensor<3x2xi64>)
//  CHECK-SAME:   outs(%[[INIT]] : tensor<3xi32>)
//       CHECK: %[[CAST2:.*]] = builtin.unrealized_conversion_cast %[[GATHER]] : tensor<3xi32> to tensor<3xui32>
//       CHECK: return %[[CAST2]]

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
//       CHECK: thlo.scatter ins(%[[INDICES]] : tensor<2x2xi32>,
//  CHECK-SAME:                    %[[UPDATE]] : tensor<2x1x3xf32>)
//  CHECK-SAME:                outs(%[[DST]] : tensor<3x3xf32>)
//  CHECK-SAME:                (%[[UPD:.*]]: f32, %[[CUR:.*]]: f32) {
//  CHECK-NEXT:    %[[CUR_T:.*]] = tensor.from_elements %[[CUR]] : tensor<f32>
//  CHECK-NEXT:    %[[UPD_T:.*]] = tensor.from_elements %[[UPD]] : tensor<f32>
//  CHECK-NEXT:    %[[CUR:.*]] = tensor.extract %[[CUR_T]][] : tensor<f32>
//  CHECK-NEXT:    %[[UPD:.*]] = tensor.extract %[[UPD_T]][] : tensor<f32>
//  CHECK-NEXT:    arith.addf %[[CUR]], %[[UPD]] : f32
//  CHECK-NEXT:    tensor.from_elements
//  CHECK-NEXT:    tensor.extract


// CHECK-LABEL: @reduce_add(
func.func @reduce_add(
    %arg0: tensor<5x4xf32>, %arg1: tensor<f32>) -> tensor<5xf32> {
  %0 = "mhlo.reduce"(%arg0, %arg1) ({
  ^bb0(%init: tensor<f32>, %arg3: tensor<f32>):
    %1 = mhlo.add %init, %arg3 : tensor<f32>
    "mhlo.return"(%1) : (tensor<f32>) -> ()
  }) {dimensions = dense<1> : tensor<1xi64>, someattr} :
    (tensor<5x4xf32>, tensor<f32>) -> tensor<5xf32>
  func.return %0 : tensor<5xf32>
}
// CHECK: %[[EXTRACT:.*]] = tensor.extract %arg1[] : tensor<f32>
// CHECK: %[[INIT:.*]] = tensor.empty() : tensor<5xf32>
// CHECK: %[[FILL:.*]] = linalg.fill ins(%[[EXTRACT]] : f32) outs(%[[INIT]] : tensor<5xf32>) -> tensor<5xf32>
// CHECK: %[[REDUCTION:.*]] = thlo.reduction ins(%arg0 : tensor<5x4xf32>) outs(%[[FILL]] : tensor<5xf32>)
// CHECK-SAME: dimensions = [1] (%[[ARG2:.*]]: f32, %[[ARG3:.*]]: f32) {
// CHECK: %[[ARG3_TENSOR:.*]] = tensor.from_elements %[[ARG3]] : tensor<f32>
// CHECK: %[[ARG2_TENSOR:.*]] = tensor.from_elements %[[ARG2]] : tensor<f32>
// CHECK: %[[ARG3_VAL:.*]] = tensor.extract %[[ARG3_TENSOR]][] : tensor<f32>
// CHECK: %[[ARG2_VAL:.*]] = tensor.extract %[[ARG2_TENSOR]][] : tensor<f32>
// CHECK: %[[ADD_VAL:.*]] = arith.addf %[[ARG3_VAL]], %[[ARG2_VAL]] : f32
// CHECK: %[[ADD_TENSOR:.*]] = tensor.from_elements %[[ADD_VAL]] : tensor<f32>
// CHECK: %[[RES:.*]] = tensor.extract %[[ADD_TENSOR]][] : tensor<f32>
// CHECK: thlo.yield %[[RES]] : f32
// CHECK: return %[[REDUCTION]] : tensor<5xf32>

// CHECK-LABEL: @variadic_reduce_add(
func.func @variadic_reduce_add(
    %arg0: tensor<5x4xf32>, %arg1: tensor<?x?xi32>,
    %arg2: tensor<f32>, %arg3: tensor<i32>) -> (tensor<5xf32>, tensor<?xi32>) {
  %reduce:2 = "mhlo.reduce"(%arg0, %arg1, %arg2, %arg3) ({
  ^bb0(%init1: tensor<f32>, %init2: tensor<i32>,
       %arg4: tensor<f32>, %arg5: tensor<i32>):
    %2 = mhlo.add %init1, %arg4 : tensor<f32>
    %3 = mhlo.add %init2, %arg5 : tensor<i32>
    "mhlo.return"(%2, %3) : (tensor<f32>, tensor<i32>) -> ()
  }) {dimensions = dense<1> : tensor<1xi64>, someattr} :
    (tensor<5x4xf32>, tensor<?x?xi32>, tensor<f32>, tensor<i32>)
    -> (tensor<5xf32>, tensor<?xi32>)
  func.return %reduce#0, %reduce#1 : tensor<5xf32>, tensor<?xi32>
}
// CHECK: %[[FILL_F32:.*]] = linalg.fill ins(%{{.*}} : f32) outs(%{{.*}} : tensor<5xf32>) -> tensor<5xf32>
// CHECK: %[[C0:.*]] = arith.constant 0 : index
// CHECK: %[[DIM_0:.*]] = tensor.dim %{{.*}}, %[[C0]] : tensor<?x?xi32>
// CHECK: %[[INIT:.*]] = tensor.empty(%[[DIM_0]]) : tensor<?xi32>
// CHECK: %[[FILL_I32:.*]] = linalg.fill ins(%{{.*}} : i32) outs(%[[INIT]] : tensor<?xi32>) -> tensor<?xi32>
// CHECK: %[[REDUCTION:.*]]:2 = thlo.reduction ins(%arg0 : tensor<5x4xf32>, %arg1 : tensor<?x?xi32>) outs(%[[FILL_F32]] : tensor<5xf32>, %[[FILL_I32]] : tensor<?xi32>)
// CHECK: thlo.yield %{{.*}}, %{{.*}} : f32, i32
// CHECK: return %[[REDUCTION]]#0, %[[REDUCTION]]#1 : tensor<5xf32>, tensor<?xi32>

// -----

// CHECK-LABEL: func @float_add
func.func @float_add(%lhs: tensor<2x2xf32>,
                %rhs: tensor<2x2xf32>) -> tensor<2x2xf32> {
  // CHECK:  %[[INIT:.*]] = tensor.empty
  // CHECK: thlo.map
  // CHECK-SAME: ins(%[[ARG0:[a-zA-Z0-9]*]] : tensor<2x2xf32>
  // CHECK-SAME: %[[ARG1:[a-zA-Z0-9]*]] : tensor<2x2xf32>)
  // CHECK-SAME: outs(%[[INIT]] : tensor<2x2xf32>)
  // CHECK-SAME: (%[[ARG2:.*]]: f32, %[[ARG3:.*]]: f32)
  // CHECK: %[[RESULT:[a-zA-Z0-9_0*]]] = arith.addf %[[ARG2]], %[[ARG3]]
  // CHECK: thlo.yield %[[RESULT]]
  %0 = "mhlo.add"(%lhs, %rhs) {someattr}
      : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  func.return %0 : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: integer_add
func.func @integer_add(%lhs: tensor<2x2xi32>,
                  %rhs: tensor<2x2xi32>) -> tensor<2x2xi32> {
  // CHECK: thlo.map
  // CHECK: addi
  %0 = "mhlo.add"(%lhs, %rhs) : (tensor<2x2xi32>,
                                    tensor<2x2xi32>) -> tensor<2x2xi32>
  func.return %0 : tensor<2x2xi32>
}

// -----

// CHECK-LABEL: complex_add
func.func @complex_add(%lhs: tensor<2x2xcomplex<f32>>,
                  %rhs: tensor<2x2xcomplex<f32>>) -> tensor<2x2xcomplex<f32>> {
  // CHECK: thlo.map
  // CHECK: complex.add
  %0 = "mhlo.add"(%lhs, %rhs) : (tensor<2x2xcomplex<f32>>,
      tensor<2x2xcomplex<f32>>) -> tensor<2x2xcomplex<f32>>
  func.return %0 : tensor<2x2xcomplex<f32>>
}

// -----

// CHECK-LABEL: func @complex_atan2
func.func @complex_atan2(%lhs: tensor<2x2xcomplex<f32>>,
    %rhs: tensor<2x2xcomplex<f32>>) -> tensor<2x2xcomplex<f32>> {
  %tensor_result = "mhlo.atan2"(%lhs, %rhs)
      : (tensor<2x2xcomplex<f32>>, tensor<2x2xcomplex<f32>>)
      -> tensor<2x2xcomplex<f32>>
  // CHECK: thlo.map
  // CHECK: complex.atan2
  func.return %tensor_result : tensor<2x2xcomplex<f32>>
}


// -----

// CHECK-LABEL: func @float_mul
func.func @float_mul(%lhs: tensor<2x2xf32>,
                %rhs: tensor<2x2xf32>) -> tensor<2x2xf32> {
  // CHECK: thlo.map
  // CHECK: mulf
  %0 = "mhlo.multiply"(%lhs, %rhs) : (tensor<2x2xf32>,
                                    tensor<2x2xf32>) -> tensor<2x2xf32>
  func.return %0 : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @integer_mul
func.func @integer_mul(%lhs: tensor<2x2xi32>,
                  %rhs: tensor<2x2xi32>) -> tensor<2x2xi32> {
  // CHECK: thlo.map
  // CHECK: muli
  %0 = "mhlo.multiply"(%lhs, %rhs) : (tensor<2x2xi32>,
                                    tensor<2x2xi32>) -> tensor<2x2xi32>
  func.return %0 : tensor<2x2xi32>
}

// -----

// CHECK-LABEL: func @complex_mul
func.func @complex_mul(%lhs: tensor<2x2xcomplex<f32>>,
                  %rhs: tensor<2x2xcomplex<f32>>) -> tensor<2x2xcomplex<f32>> {
  // CHECK: thlo.map
  // CHECK: complex.mul
  %0 = "mhlo.multiply"(%lhs, %rhs)
          : (tensor<2x2xcomplex<f32>>, tensor<2x2xcomplex<f32>>)
          -> tensor<2x2xcomplex<f32>>
  func.return %0 : tensor<2x2xcomplex<f32>>
}

// -----

// CHECK-LABEL: func @float_remainder
func.func @float_remainder(%lhs: tensor<2x2xf32>,
                      %rhs: tensor<2x2xf32>) -> tensor<2x2xf32> {
  // CHECK: thlo.map
  // CHECK: remf
  %0 = "mhlo.remainder"(%lhs, %rhs) : (tensor<2x2xf32>,
                                    tensor<2x2xf32>) -> tensor<2x2xf32>
  func.return %0 : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @integer_remainder
func.func @integer_remainder(%lhs: tensor<2x2xi32>,
                        %rhs: tensor<2x2xi32>) -> tensor<2x2xi32> {
  // CHECK: thlo.map
  // CHECK: arith.remsi
  %0 = "mhlo.remainder"(%lhs, %rhs) : (tensor<2x2xi32>,
                                          tensor<2x2xi32>) -> tensor<2x2xi32>
  func.return %0 : tensor<2x2xi32>
}

// -----

// CHECK-LABEL: func @population_count_integer
func.func @population_count_integer(%lhs: tensor<2x2xi32>) -> tensor<2x2xi32> {
  // CHECK: thlo.map
  // CHECK: math.ctpop
  %0 = "mhlo.popcnt"(%lhs) : (tensor<2x2xi32>) -> tensor<2x2xi32>
  func.return %0 : tensor<2x2xi32>
}

// -----

// CHECK-LABEL: func @complex_sqrt
func.func @complex_sqrt(%operand: tensor<2x2xcomplex<f32>>) -> tensor<2x2xcomplex<f32>> {
  %tensor_result = "mhlo.sqrt"(%operand)
      : (tensor<2x2xcomplex<f32>>) -> tensor<2x2xcomplex<f32>>
  // CHECK: thlo.map
  // CHECK: complex.sqrt
  func.return %tensor_result : tensor<2x2xcomplex<f32>>
}

// -----

// CHECK-LABEL: func @float_rsqrt
func.func @float_rsqrt(%operand: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %tensor_result = "mhlo.rsqrt"(%operand)
      : (tensor<2x2xf32>) -> tensor<2x2xf32>
  // CHECK: thlo.map
  // CHECK: rsqrt
  func.return %tensor_result : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @complex_rsqrt
func.func @complex_rsqrt(%operand: tensor<2x2xcomplex<f32>>)
    -> tensor<2x2xcomplex<f32>> {
  %tensor_result = "mhlo.rsqrt"(%operand)
      : (tensor<2x2xcomplex<f32>>) -> tensor<2x2xcomplex<f32>>
  // CHECK: thlo.map
  // CHECK: complex.rsqrt
  func.return %tensor_result : tensor<2x2xcomplex<f32>>
}

// -----

// CHECK-LABEL: func @float_cbrt
func.func @float_cbrt(%operand: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %tensor_result = "mhlo.cbrt"(%operand)
      : (tensor<2x2xf32>) -> tensor<2x2xf32>
  // CHECK-DAG: %[[THIRD:.+]] = arith.constant 0.333333343
  // CHECK-DAG: %[[ABS:.+]] = math.absf %arg1
  // CHECK-DAG: %[[POW:.+]] = math.powf %[[ABS]], %[[THIRD]]
  // CHECK-DAG: %[[RESULT:.+]] = math.copysign %[[POW]], %arg1
  // CHECK: thlo.yield %[[RESULT]]
  func.return %tensor_result : tensor<2x2xf32>
}

// -----


// CHECK-LABEL: func @float_sub
func.func @float_sub(%lhs: tensor<2x2xf32>,
                %rhs: tensor<2x2xf32>) -> tensor<2x2xf32> {
  // CHECK: thlo.map
  // CHECK: subf
  %0 = "mhlo.subtract"(%lhs, %rhs) : (tensor<2x2xf32>,
                                    tensor<2x2xf32>) -> tensor<2x2xf32>
  func.return %0 : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @integer_sub
func.func @integer_sub(%lhs: tensor<2x2xi32>,
                  %rhs: tensor<2x2xi32>) -> tensor<2x2xi32> {
  // CHECK: thlo.map
  // CHECK: subi
  %0 = "mhlo.subtract"(%lhs, %rhs) : (tensor<2x2xi32>,
                                    tensor<2x2xi32>) -> tensor<2x2xi32>
  func.return %0 : tensor<2x2xi32>
}

// -----

// CHECK-LABEL: complex_sub
func.func @complex_sub(%lhs: tensor<2x2xcomplex<f32>>,
                  %rhs: tensor<2x2xcomplex<f32>>) -> tensor<2x2xcomplex<f32>> {
  // CHECK: thlo.map
  // CHECK: complex.sub
  %0 = "mhlo.subtract"(%lhs, %rhs) : (tensor<2x2xcomplex<f32>>,
      tensor<2x2xcomplex<f32>>) -> tensor<2x2xcomplex<f32>>
  func.return %0 : tensor<2x2xcomplex<f32>>
}

// -----

// CHECK-LABEL: func @float_abs
func.func @float_abs(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
  // CHECK: thlo.map
  // CHECK: math.absf
  %0 = "mhlo.abs"(%arg0) {someattr} : (tensor<2x2xf32>) -> tensor<2x2xf32>
  func.return %0 : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @float_exp
func.func @float_exp(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
  // CHECK: thlo.map
  // CHECK: exp
  %0 = "mhlo.exponential"(%arg0) : (tensor<2x2xf32>) -> tensor<2x2xf32>
  func.return %0 : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @complex_exp
func.func @complex_exp(%arg0: tensor<2x2xcomplex<f32>>) -> tensor<2x2xcomplex<f32>> {
  // CHECK: thlo.map
  // CHECK: complex.exp
  %0 = "mhlo.exponential"(%arg0) : (tensor<2x2xcomplex<f32>>)
                                 -> tensor<2x2xcomplex<f32>>
  func.return %0 : tensor<2x2xcomplex<f32>>
}

// -----

// CHECK-LABEL: func @float_expm1
func.func @float_expm1(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
  // CHECK: thlo.map
  // CHECK: expm1
  %0 = "mhlo.exponential_minus_one"(%arg0) : (tensor<2x2xf32>) -> tensor<2x2xf32>
  func.return %0 : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @complex_expm1
func.func @complex_expm1(%arg0: tensor<2x2xcomplex<f32>>)
    -> tensor<2x2xcomplex<f32>> {
  // CHECK: thlo.map
  // CHECK: complex.expm1
  %0 = "mhlo.exponential_minus_one"(%arg0)
    : (tensor<2x2xcomplex<f32>>) -> tensor<2x2xcomplex<f32>>
  func.return %0 : tensor<2x2xcomplex<f32>>
}

// -----

// CHECK-LABEL: func @float_log
func.func @float_log(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
  // CHECK: thlo.map
  // CHECK: math.log
  %0 = "mhlo.log"(%arg0) : (tensor<2x2xf32>) -> tensor<2x2xf32>
  func.return %0 : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @complex_log
func.func @complex_log(%arg0: tensor<2x2xcomplex<f32>>) -> tensor<2x2xcomplex<f32>> {
  // CHECK: thlo.map
  // CHECK: complex.log
  %0 = "mhlo.log"(%arg0) : (tensor<2x2xcomplex<f32>>)
                         -> tensor<2x2xcomplex<f32>>
  func.return %0 : tensor<2x2xcomplex<f32>>
}

// -----

// CHECK-LABEL: func @float_log1p
func.func @float_log1p(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
  // CHECK: thlo.map
  // CHECK: math.log1p
  %0 = "mhlo.log_plus_one"(%arg0) : (tensor<2x2xf32>) -> tensor<2x2xf32>
  func.return %0 : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @complex_log1p
func.func @complex_log1p(%arg0: tensor<2x2xcomplex<f32>>)
    -> tensor<2x2xcomplex<f32>> {
  // CHECK: thlo.map
  // CHECK: complex.log1p
  %0 = "mhlo.log_plus_one"(%arg0) : (tensor<2x2xcomplex<f32>>)
                                  -> tensor<2x2xcomplex<f32>>
  func.return %0 : tensor<2x2xcomplex<f32>>
}

// -----

// CHECK-LABEL: func @float_logistic
func.func @float_logistic(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
  // CHECK: thlo.map
  // CHECK: (%[[ARG:[0-9a-z]*]]: f32) {
  // CHECK: %[[NEG_ARG:.*]] = arith.negf %[[ARG]]
  // CHECK: %[[EXP_NEG_ARG:.*]] = math.exp %[[NEG_ARG]]
  // CHECK: %[[C1:.*]] = arith.constant 1.{{.*}}e+00
  // CHECK: %[[ONE_ADD_EXP_NEG_ARG:.*]] = arith.addf %[[EXP_NEG_ARG]], %[[C1]]
  // CHECK: %[[RESULT:.*]] = arith.divf %[[C1]], %[[ONE_ADD_EXP_NEG_ARG]]
  // CHECK: thlo.yield %[[RESULT]]
  %0 = "mhlo.logistic"(%arg0) : (tensor<2x2xf32>) -> tensor<2x2xf32>
  func.return %0 : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @float_ceil
func.func @float_ceil(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
  // CHECK: thlo.map
  // CHECK: math.ceil
  %0 = "mhlo.ceil"(%arg0) : (tensor<2x2xf32>) -> tensor<2x2xf32>
  func.return %0 : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @floor
func.func @floor(%input: tensor<2x2xf32>) -> tensor<2x2xf32> {
  // CHECK: thlo.map
  // CHECK: math.floor
  %0 = "mhlo.floor"(%input) : (tensor<2x2xf32>) -> tensor<2x2xf32>
  func.return %0 : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @float_neg
func.func @float_neg(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
  // CHECK: thlo.map
  // CHECK: negf
  %0 = "mhlo.negate"(%arg0) : (tensor<2x2xf32>) -> tensor<2x2xf32>
  func.return %0 : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @complex_neg
func.func @complex_neg(%arg0: tensor<2x2xcomplex<f32>>) -> tensor<2x2xcomplex<f32>> {
  // CHECK: thlo.map
  // CHECK: complex.neg
  %0 = "mhlo.negate"(%arg0) : (tensor<2x2xcomplex<f32>>)
                            -> tensor<2x2xcomplex<f32>>
  func.return %0 : tensor<2x2xcomplex<f32>>
}

// -----

// CHECK-LABEL: func @complex_sign
func.func @complex_sign(
    %arg0: tensor<2x2xcomplex<f32>>) -> tensor<2x2xcomplex<f32>> {
  // CHECK: thlo.map
  // CHECK: complex.sign
  %0 = "mhlo.sign"(%arg0) : (tensor<2x2xcomplex<f32>>)
                          -> tensor<2x2xcomplex<f32>>
  func.return %0 : tensor<2x2xcomplex<f32>>
}

// -----

// CHECK-LABEL: func @float_tanh
func.func @float_tanh(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
  // CHECK: thlo.map
  // CHECK: tanh
  %0 = "mhlo.tanh"(%arg0) : (tensor<2x2xf32>) -> tensor<2x2xf32>
  func.return %0 : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @complex_tanh
func.func @complex_tanh(%operand: tensor<2x2xcomplex<f32>>) -> tensor<2x2xcomplex<f32>> {
  %tensor_result = "mhlo.tanh"(%operand)
      : (tensor<2x2xcomplex<f32>>) -> tensor<2x2xcomplex<f32>>
  // CHECK: thlo.map
  // CHECK: complex.tanh
  func.return %tensor_result : tensor<2x2xcomplex<f32>>
}

// -----

// CHECK-LABEL: func @integer_and
func.func @integer_and(%lhs: tensor<2x2xi32>,
                  %rhs: tensor<2x2xi32>) -> tensor<2x2xi32> {
  // CHECK: thlo.map
  // CHECK: and
  %0 = "mhlo.and"(%lhs, %rhs) : (tensor<2x2xi32>,
                                    tensor<2x2xi32>) -> tensor<2x2xi32>
  func.return %0 : tensor<2x2xi32>
}

// -----

// CHECK-LABEL: func @integer_or
func.func @integer_or(%lhs: tensor<2x2xi32>,
                  %rhs: tensor<2x2xi32>) -> tensor<2x2xi32> {
  // CHECK: thlo.map
  // CHECK: or
  %0 = "mhlo.or"(%lhs, %rhs) : (tensor<2x2xi32>,
                                    tensor<2x2xi32>) -> tensor<2x2xi32>
  func.return %0 : tensor<2x2xi32>
}

// -----

// CHECK-LABEL: func @integer_xor
func.func @integer_xor(%lhs: tensor<2x2xi32>,
                  %rhs: tensor<2x2xi32>) -> tensor<2x2xi32> {
  // CHECK: thlo.map
  // CHECK: xor
  %0 = "mhlo.xor"(%lhs, %rhs) : (tensor<2x2xi32>,
                                    tensor<2x2xi32>) -> tensor<2x2xi32>
  func.return %0 : tensor<2x2xi32>
}

// -----

// CHECK-LABEL: func @count_leading_zeros
func.func @count_leading_zeros(%lhs: tensor<2x2xi32>) -> tensor<2x2xi32> {
  // CHECK: thlo.map
  // CHECK: math.ctlz
  %0 = "mhlo.count_leading_zeros"(%lhs) : (tensor<2x2xi32>) -> tensor<2x2xi32>
  func.return %0 : tensor<2x2xi32>
}

// -----

// CHECK-LABEL: func @float_cmp
func.func @float_cmp(%lhs: tensor<2x2xf32>,
                %rhs: tensor<2x2xf32>) -> (tensor<2x2xi1>) {
  %0 = "mhlo.compare"(%lhs, %rhs) {comparison_direction = #mhlo<comparison_direction EQ>}
          : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xi1>
  func.return %0 : tensor<2x2xi1>
}
// CHECK: tensor.empty() : tensor<2x2xi1>
// CHECK: thlo.map
// CHECK-SAME: ins({{[^)]*}}) outs(%{{[^)]*}})
// CHECK-SAME: (%[[LHS_IN:.*]]: f32, %[[RHS_IN:.*]]: f32) {
// CHECK-NEXT:   %[[RESULT:.*]] = arith.cmpf oeq, %[[LHS_IN]], %[[RHS_IN]] : f32
// CHECK-NEXT:   thlo.yield %[[RESULT]] : i1

// -----

// CHECK-LABEL: func @float_cmp_ne
func.func @float_cmp_ne(%lhs: tensor<2x2xf32>,
                %rhs: tensor<2x2xf32>) -> (tensor<2x2xi1>) {
  %0 = "mhlo.compare"(%lhs, %rhs) {comparison_direction = #mhlo<comparison_direction NE>}
          : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xi1>
  func.return %0 : tensor<2x2xi1>
}
// CHECK: tensor.empty() : tensor<2x2xi1>
// CHECK: thlo.map
// CHECK-SAME: ins({{[^)]*}}) outs(%{{[^)]*}})
// CHECK-SAME: (%[[LHS_IN:.*]]: f32, %[[RHS_IN:.*]]: f32) {
// CHECK-NEXT:   %[[RESULT:.*]] = arith.cmpf une, %[[LHS_IN]], %[[RHS_IN]] : f32
// CHECK-NEXT:   thlo.yield %[[RESULT]] : i1

// -----

// CHECK-LABEL: func @int_cmp
func.func @int_cmp(%lhs: tensor<2x2xi32>,
              %rhs: tensor<2x2xi32>) -> tensor<2x2xi1> {
  %0 = "mhlo.compare"(%lhs, %rhs) {comparison_direction = #mhlo<comparison_direction LT>}
          : (tensor<2x2xi32>, tensor<2x2xi32>) -> (tensor<2x2xi1>)
  func.return %0 : tensor<2x2xi1>
}
// CHECK: tensor.empty() : tensor<2x2xi1>
// CHECK: thlo.map
// CHECK-SAME: ins({{[^)]*}}) outs(%{{[^)]*}})
// CHECK-SAME: (%[[LHS_IN:.*]]: i32, %[[RHS_IN:.*]]: i32) {
// CHECK-NEXT:   %[[RESULT:.*]] = arith.cmpi slt, %[[LHS_IN]], %[[RHS_IN]] : i32
// CHECK-NEXT:   thlo.yield %[[RESULT]] : i1

// -----

// CHECK-LABEL: func @complex_cmp_eq
func.func @complex_cmp_eq(%lhs: tensor<2xcomplex<f32>>,
                     %rhs: tensor<2xcomplex<f32>>) -> tensor<2xi1> {
  %0 = "mhlo.compare"(%lhs, %rhs) {comparison_direction = #mhlo<comparison_direction EQ>}
          : (tensor<2xcomplex<f32>>, tensor<2xcomplex<f32>>) -> (tensor<2xi1>)
  func.return %0 : tensor<2xi1>
}
// CHECK: tensor.empty() : tensor<2xi1>
// CHECK: thlo.map
// CHECK-SAME: ins({{[^)]*}}) outs(%{{[^)]*}})
// CHECK-SAME: (%[[LHS_IN:.*]]: complex<f32>, %[[RHS_IN:.*]]: complex<f32>) {
// CHECK-NEXT:   %[[RESULT:.*]] = complex.eq %[[LHS_IN]], %[[RHS_IN]] : complex<f32>
// CHECK-NEXT:   thlo.yield %[[RESULT]] : i1

// -----

// CHECK-LABEL: func @complex_cmp_neq
func.func @complex_cmp_neq(%lhs: tensor<2xcomplex<f64>>,
                      %rhs: tensor<2xcomplex<f64>>) -> tensor<2xi1> {
  %0 = "mhlo.compare"(%lhs, %rhs) {comparison_direction = #mhlo<comparison_direction NE>}
          : (tensor<2xcomplex<f64>>, tensor<2xcomplex<f64>>) -> (tensor<2xi1>)
  func.return %0 : tensor<2xi1>
}
// CHECK: tensor.empty() : tensor<2xi1>
// CHECK: thlo.map
// CHECK-SAME: ins({{[^)]*}}) outs(%{{[^)]*}})
// CHECK-SAME: (%[[LHS_IN:.*]]: complex<f64>, %[[RHS_IN:.*]]: complex<f64>) {
// CHECK-NEXT:   %[[RESULT:.*]] = complex.neq %[[LHS_IN]], %[[RHS_IN]] : complex<f64>
// CHECK-NEXT:   thlo.yield %[[RESULT]] : i1

// -----

// CHECK-LABEL: func @float_cos
func.func @float_cos(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
  // CHECK: thlo.map
  // CHECK: math.cos
  %0 = "mhlo.cosine"(%arg0) : (tensor<2x2xf32>) -> tensor<2x2xf32>
  func.return %0 : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @complex_cos
func.func @complex_cos(%arg0: tensor<2x2xcomplex<f32>>) -> tensor<2x2xcomplex<f32>> {
  // CHECK: thlo.map
  // CHECK: complex.cos
  %0 = "mhlo.cosine"(%arg0) : (tensor<2x2xcomplex<f32>>) -> tensor<2x2xcomplex<f32>>
  func.return %0 : tensor<2x2xcomplex<f32>>
}

// -----

// CHECK-LABEL: func @float_sin
func.func @float_sin(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
  // CHECK: thlo.map
  // CHECK: math.sin
  %0 = "mhlo.sine"(%arg0) : (tensor<2x2xf32>) -> tensor<2x2xf32>
  func.return %0 : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @complex_sin
func.func @complex_sin(%arg0: tensor<2x2xcomplex<f32>>) -> tensor<2x2xcomplex<f32>> {
  // CHECK: thlo.map
  // CHECK: complex.sin
  %0 = "mhlo.sine"(%arg0) : (tensor<2x2xcomplex<f32>>) -> tensor<2x2xcomplex<f32>>
  func.return %0 : tensor<2x2xcomplex<f32>>
}

// -----

// CHECK-LABEL: func @copy
// CHECK-SAME: [[ARG:%[a-zA-Z0-9]+]]
func.func @copy(%input: tensor<2x4x8xf32>) -> tensor<2x4x8xf32> {
  %0 = "mhlo.copy"(%input) : (tensor<2x4x8xf32>) -> (tensor<2x4x8xf32>)
  func.return %0 : tensor<2x4x8xf32>
}
// CHECK: return [[ARG]] : tensor<2x4x8xf32>

// -----

// CHECK-LABEL: func @is_finite
func.func @is_finite(%input: tensor<2x2xf32>) -> tensor<2x2xi1> {
  %0 = "mhlo.is_finite"(%input) : (tensor<2x2xf32>) -> tensor<2x2xi1>
  func.return %0 : tensor<2x2xi1>
}
// CHECK: thlo.map
// CHECK-SAME: ins({{[^)]*}}) outs(%{{[^)]*}})
// CHECK-SAME: (%[[OPERAND_IN:.*]]: f32) {
// CHECK-NEXT:   %[[POS_INF:.+]] = arith.constant 0x7F800000 : f32
// CHECK-NEXT:   %[[ABS_X:.+]] = math.absf %[[OPERAND_IN]] : f32
// CHECK-NEXT:   %[[RESULT:.+]] = arith.cmpf one, %[[ABS_X]], %[[POS_INF]] : f32
// CHECK-NEXT:   thlo.yield %[[RESULT]] : i1

// -----

// CHECK-LABEL: func @round_nearest_even
func.func @round_nearest_even(%val: tensor<2x2xf32>) -> tensor<2x2xf32> {
  // CHECK: %[[ROUND:.+]] = math.roundeven %arg1
  // CHECK: thlo.yield %[[ROUND]]
  %0 = "mhlo.round_nearest_even"(%val) : (tensor<2x2xf32>) -> (tensor<2x2xf32>)
  func.return %0 : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @round
func.func @round(%val: tensor<2x2xf32>) -> tensor<2x2xf32> {
  // CHECK: %[[ROUND:.+]] = math.round %arg1
  // CHECK: thlo.yield %[[ROUND]]
  %0 = "mhlo.round_nearest_afz"(%val) : (tensor<2x2xf32>) -> (tensor<2x2xf32>)
  func.return %0 : tensor<2x2xf32>
}

// -----

func.func @transpose(%arg0: tensor<1x2x3x4xi32>) -> tensor<2x1x4x3xi32> {
  %0 = "mhlo.transpose"(%arg0) {
    permutation = dense<[1, 0, 3, 2]> : tensor<4xi64>
  } : (tensor<1x2x3x4xi32>) -> tensor<2x1x4x3xi32>
  func.return %0: tensor<2x1x4x3xi32>
}

// CHECK-LABEL: @transpose
// CHECK:       %[[INIT:.*]] = tensor.empty()
// CHECK-SAME:      : tensor<2x1x4x3xi32>
// CHECK:       %[[TRANSPOSE:.*]] = thlo.transpose
// CHECK-SAME:      ins(%arg0 : tensor<1x2x3x4xi32>)
// CHECK-SAME:      outs(%[[INIT]] : tensor<2x1x4x3xi32>)
// CHECK-SAME:      permutation = [1, 0, 3, 2]
// CHECK:       return %[[TRANSPOSE]]

// -----

// CHECK-LABEL: func @select
func.func @select(%pred: tensor<2x2xi1>, %lhs: tensor<2x2xf32>,
             %rhs: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %0 = "mhlo.select"(%pred, %lhs, %rhs)
         : (tensor<2x2xi1>, tensor<2x2xf32>, tensor<2x2xf32>)
         -> (tensor<2x2xf32>)
  func.return %0 : tensor<2x2xf32>
}
// CHECK: tensor.empty() : tensor<2x2xf32>
// CHECK: thlo.map
// CHECK-SAME: (%[[PRED_IN:[a-zA-Z0-9]*]]: i1, %[[LHS_IN:.*]]: f32, %[[RHS_IN:.*]]: f32) {
// CHECK-NEXT:   %[[RESULT:.*]] = arith.select %[[PRED_IN]], %[[LHS_IN]], %[[RHS_IN]] : f32
// CHECK-NEXT:   thlo.yield %[[RESULT]] : f32

// -----

// CHECK-LABEL: func @select_scalar_pred_dyn
// CHECK-SAME:  (%[[PRED:.*]]: tensor<i1>, %[[LHS:.*]]: tensor<2x?xf32>, %[[RHS:.*]]: tensor<2x?xf32>)
func.func @select_scalar_pred_dyn(%pred : tensor<i1>, %lhs: tensor<2x?xf32>,
                                  %rhs: tensor<2x?xf32>) -> tensor<2x?xf32> {
  %0 = "mhlo.select"(%pred, %lhs, %rhs) {someattr} :
    (tensor<i1>, tensor<2x?xf32>, tensor<2x?xf32>) -> (tensor<2x?xf32>)
  func.return %0 : tensor<2x?xf32>
}
// CHECK-DAG:  %[[PRED_:.*]] = tensor.extract %[[PRED]][] : tensor<i1>
// CHECK-DAG:  %[[SHAPE:.*]] = shape.shape_of %[[LHS]]
// CHECK-DAG:  %[[C1:.*]] = arith.constant 1
// CHECK-DAG:  %[[DIM:.*]] = tensor.extract %[[SHAPE]][%[[C1]]] : tensor<2xindex>
// CHECK-DAG:  %[[DST:.*]] = tensor.empty(%[[DIM]])
// CHECK:      thlo.map
// CHECK-SAME: (%[[LHS_IN:[a-zA-Z0-9]*]]: f32, %[[RHS_IN:.*]]: f32) {
// CHECK-NEXT:   %[[RES:.*]] = arith.select %[[PRED_]], %[[LHS_IN]], %[[RHS_IN]] : f32
// CHECK-NEXT:   thlo.yield %[[RES]]

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
// CHECK-SAME:  ins(%[[IN0]] : tensor<16x16xf32>, %[[IN1]] : tensor<16x16xi32>)
// CHECK-SAME:  outs(%[[INIT0]] : tensor<16x16xf32>, %[[INIT1]] : tensor<16x16xi32>)
// CHECK-DAG:   dimension = 1 : i64
// CHECK-DAG:   is_stable = true
// CHECK:       (%[[FLOAT0:.*]]: f32, %[[FLOAT1:.*]]: f32, %[[INT0:.*]]: i32, %[[INT1:.*]]: i32)
// CHECK-DAG:     %[[TENSOR0:.*]] = tensor.from_elements %[[FLOAT0]] : tensor<f32>
// CHECK-DAG:     %[[TENSOR1:.*]] = tensor.from_elements %[[FLOAT1]] : tensor<f32>
// CHECK-DAG:     %[[EXTRACTED0:.*]] = tensor.extract %[[TENSOR0]][] : tensor<f32>
// CHECK-DAG:     %[[EXTRACTED1:.*]] = tensor.extract %[[TENSOR1]][] : tensor<f32>
// CHECK:         %[[CMPRESULT:.*]] = arith.cmpf ogt, %[[EXTRACTED0]], %[[EXTRACTED1]] : f32
// CHECK-NEXT:    %[[RESULT:.*]] = tensor.from_elements %[[CMPRESULT]] : tensor<i1>
// CHECK-NEXT:    %[[EXTRACTED_RESULT:.*]] = tensor.extract %[[RESULT]][] : tensor<i1>
// CHECK-NEXT:    thlo.yield %[[EXTRACTED_RESULT]] : i1
