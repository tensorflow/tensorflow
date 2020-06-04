// RUN: xla-opt %s -hlo-legalize-to-linalg -split-input-file | FileCheck %s

// CHECK: #map0 = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL: func @float_add
func @float_add(%lhs: tensor<2x2xf32>,
                %rhs: tensor<2x2xf32>) -> tensor<2x2xf32> {
  // CHECK: linalg.generic
  // CHECK: ^{{[a-z0-9_]*}}
  // CHECK-SAME: %[[ARG0:[a-zA-Z0-9_]*]]: f32
  // CHECK-SAME: %[[ARG1:[a-zA-Z0-9_]*]]: f32
  // CHECK: %[[RESULT:[a-zA-Z0-9_]*]] = addf %[[ARG0]], %[[ARG1]]
  // CHECK: linalg.yield %[[RESULT]]
  %0 = "xla_hlo.add"(%lhs, %rhs) : (tensor<2x2xf32>,
                                    tensor<2x2xf32>) -> tensor<2x2xf32>
  return %0 : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: integer_add
func @integer_add(%lhs: tensor<2x2xi32>,
                  %rhs: tensor<2x2xi32>) -> tensor<2x2xi32> {
  // CHECK: linalg.generic
  // CHECK: addi
  %0 = "xla_hlo.add"(%lhs, %rhs) : (tensor<2x2xi32>,
                                    tensor<2x2xi32>) -> tensor<2x2xi32>
  return %0 : tensor<2x2xi32>
}

// -----

// CHECK-LABEL: func @float_mul
func @float_mul(%lhs: tensor<2x2xf32>,
                %rhs: tensor<2x2xf32>) -> tensor<2x2xf32> {
  // CHECK: linalg.generic
  // CHECK: mulf
  %0 = "xla_hlo.multiply"(%lhs, %rhs) : (tensor<2x2xf32>,
                                    tensor<2x2xf32>) -> tensor<2x2xf32>
  return %0 : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @integer_mul
func @integer_mul(%lhs: tensor<2x2xi32>,
                  %rhs: tensor<2x2xi32>) -> tensor<2x2xi32> {
  // CHECK: linalg.generic
  // CHECK: muli
  %0 = "xla_hlo.multiply"(%lhs, %rhs) : (tensor<2x2xi32>,
                                    tensor<2x2xi32>) -> tensor<2x2xi32>
  return %0 : tensor<2x2xi32>
}

// -----

// CHECK-LABEL: func @float_remainder
func @float_remainder(%lhs: tensor<2x2xf32>,
                      %rhs: tensor<2x2xf32>) -> tensor<2x2xf32> {
  // CHECK: linalg.generic
  // CHECK: remf
  %0 = "xla_hlo.remainder"(%lhs, %rhs) : (tensor<2x2xf32>,
                                    tensor<2x2xf32>) -> tensor<2x2xf32>
  return %0 : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @integer_remainder
func @integer_remainder(%lhs: tensor<2x2xi32>,
                        %rhs: tensor<2x2xi32>) -> tensor<2x2xi32> {
  // CHECK: linalg.generic
  // CHECK: remi_signed
  %0 = "xla_hlo.remainder"(%lhs, %rhs) : (tensor<2x2xi32>,
                                          tensor<2x2xi32>) -> tensor<2x2xi32>
  return %0 : tensor<2x2xi32>
}

// -----

// CHECK-LABEL: func @float_rsqrt
func @float_rsqrt(%operand: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %tensor_result = "xla_hlo.rsqrt"(%operand)
      : (tensor<2x2xf32>) -> tensor<2x2xf32>
  // CHECK: linalg.generic
  // CHECK: rsqrt
  return %tensor_result : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @float_sub
func @float_sub(%lhs: tensor<2x2xf32>,
                %rhs: tensor<2x2xf32>) -> tensor<2x2xf32> {
  // CHECK: linalg.generic
  // CHECK: subf
  %0 = "xla_hlo.subtract"(%lhs, %rhs) : (tensor<2x2xf32>,
                                    tensor<2x2xf32>) -> tensor<2x2xf32>
  return %0 : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @integer_sub
func @integer_sub(%lhs: tensor<2x2xi32>,
                  %rhs: tensor<2x2xi32>) -> tensor<2x2xi32> {
  // CHECK: linalg.generic
  // CHECK: subi
  %0 = "xla_hlo.subtract"(%lhs, %rhs) : (tensor<2x2xi32>,
                                    tensor<2x2xi32>) -> tensor<2x2xi32>
  return %0 : tensor<2x2xi32>
}

// -----

// CHECK-LABEL: func @float_abs
func @float_abs(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
  // CHECK: linalg.generic
  // CHECK: absf
  %0 = "xla_hlo.abs"(%arg0) : (tensor<2x2xf32>) -> tensor<2x2xf32>
  return %0 : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @float_exp
func @float_exp(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
  // CHECK: linalg.generic
  // CHECK: exp
  %0 = "xla_hlo.exponential"(%arg0) : (tensor<2x2xf32>) -> tensor<2x2xf32>
  return %0 : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @float_log
func @float_log(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
  // CHECK: linalg.generic
  // CHECK: log
  %0 = "xla_hlo.log"(%arg0) : (tensor<2x2xf32>) -> tensor<2x2xf32>
  return %0 : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @float_ceil
func @float_ceil(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
  // CHECK: linalg.generic
  // CHECK: ceilf
  %0 = "xla_hlo.ceil"(%arg0) : (tensor<2x2xf32>) -> tensor<2x2xf32>
  return %0 : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @float_neg
func @float_neg(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
  // CHECK: linalg.generic
  // CHECK: negf
  %0 = "xla_hlo.negate"(%arg0) : (tensor<2x2xf32>) -> tensor<2x2xf32>
  return %0 : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @float_tanh
func @float_tanh(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
  // CHECK: linalg.generic
  // CHECK: tanh
  %0 = "xla_hlo.tanh"(%arg0) : (tensor<2x2xf32>) -> tensor<2x2xf32>
  return %0 : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @integer_and
func @integer_and(%lhs: tensor<2x2xi32>,
                  %rhs: tensor<2x2xi32>) -> tensor<2x2xi32> {
  // CHECK: linalg.generic
  // CHECK: and
  %0 = "xla_hlo.and"(%lhs, %rhs) : (tensor<2x2xi32>,
                                    tensor<2x2xi32>) -> tensor<2x2xi32>
  return %0 : tensor<2x2xi32>
}

// -----

// CHECK-LABEL: func @float_cmp
func @float_cmp(%lhs: tensor<2x2xf32>,
                %rhs: tensor<2x2xf32>) -> (tensor<2x2xi1>) {
  %0 = "xla_hlo.compare"(%lhs, %rhs) {comparison_direction = "EQ"}
          : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xi1>
  return %0 : tensor<2x2xi1>
}
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[LHS_IN:.*]]: f32, %[[RHS_IN:.*]]: f32):
// CHECK-NEXT:   %[[RESULT:.*]] = cmpf "oeq", %[[LHS_IN]], %[[RHS_IN]] : f32
// CHECK-NEXT:   linalg.yield %[[RESULT]] : i1

// -----

// CHECK-LABEL: func @int_cmp
func @int_cmp(%lhs: tensor<2x2xi32>,
              %rhs: tensor<2x2xi32>) -> tensor<2x2xi1> {
  %0 = "xla_hlo.compare"(%lhs, %rhs) {comparison_direction = "LT"}
          : (tensor<2x2xi32>, tensor<2x2xi32>) -> (tensor<2x2xi1>)
  return %0 : tensor<2x2xi1>
}
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[LHS_IN:.*]]: i32, %[[RHS_IN:.*]]: i32):
// CHECK-NEXT:   %[[RESULT:.*]] = cmpi "slt", %[[LHS_IN]], %[[RHS_IN]] : i32
// CHECK-NEXT:   linalg.yield %[[RESULT]] : i1

// -----

// CHECK-LABEL: func @float_cos
func @float_cos(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
  // CHECK: linalg.generic
  // CHECK: cos
  %0 = "xla_hlo.cosine"(%arg0) : (tensor<2x2xf32>) -> tensor<2x2xf32>
  return %0 : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @float_sin
func @float_sin(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
  // CHECK: linalg.generic
  // CHECK: sin
  %0 = "xla_hlo.sine"(%arg0) : (tensor<2x2xf32>) -> tensor<2x2xf32>
  return %0 : tensor<2x2xf32>
}

// -----

// CHECK-LABEL: func @copy
// CHECK-SAME: [[ARG:%[a-zA-Z0-9]+]]
func @copy(%input: tensor<2x4x8xf32>) -> tensor<2x4x8xf32> {
  %0 = "xla_hlo.copy"(%input) : (tensor<2x4x8xf32>) -> (tensor<2x4x8xf32>)
  return %0 : tensor<2x4x8xf32>
}
// CHECK: return [[ARG]] : tensor<2x4x8xf32>

// -----

// CHECK-LABEL: func @select
func @select(%pred: tensor<2x2xi1>, %lhs: tensor<2x2xf32>,
             %rhs: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %0 = "xla_hlo.select"(%pred, %lhs, %rhs)
         : (tensor<2x2xi1>, tensor<2x2xf32>, tensor<2x2xf32>) -> (tensor<2x2xf32>)
  return %0 : tensor<2x2xf32>
}
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[PRED_IN:.*]]: i1, %[[LHS_IN:.*]]: f32, %[[RHS_IN:.*]]: f32):
// CHECK-NEXT:   %[[RESULT:.*]] = select %[[PRED_IN]], %[[LHS_IN]], %[[RHS_IN]] : f32
// CHECK-NEXT:   linalg.yield %[[RESULT]] : f32

// -----

// CHECK-DAG: #[[OPERAND_MAP:.+]] = affine_map<(d0, d1, d2) -> ()>
// CHECK-DAG: #[[RESULT_MAP:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK-LABEL: func @broadcast_scalar
func @broadcast_scalar(%arg: tensor<f32>) -> tensor<4x2x1xf32> {
  %0 = "xla_hlo.broadcast"(%arg) {broadcast_sizes = dense<[4, 2, 1]> : tensor<3xi64>} : (tensor<f32>) -> tensor<4x2x1xf32>
  return %0: tensor<4x2x1xf32>
}
// CHECK: linalg.generic {{{.*}}indexing_maps = [#[[OPERAND_MAP]], #[[RESULT_MAP]]]
// CHECK-NEXT: ^bb0(%[[OPERAND:.*]]: f32):
// CHECK-NEXT:   linalg.yield %[[OPERAND]] : f32

// -----

// CHECK-DAG: #[[OPERAND_MAP:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d3, d4, d5)>
// CHECK-DAG: #[[RESULT_MAP:.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>
// CHECK-LABEL: func @broadcast
func @broadcast(%arg: tensor<4x?x16xf32>) -> tensor<4x2x1x4x?x16xf32> {
  %0 = "xla_hlo.broadcast"(%arg) {broadcast_sizes = dense<[4, 2, 1]> : tensor<3xi64>} : (tensor<4x?x16xf32>) -> tensor<4x2x1x4x?x16xf32>
  return %0: tensor<4x2x1x4x?x16xf32>
}
// CHECK: linalg.generic {{{.*}}indexing_maps = [#[[OPERAND_MAP]], #[[RESULT_MAP]]]
// CHECK-NEXT: ^bb0(%[[OPERAND:.*]]: f32):
// CHECK-NEXT:   linalg.yield %[[OPERAND]] : f32

// -----

// CHECK-DAG: #[[OPERAND_MAP:.*]] = affine_map<(d0, d1, d2, d3, d4) -> (d4, d0, 0)>
// CHECK-DAG: #[[RESULT_MAP:.*]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
// CHECK-LABEL: func @broadcast_in_dim
func @broadcast_in_dim(%operand: tensor<5x7x1xf32>) -> tensor<7x10x6x4x5xf32> {
  %0 = "xla_hlo.broadcast_in_dim"(%operand)
         {broadcast_dimensions = dense<[4,0,2]> : tensor<3xi64>}
         : (tensor<5x7x1xf32>) -> tensor<7x10x6x4x5xf32>
  return %0 : tensor<7x10x6x4x5xf32>
}
// CHECK: linalg.generic {{{.*}}indexing_maps = [#[[OPERAND_MAP]], #[[RESULT_MAP]]]
// CHECK-NEXT: ^bb0(%[[OPERAND:.*]]: f32):
// CHECK-NEXT:   linalg.yield %[[OPERAND]] : f32

// -----

// CHECK-DAG: #[[OPERAND_MAP:.+]] = affine_map<(d0, d1) -> (d0)>
// CHECK-DAG: #[[RESULT_MAP:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL: func @broadcast_in_dim_with_one_to_one
func @broadcast_in_dim_with_one_to_one(
         %operand: tensor<1xf32>) -> tensor<1x5xf32> {
  %0 = "xla_hlo.broadcast_in_dim"(%operand)
         {broadcast_dimensions = dense<[0]> : tensor<1xi64>}
         : (tensor<1xf32>) -> tensor<1x5xf32>
  return %0 : tensor<1x5xf32>
}
// CHECK: linalg.generic {{{.*}}indexing_maps = [#[[OPERAND_MAP]], #[[RESULT_MAP]]]
// CHECK-NEXT: ^bb0(%[[OPERAND:.*]]: f32):
// CHECK-NEXT:   linalg.yield %[[OPERAND]] : f32

// -----

// CHECK-DAG: #[[OPERAND_MAP:.*]] = affine_map<(d0, d1, d2) -> ()>
// CHECK-DAG: #[[RESULT_MAP:.*]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK-LABEL: func @broadcast_scalar
func @broadcast_scalar(%operand: tensor<f32>) -> tensor<7x10x6xf32> {
  %0 = "xla_hlo.broadcast_in_dim"(%operand)
        {broadcast_dimensions = dense<[]> : tensor<0xi64>}
        : (tensor<f32>) -> tensor<7x10x6xf32>
  return %0 : tensor<7x10x6xf32>
}
// CHECK: linalg.generic {{{.*}}indexing_maps = [#[[OPERAND_MAP]], #[[RESULT_MAP]]]
// CHECK-NEXT: ^bb0(%[[OPERAND:.*]]: f32):
// CHECK-NEXT:   linalg.yield %[[OPERAND]] : f32

// -----

// CHECK-DAG: #[[OPERAND_MAP:.*]] = affine_map<(d0, d1, d2, d3) -> (d1, d0, d3, d2)>
// CHECK-DAG: #[[RESULT_MAP:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK-LABEL: func @transpose
func @transpose(%arg0: tensor<2x3x9x5xi32>) -> tensor<3x2x5x9xi32> {
  %0 = "xla_hlo.transpose"(%arg0) {permutation = dense<[1, 0, 3, 2]> : tensor<4xi64>}
        : (tensor<2x3x9x5xi32>) -> tensor<3x2x5x9xi32>
  return %0 : tensor<3x2x5x9xi32>
}
// CHECK: linalg.generic {{{.*}}indexing_maps = [#[[OPERAND_MAP]], #[[RESULT_MAP]]]

// -----

// CHECK-DAG: #[[OPERAND_MAP:.*]] = affine_map<(d0, d1) -> (d0, 0, d1)>
// CHECK-DAG: #[[RESULT_MAP:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL: func @reshape_3D_2D
func @reshape_3D_2D(%arg0: tensor<12x1x42xi32>) -> tensor<12x42xi32> {
  %0 = "xla_hlo.reshape"(%arg0) : (tensor<12x1x42xi32>) -> tensor<12x42xi32>
  return %0 : tensor<12x42xi32>
}
// CHECK: linalg.generic {{{.*}}indexing_maps = [#[[OPERAND_MAP]], #[[RESULT_MAP]]]

// -----

// CHECK-DAG: #[[OPERAND_MAP:.*]] = affine_map<(d0, d1) -> (d0, d1, 0, 0)>
// CHECK-DAG: #[[RESULT_MAP:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL: func @reshape_4D_2D
func @reshape_4D_2D(%arg0: tensor<12x42x1x1xi32>) -> tensor<12x42xi32> {
  %0 = "xla_hlo.reshape"(%arg0) : (tensor<12x42x1x1xi32>) -> tensor<12x42xi32>
  return %0 : tensor<12x42xi32>
}
// CHECK: linalg.generic {{{.*}}indexing_maps = [#[[OPERAND_MAP]], #[[RESULT_MAP]]]

// -----

// CHECK-DAG: #[[OPERAND_MAP:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d2)>
// CHECK-DAG: #[[RESULT_MAP:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK-LABEL: func @reshape_2D_4D
func @reshape_2D_4D(%arg0: tensor<12x42xi32>) -> tensor<12x1x42x1xi32> {
  %0 = "xla_hlo.reshape"(%arg0) : (tensor<12x42xi32>) -> tensor<12x1x42x1xi32>
  return %0 : tensor<12x1x42x1xi32>
}
// CHECK: linalg.generic {{{.*}}indexing_maps = [#[[OPERAND_MAP]], #[[RESULT_MAP]]]

// -----

// CHECK-LABEL: func @minf
func @minf(%lhs: tensor<2x2xf32>, %rhs: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %0 = "xla_hlo.minimum"(%lhs, %rhs)
          : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  return %0 : tensor<2x2xf32>
}
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[LHS_IN:.*]]: f32, %[[RHS_IN:.*]]: f32):
// CHECK-NEXT:   %[[CMP:.*]] = cmpf "olt", %[[LHS_IN]], %[[RHS_IN]] : f32
// CHECK-NEXT:   %[[RESULT:.*]] = select %[[CMP]], %[[LHS_IN]], %[[RHS_IN]] : f32
// CHECK-NEXT:   linalg.yield %[[RESULT]] : f32

// -----

// CHECK-LABEL: func @maxi
func @maxi(%lhs: tensor<2x2xi32>, %rhs: tensor<2x2xi32>) -> tensor<2x2xi32> {
  %0 = "xla_hlo.maximum"(%lhs, %rhs)
          : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
  return %0 : tensor<2x2xi32>
}
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[LHS_IN:.*]]: i32, %[[RHS_IN:.*]]: i32):
// CHECK-NEXT:   %[[CMP:.*]] = cmpi "sgt", %[[LHS_IN]], %[[RHS_IN]] : i32
// CHECK-NEXT:   %[[RESULT:.*]] = select %[[CMP]], %[[LHS_IN]], %[[RHS_IN]] : i32
// CHECK-NEXT:   linalg.yield %[[RESULT]] : i32

// -----

// CHECK-DAG: #[[MAP:.*]] = affine_map<() -> ()>
// CHECK-LABEL: func @add_scalar
func @add_scalar(%lhs: tensor<f32>, %rhs: tensor<f32>) -> tensor<f32> {
  %0 = "xla_hlo.add"(%lhs, %rhs) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  return %0 : tensor<f32>
}
// CHECK: linalg.generic {{{.*}}indexing_maps = [#[[MAP]], #[[MAP]], #[[MAP]]]
// CHECK-NEXT: ^bb0(%[[LHS:.*]]: f32, %[[RHS:.*]]: f32):
// CHECK: %[[RESULT:.*]] = addf %[[LHS]], %[[RHS]]
// CHECK-NEXT:   linalg.yield %[[RESULT]] : f32

// -----

func @reshape_collapse_single_dim
  (%arg0: tensor<1x28x28x1xf32>) -> tensor<1x784xf32> {
  %0 = "xla_hlo.reshape"(%arg0) : (tensor<1x28x28x1xf32>) -> tensor<1x784xf32>
  return %0 : tensor<1x784xf32>
}
//   CHECK-DAG: #[[MAP0:.*]] = affine_map<(d0, d1, d2, d3) -> (d0)>
//   CHECK-DAG: #[[MAP1:.*]] = affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>
// CHECK-LABEL: func @reshape_collapse_single_dim
//       CHECK: linalg.tensor_reshape %{{.*}} [#[[MAP0]], #[[MAP1]]]

// -----

func @reshape_collapse(%arg0: tensor<2x2x2x3xf32>) -> tensor<2x4x3xf32> {
    %0 = "xla_hlo.reshape"(%arg0) : (tensor<2x2x2x3xf32>) -> tensor<2x4x3xf32>
    return %0 : tensor<2x4x3xf32>
}
//   CHECK-DAG: #[[MAP0:.*]] = affine_map<(d0, d1, d2, d3) -> (d0)>
//   CHECK-DAG: #[[MAP1:.*]] = affine_map<(d0, d1, d2, d3) -> (d1, d2)>
//   CHECK-DAG: #[[MAP2:.*]] = affine_map<(d0, d1, d2, d3) -> (d3)>
// CHECK-LABEL: func @reshape_collapse
//       CHECK: linalg.tensor_reshape %{{.*}} [#[[MAP0]], #[[MAP1]], #[[MAP2]]]

// -----

func @reshape_expand(%arg0: tensor<2x8xf32>) -> tensor<2x4x2xf32> {
    %0 = "xla_hlo.reshape"(%arg0) : (tensor<2x8xf32>) -> tensor<2x4x2xf32>
    return %0 : tensor<2x4x2xf32>
}
//   CHECK-DAG: #[[MAP0:.*]] = affine_map<(d0, d1, d2) -> (d0)>
//   CHECK-DAG: #[[MAP1:.*]] = affine_map<(d0, d1, d2) -> (d1, d2)>
// CHECK-LABEL: func @reshape_expand
//       CHECK: linalg.tensor_reshape %{{.*}} [#[[MAP0]], #[[MAP1]]]

// -----

func @reshape_single_expand(%arg0 : tensor<8xf32>) -> tensor<1x4x2xf32> {
    %0 = "xla_hlo.reshape"(%arg0) : (tensor<8xf32>) -> tensor<1x4x2xf32>
    return %0 : tensor<1x4x2xf32>
}
//       CHECK: #[[MAP0:.*]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK-LABEL: func @reshape_single_expand
//       CHECK: linalg.tensor_reshape %{{.*}} [#[[MAP0]]]

// -----

func @reshape_multiple_collapse
  (%arg0 : tensor<1x2x2x5x3x2xf32>) -> tensor<1x4x5x6xf32> {
    %0 = "xla_hlo.reshape"(%arg0) : (tensor<1x2x2x5x3x2xf32>) -> tensor<1x4x5x6xf32>
    return %0 : tensor<1x4x5x6xf32>
}
//   CHECK-DAG: #[[MAP0:.*]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0)>
//   CHECK-DAG: #[[MAP1:.*]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2)>
//   CHECK-DAG: #[[MAP2:.*]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d3)>
//   CHECK-DAG: #[[MAP3:.*]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d4, d5)>
// CHECK-LABEL: func @reshape_multiple_collapse
//       CHECK: linalg.tensor_reshape %{{.*}} [#[[MAP0]], #[[MAP1]], #[[MAP2]], #[[MAP3]]]

// -----

// CHECK-LABEL: func @convert_i32_to_f32
func @convert_i32_to_f32(%input: tensor<2x2xi32>) -> tensor<2x2xf32> {
  %result = "xla_hlo.convert"(%input) : (tensor<2x2xi32>) -> tensor<2x2xf32>
  return %result : tensor<2x2xf32>
}
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[OPERAND_IN:.*]]: i32):
// CHECK-NEXT:   %[[RESULT:.*]] = sitofp %[[OPERAND_IN]] : i32 to f32
// CHECK-NEXT:   linalg.yield %[[RESULT]] : f32

// -----

// CHECK-LABEL: func @convert_i16_to_i32
func @convert_i16_to_i32(%input: tensor<2x2xi16>) -> tensor<2x2xi32> {
  %result = "xla_hlo.convert"(%input) : (tensor<2x2xi16>) -> tensor<2x2xi32>
  return %result : tensor<2x2xi32>
}
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[OPERAND_IN:.*]]: i16):
// CHECK-NEXT:   %[[RESULT:.*]] = zexti %[[OPERAND_IN]] : i16 to i32
// CHECK-NEXT:   linalg.yield %[[RESULT]] : i32

// -----

// CHECK-LABEL: func @convert_i32_to_i16
func @convert_i32_to_i16(%input: tensor<2x2xi32>) -> tensor<2x2xi16> {
  %result = "xla_hlo.convert"(%input) : (tensor<2x2xi32>) -> tensor<2x2xi16>
  return %result : tensor<2x2xi16>
}
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[OPERAND_IN:.*]]: i32):
// CHECK-NEXT:   %[[RESULT:.*]] = trunci %[[OPERAND_IN]] : i32 to i16
// CHECK-NEXT:   linalg.yield %[[RESULT]] : i16

// -----

// CHECK-LABEL: func @convert_f32_to_f64
func @convert_f32_to_f64(%input: tensor<2x2xf32>) -> tensor<2x2xf64> {
  %result = "xla_hlo.convert"(%input) : (tensor<2x2xf32>) -> tensor<2x2xf64>
  return %result : tensor<2x2xf64>
}
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[OPERAND_IN:.*]]: f32):
// CHECK-NEXT:   %[[RESULT:.*]] = fpext %[[OPERAND_IN]] : f32 to f64
// CHECK-NEXT:   linalg.yield %[[RESULT]] : f64

// -----

// CHECK-LABEL: func @convert_f64_to_f32
func @convert_f64_to_f32(%input: tensor<2x2xf64>) -> tensor<2x2xf32> {
  %result = "xla_hlo.convert"(%input) : (tensor<2x2xf64>) -> tensor<2x2xf32>
  return %result : tensor<2x2xf32>
}
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[OPERAND_IN:.*]]: f64):
// CHECK-NEXT:   %[[RESULT:.*]] = fptrunc %[[OPERAND_IN]] : f64 to f32
// CHECK-NEXT:   linalg.yield %[[RESULT]] : f32

// -----

// CHECK-LABEL: func @convert_f32_to_i32
func @convert_f32_to_i32(%input: tensor<2x2xf32>) -> tensor<2x2xi32> {
  %result = "xla_hlo.convert"(%input) : (tensor<2x2xf32>) -> tensor<2x2xi32>
  return %result : tensor<2x2xi32>
}
// CHECK: linalg.generic
// CHECK-NEXT: ^bb0(%[[OPERAND_IN:.*]]: f32):
// CHECK-NEXT:   %[[RESULT:.*]] = fptosi %[[OPERAND_IN]] : f32 to i32
// CHECK-NEXT:   linalg.yield %[[RESULT]] : i32

// -----

// CHECK-DAG: #[[OPERAND_MAP:.*]] = affine_map<(d0, d1) -> (d0, -d1 + 2)>
// CHECK-DAG: #[[RESULT_MAP:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL: func @reverse
func @reverse(%input: tensor<2x3xf32>) -> tensor<2x3xf32> {
  %result = "xla_hlo.reverse"(%input) {
    dimensions = dense<1> : tensor<1xi64>
  } : (tensor<2x3xf32>) -> tensor<2x3xf32>
  return %result : tensor<2x3xf32>
}
// CHECK: linalg.generic {{{.*}}indexing_maps = [#[[OPERAND_MAP]], #[[RESULT_MAP]]]
