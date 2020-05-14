// RUN: tf-opt -xla-legalize-tf=allow-partial-conversion %s | FileCheck %s --dump-input-on-failure

//===----------------------------------------------------------------------===//
// tf.BatchMatMulV2 op legalizations.
//===----------------------------------------------------------------------===//

func @batchmatmulv2_basic(%arg0: tensor<1x4x2xf32>, %arg1: tensor<3x2x4xf32>) -> tensor<3x4x4xf32> {
// CHECK-LABEL:   func @batchmatmulv2_basic
// CHECK-SAME:        ([[LHS:%.*]]: tensor<1x4x2xf32>, [[RHS:%.*]]: tensor<3x2x4xf32>) -> tensor<3x4x4xf32>
// CHECK:           [[LHSSHAPE:%.*]] = shape.shape_of [[LHS]] : tensor<1x4x2xf32>
// CHECK:           [[RHSSHAPE:%.*]] = shape.shape_of [[RHS]] : tensor<3x2x4xf32>
// CHECK:           [[CM2:%.*]] = constant -2 : i32
// CHECK:           [[LHSHEAD:%.*]], [[LHSTAIL:%.*]] = "shape.split_at"([[LHSSHAPE]], [[CM2]]) : (!shape.shape, i32) -> (!shape.shape, !shape.shape)
// CHECK:           [[RHSHEAD:%.*]], [[RHSTAIL:%.*]] = "shape.split_at"([[RHSSHAPE]], [[CM2]]) : (!shape.shape, i32) -> (!shape.shape, !shape.shape)
// CHECK:           [[BCASTHEAD:%.*]] = "shape.broadcast"([[LHSHEAD]], [[RHSHEAD]]) : (!shape.shape, !shape.shape) -> !shape.shape
// CHECK:           [[LHSBCASTSHAPE:%.*]] = "shape.concat"([[BCASTHEAD]], [[LHSTAIL]]) : (!shape.shape, !shape.shape) -> !shape.shape
// CHECK:           [[LHSSHAPEEXTENTS:%.*]] = "shape.to_extent_tensor"([[LHSBCASTSHAPE]]) : (!shape.shape) -> tensor<3xindex>
// CHECK:           [[LHSBCAST:%.*]] = "xla_hlo.dynamic_broadcast_in_dim"([[LHS]], [[LHSSHAPEEXTENTS]]) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<1x4x2xf32>, tensor<3xindex>) -> tensor<3x4x2xf32>
// CHECK:           [[RHSBCASTSHAPE:%.*]] = "shape.concat"([[BCASTHEAD]], [[RHSTAIL]]) : (!shape.shape, !shape.shape) -> !shape.shape
// CHECK:           [[RHSSHAPEEXTENTS:%.*]] = "shape.to_extent_tensor"([[RHSBCASTSHAPE]]) : (!shape.shape) -> tensor<3xindex>
// CHECK:           [[RHSBCAST:%.*]] = "xla_hlo.dynamic_broadcast_in_dim"([[RHS]], [[RHSSHAPEEXTENTS]]) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<3x2x4xf32>, tensor<3xindex>) -> tensor<3x2x4xf32>
// CHECK:           [[RESULT:%.*]] = "xla_hlo.dot_general"([[LHSBCAST]], [[RHSBCAST]]) {dot_dimension_numbers = {lhs_batching_dimensions = dense<0> : tensor<1xi64>, lhs_contracting_dimensions = dense<2> : tensor<1xi64>, rhs_batching_dimensions = dense<0> : tensor<1xi64>, rhs_contracting_dimensions = dense<1> : tensor<1xi64>}} : (tensor<3x4x2xf32>, tensor<3x2x4xf32>) -> tensor<3x4x4xf32>
// CHECK:           return [[RESULT]] : tensor<3x4x4xf32>
// CHECK:         }

  %0 = "tf.BatchMatMulV2"(%arg0, %arg1) {T = f32, adj_x = false, adj_y = false, device = ""} : (tensor<1x4x2xf32>, tensor<3x2x4xf32>) -> tensor<3x4x4xf32>
  return %0 : tensor<3x4x4xf32>
}

func @batchmatmulv2_lhs_batch(%arg0: tensor<3x4x2xf32>, %arg1: tensor<2x4xf32>) -> tensor<3x4x4xf32> {
// CHECK-LABEL:   func @batchmatmulv2_lhs_batch
// CHECK:           "xla_hlo.dynamic_broadcast_in_dim"({{.*}}, {{.*}}) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>}
// CHECK:           "xla_hlo.dynamic_broadcast_in_dim"({{.*}}, {{.*}}) {broadcast_dimensions = dense<[1, 2]> : tensor<2xi64>}
// CHECK:           "xla_hlo.dot_general"({{.*}}, {{.*}}) {dot_dimension_numbers = {
// CHECK-SAME:        lhs_batching_dimensions = dense<0> : tensor<1xi64>,
// CHECK-SAME:        lhs_contracting_dimensions = dense<2> : tensor<1xi64>,
// CHECK-SAME:        rhs_batching_dimensions = dense<0> : tensor<1xi64>,
// CHECK-SAME:        rhs_contracting_dimensions = dense<1> : tensor<1xi64>}}
  %0 = "tf.BatchMatMulV2"(%arg0, %arg1) {T = f32, adj_x = false, adj_y = false, device = ""} : (tensor<3x4x2xf32>, tensor<2x4xf32>) -> tensor<3x4x4xf32>
  return %0 : tensor<3x4x4xf32>
}

func @batchmatmulv2_rhs_batch(%arg0: tensor<4x2xf32>, %arg1: tensor<3x2x4xf32>) -> tensor<3x4x4xf32> {
// CHECK-LABEL:   func @batchmatmulv2_rhs_batch
// CHECK:           "xla_hlo.dynamic_broadcast_in_dim"({{.*}}, {{.*}}) {broadcast_dimensions = dense<[1, 2]> : tensor<2xi64>}
// CHECK:           "xla_hlo.dynamic_broadcast_in_dim"({{.*}}, {{.*}}) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>}
// CHECK:           "xla_hlo.dot_general"({{.*}}, {{.*}}) {dot_dimension_numbers = {
// CHECK-SAME:        lhs_batching_dimensions = dense<0> : tensor<1xi64>,
// CHECK-SAME:        lhs_contracting_dimensions = dense<2> : tensor<1xi64>,
// CHECK-SAME:        rhs_batching_dimensions = dense<0> : tensor<1xi64>,
// CHECK-SAME:        rhs_contracting_dimensions = dense<1> : tensor<1xi64>}}
  %0 = "tf.BatchMatMulV2"(%arg0, %arg1) {T = f32, adj_x = false, adj_y = false, device = ""} : (tensor<4x2xf32>, tensor<3x2x4xf32>) -> tensor<3x4x4xf32>
  return %0 : tensor<3x4x4xf32>
}

func @batchmatmulv2_dynamic(%arg0: tensor<?x?x?xf32>, %arg1: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
// CHECK-LABEL:   func @batchmatmulv2_dynamic
// CHECK:           "xla_hlo.dot_general"({{.*}}, {{.*}}) {dot_dimension_numbers = {
// CHECK-SAME:  lhs_batching_dimensions = dense<0> : tensor<1xi64>,
// CHECK-SAME:  lhs_contracting_dimensions = dense<2> : tensor<1xi64>,
// CHECK-SAME:  rhs_batching_dimensions = dense<0> : tensor<1xi64>,
// CHECK-SAME:  rhs_contracting_dimensions = dense<1> : tensor<1xi64>}}
  %0 = "tf.BatchMatMulV2"(%arg0, %arg1) {T = f32, adj_x = false, adj_y = false, device = ""} : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  return %0 : tensor<?x?x?xf32>
}

func @batchmatmulv2_adj_real(%arg0: tensor<5x2xf32>, %arg1: tensor<2x4xf32>) -> tensor<5x4xf32> {
// CHECK-LABEL:   func @batchmatmulv2_adj_real
// CHECK:           "xla_hlo.dot_general"({{.*}}, {{.*}}) {dot_dimension_numbers = {
// CHECK-SAME:        lhs_batching_dimensions = dense<[]> : tensor<0xi64>,
// CHECK-SAME:        lhs_contracting_dimensions = dense<0> : tensor<1xi64>,
// CHECK-SAME:        rhs_batching_dimensions = dense<[]> : tensor<0xi64>,
// CHECK-SAME:        rhs_contracting_dimensions = dense<1> : tensor<1xi64>}}
  %0 = "tf.BatchMatMulV2"(%arg0, %arg1) {adj_x = true, adj_y = true, device = ""} : (tensor<5x2xf32>, tensor<2x4xf32>) -> tensor<5x4xf32>
  return %0 : tensor<5x4xf32>
}

func @batchmatmulv2_adj_complex(%arg0: tensor<5x2xcomplex<f32>>, %arg1: tensor<2x4xcomplex<f32>>) -> tensor<5x4xcomplex<f32>> {
// CHECK-LABEL:   func @batchmatmulv2_adj_complex(
// CHECK-SAME:                                    [[LHS:%.*]]: tensor<5x2xcomplex<f32>>, [[RHS:%.*]]: tensor<2x4xcomplex<f32>>) -> tensor<5x4xcomplex<f32>> {
// CHECK:           [[LHSRE:%.*]] = "xla_hlo.real"([[LHS]])
// CHECK:           [[LHSIM:%.*]] = "xla_hlo.imag"([[LHS]])
// CHECK:           [[LHSIMNEG:%.*]] = "xla_hlo.negate"([[LHSIM]])
// CHECK:           [[LHSCONJ:%.*]] = "xla_hlo.complex"([[LHSRE]], [[LHSIMNEG]])
// CHECK:           [[RHSRE:%.*]] = "xla_hlo.real"([[RHS]])
// CHECK:           [[RHSIM:%.*]] = "xla_hlo.imag"([[RHS]])
// CHECK:           [[RHSIMNEG:%.*]] = "xla_hlo.negate"([[RHSIM]])
// CHECK:           [[RHSCONJ:%.*]] = "xla_hlo.complex"([[RHSRE]], [[RHSIMNEG]])
// CHECK:           shape.shape_of [[LHSCONJ]]
// CHECK:           shape.shape_of [[RHSCONJ]]
  %0 = "tf.BatchMatMulV2"(%arg0, %arg1) {adj_x = true, adj_y = true, device = ""} : (tensor<5x2xcomplex<f32>>, tensor<2x4xcomplex<f32>>) -> tensor<5x4xcomplex<f32>>
  return %0 : tensor<5x4xcomplex<f32>>
}
