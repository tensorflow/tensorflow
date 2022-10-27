// RUN: xla-opt -xla-legalize-tf=allow-partial-conversion %s | FileCheck %s

//===----------------------------------------------------------------------===//
// tf.BatchMatMulV2 op legalizations.
//===----------------------------------------------------------------------===//

func.func @batchmatmulv2_basic(%arg0: tensor<1x4x2xf32>, %arg1: tensor<3x2x4xf32>) -> tensor<3x4x4xf32> {
// CHECK-LABEL:   func @batchmatmulv2_basic
// CHECK-SAME:        ([[LHS:%.*]]: tensor<1x4x2xf32>, [[RHS:%.*]]: tensor<3x2x4xf32>) -> tensor<3x4x4xf32>
// CHECK:           [[LHSSHAPE:%.*]] = shape.shape_of [[LHS]] : tensor<1x4x2xf32>
// CHECK:           [[RHSSHAPE:%.*]] = shape.shape_of [[RHS]] : tensor<3x2x4xf32>
// CHECK:           [[CM2:%.*]] = arith.constant -2 : index
// CHECK:           [[LHSHEAD:%.*]], [[LHSTAIL:%.*]] = "shape.split_at"([[LHSSHAPE]], [[CM2]])
// CHECK:           [[RHSHEAD:%.*]], [[RHSTAIL:%.*]] = "shape.split_at"([[RHSSHAPE]], [[CM2]])
// CHECK:           [[BCASTHEAD:%.*]] = shape.broadcast [[LHSHEAD]], [[RHSHEAD]]
// CHECK:           [[LHSBCASTSHAPE:%.*]] = shape.concat [[BCASTHEAD]], [[LHSTAIL]]
// CHECK:           [[LHSSHAPEEXTENTS:%.*]] = shape.to_extent_tensor [[LHSBCASTSHAPE]]
// CHECK:           [[LHSBCAST:%.*]] = "mhlo.dynamic_broadcast_in_dim"([[LHS]], [[LHSSHAPEEXTENTS]]) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<1x4x2xf32>, tensor<3xindex>) -> tensor<3x4x2xf32>
// CHECK:           [[RHSBCASTSHAPE:%.*]] = shape.concat [[BCASTHEAD]], [[RHSTAIL]]
// CHECK:           [[RESULT:%.*]] = "mhlo.dot_general"([[LHSBCAST]], [[RHS]])
// CHECK:           return [[RESULT]] : tensor<3x4x4xf32>
// CHECK:         }

  %0 = "tf.BatchMatMulV2"(%arg0, %arg1) {T = f32, adj_x = false, adj_y = false, device = ""} : (tensor<1x4x2xf32>, tensor<3x2x4xf32>) -> tensor<3x4x4xf32>
  func.return %0 : tensor<3x4x4xf32>
}

func.func @batchmatmulv2_lhs_batch(%arg0: tensor<3x4x2xf32>, %arg1: tensor<2x4xf32>) -> tensor<3x4x4xf32> {
// CHECK-LABEL:   func @batchmatmulv2_lhs_batch
// CHECK:           "mhlo.dynamic_broadcast_in_dim"({{.*}}, {{.*}}) {broadcast_dimensions = dense<[1, 2]> : tensor<2xi64>}
// CHECK:           "mhlo.dot_general"({{.*}}, {{.*}}) {
// CHECK-SAME:        lhs_batching_dimensions = [0]
// CHECK-SAME:        rhs_batching_dimensions = [0]
// CHECK-SAME:        lhs_contracting_dimensions = [2]
// CHECK-SAME:        rhs_contracting_dimensions = [1]
  %0 = "tf.BatchMatMulV2"(%arg0, %arg1) {T = f32, adj_x = false, adj_y = false, device = ""} : (tensor<3x4x2xf32>, tensor<2x4xf32>) -> tensor<3x4x4xf32>
  func.return %0 : tensor<3x4x4xf32>
}

func.func @batchmatmulv2_rhs_batch(%arg0: tensor<4x2xf32>, %arg1: tensor<3x2x4xf32>) -> tensor<3x4x4xf32> {
// CHECK-LABEL:   func @batchmatmulv2_rhs_batch
// CHECK:           "mhlo.dynamic_broadcast_in_dim"({{.*}}, {{.*}}) {broadcast_dimensions = dense<[1, 2]> : tensor<2xi64>}
// CHECK:           "mhlo.dot_general"({{.*}}, {{.*}}) {
// CHECK-SAME:        lhs_batching_dimensions = [0]
// CHECK-SAME:        rhs_batching_dimensions = [0]
// CHECK-SAME:        lhs_contracting_dimensions = [2]
// CHECK-SAME:        rhs_contracting_dimensions = [1]
  %0 = "tf.BatchMatMulV2"(%arg0, %arg1) {T = f32, adj_x = false, adj_y = false, device = ""} : (tensor<4x2xf32>, tensor<3x2x4xf32>) -> tensor<3x4x4xf32>
  func.return %0 : tensor<3x4x4xf32>
}

func.func @batchmatmulv2_dynamic(%arg0: tensor<?x?x?xf32>, %arg1: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
// CHECK-LABEL:   func @batchmatmulv2_dynamic
// CHECK:           "mhlo.dot_general"({{.*}}, {{.*}}) {
// CHECK-SAME:  lhs_batching_dimensions = [0]
// CHECK-SAME:  rhs_batching_dimensions = [0]
// CHECK-SAME:  lhs_contracting_dimensions = [2]
// CHECK-SAME:  rhs_contracting_dimensions = [1]
  %0 = "tf.BatchMatMulV2"(%arg0, %arg1) {T = f32, adj_x = false, adj_y = false, device = ""} : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  func.return %0 : tensor<?x?x?xf32>
}

func.func @batchmatmulv2_adj_real(%arg0: tensor<2x5xf32>, %arg1: tensor<4x2xf32>) -> tensor<5x4xf32> {
// CHECK-LABEL:   func @batchmatmulv2_adj_real
// CHECK:           "mhlo.dot_general"({{.*}}, {{.*}}) {
// CHECK-NOT:         lhs_batching_dimensions
// CHECK-NOT:         rhs_batching_dimensions
// CHECK-SAME:        lhs_contracting_dimensions = [0]
// CHECK-SAME:        rhs_contracting_dimensions = [1]
  %0 = "tf.BatchMatMulV2"(%arg0, %arg1) {adj_x = true, adj_y = true, device = ""} : (tensor<2x5xf32>, tensor<4x2xf32>) -> tensor<5x4xf32>
  func.return %0 : tensor<5x4xf32>
}

func.func @batchmatmulv2_adj_complex(%arg0: tensor<2x5xcomplex<f32>>, %arg1: tensor<4x2xcomplex<f32>>) -> tensor<5x4xcomplex<f32>> {
// CHECK-LABEL:   func @batchmatmulv2_adj_complex(
// CHECK-SAME:                                    [[LHS:%.*]]: tensor<2x5xcomplex<f32>>, [[RHS:%.*]]: tensor<4x2xcomplex<f32>>) -> tensor<5x4xcomplex<f32>> {
// CHECK:           [[LHSRE:%.*]] = mhlo.real [[LHS]]
// CHECK:           [[LHSIM:%.*]] = mhlo.imag [[LHS]]
// CHECK:           [[LHSIMNEG:%.*]] = mhlo.negate [[LHSIM]]
// CHECK:           [[LHSCONJ:%.*]] = mhlo.complex [[LHSRE]], [[LHSIMNEG]]
// CHECK:           [[RHSRE:%.*]] = mhlo.real [[RHS]]
// CHECK:           [[RHSIM:%.*]] = mhlo.imag [[RHS]]
// CHECK:           [[RHSIMNEG:%.*]] = mhlo.negate [[RHSIM]]
// CHECK:           [[RHSCONJ:%.*]] = mhlo.complex [[RHSRE]], [[RHSIMNEG]]
// CHECK:           shape.shape_of [[LHSCONJ]]
// CHECK:           shape.shape_of [[RHSCONJ]]
  %0 = "tf.BatchMatMulV2"(%arg0, %arg1) {adj_x = true, adj_y = true, device = ""} : (tensor<2x5xcomplex<f32>>, tensor<4x2xcomplex<f32>>) -> tensor<5x4xcomplex<f32>>
  func.return %0 : tensor<5x4xcomplex<f32>>
}
