// RUN: xla-opt -test-xla-infer-shaped-type-methods -allow-unregistered-dialect -split-input-file -verify-diagnostics %s -o - | FileCheck --dump-input=fail %s

// CHECK-LABEL: @broadcast_add
// Note that all broadcast_ops are expanded from the same template, so
// only test reification on an examplar op.
// CHECK-SAME: %[[ARG0:.+]]: tensor<?xf32>,
// CHECK-SAME: %[[ARG1:.+]]: tensor<?xf32>
func @broadcast_add(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>) -> tensor<1xindex> {
  // CHECK-DAG: %[[ARG0_S:.+]] = "shape.shape_of"(%[[ARG0]])
  // CHECK-DAG: %[[ARG1_S:.+]] = "shape.shape_of"(%[[ARG1]])
  // CHECK-DAG: %[[BCAST_S:.+]] = "shape.broadcast"(%[[ARG0_S]], %[[ARG1_S]])
  // CHECK: %[[EXTENTS:.+]] = "shape.to_extent_tensor"(%[[BCAST_S]])
  // CHECK: return %[[EXTENTS]]
  %0 = xla_chlo.broadcast_add %arg0, %arg1 : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  %1 = "xla_test.reify_return_type_shapes"(%0) : (tensor<?xf32>) -> tensor<1xindex>
  return %1 : tensor<1xindex>
}

// -----
// CHECK-LABEL: @complex_ranked_components
func @complex_ranked_components(%arg0: tensor<?xf32>, %arg1: tensor<?x?xf32>) -> tensor<?x?xcomplex<f32>> {
  %0 = xla_chlo.broadcast_complex %arg0, %arg1 : (tensor<?xf32>, tensor<?x?xf32>) -> tensor<?x?xcomplex<f32>>
  // CHECK: "xla_test.return_type_components"(%0) {dims0 = [-1, -1], element_type0 = complex<f32>}
  %1 = "xla_test.get_return_type_components"(%0) : (tensor<?x?xcomplex<f32>>) -> tensor<?x?xcomplex<f32>>
  return %1 : tensor<?x?xcomplex<f32>>
}

// -----
// CHECK-LABEL: @compare_ranked_components
func @compare_ranked_components(%arg0: tensor<?xf32>, %arg1: tensor<?x?xf32>) -> tensor<?x?xi1> {
  %0 = xla_chlo.broadcast_compare %arg0, %arg1 {comparison_direction = "EQ"} : (tensor<?xf32>, tensor<?x?xf32>) -> tensor<?x?xi1>
  // CHECK: "xla_test.return_type_components"(%0) {dims0 = [-1, -1], element_type0 = i1}
  %1 = "xla_test.get_return_type_components"(%0) : (tensor<?x?xi1>) -> tensor<?x?xi1>
  return %0 : tensor<?x?xi1>
}

// -----
// CHECK-LABEL: @broadcast_add_ranked_components_r1
func @broadcast_add_ranked_components_r1(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>) -> tensor<?xf32> {
  %0 = xla_chlo.broadcast_add %arg0, %arg1 : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  // CHECK: "xla_test.return_type_components"(%0) {dims0 = [-1], element_type0 = f32}
  %1 = "xla_test.get_return_type_components"(%0) : (tensor<?xf32>) -> tensor<?xf32>
  return %1 : tensor<?xf32>
}

// -----
// CHECK-LABEL: @broadcast_add_ranked_components_r1x2
func @broadcast_add_ranked_components_r1x2(%arg0: tensor<?xf32>, %arg1: tensor<?x3xf32>) -> tensor<?x3xf32> {
  %0 = xla_chlo.broadcast_add %arg0, %arg1 : (tensor<?xf32>, tensor<?x3xf32>) -> tensor<?x3xf32>
  // TODO: Overly broad shapes are being returned. Tighten the calculation
  // and update/extend these tests.
  // CHECK: "xla_test.return_type_components"(%0) {dims0 = [-1, -1], element_type0 = f32}
  %1 = "xla_test.get_return_type_components"(%0) : (tensor<?x3xf32>) -> tensor<?x3xf32>
  return %1 : tensor<?x3xf32>
}

