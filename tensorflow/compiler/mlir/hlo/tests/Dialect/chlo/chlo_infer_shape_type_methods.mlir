// RUN: mlir-hlo-opt --mhlo-test-infer-shaped-type-methods --allow-unregistered-dialect --split-input-file %s | FileCheck %s

// CHECK-LABEL: @broadcast_add
// Note that all broadcast_ops are expanded from the same template, so
// only test reification on an examplar op.
// CHECK-SAME: %[[ARG0:.+]]: tensor<?xf32>,
// CHECK-SAME: %[[ARG1:.+]]: tensor<?xf32>
func.func @broadcast_add(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>) -> tensor<1xindex> {
  // CHECK-DAG: %[[ARG0_S:.+]] = shape.shape_of %[[ARG0]]
  // CHECK-DAG: %[[ARG1_S:.+]] = shape.shape_of %[[ARG1]]
  // CHECK-DAG: %[[BCAST_S:.+]] = shape.broadcast %[[ARG0_S]], %[[ARG1_S]] : tensor<1xindex>, tensor<1xindex> -> tensor<1xindex>
  // CHECK: return %[[BCAST_S]] : tensor<1xindex>
  %0 = chlo.broadcast_add %arg0, %arg1 : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  %1 = "mhlo_test.reify_return_type_shapes"(%0) : (tensor<?xf32>) -> tensor<1xindex>
  func.return %1 : tensor<1xindex>
}

// -----
// CHECK-LABEL: @complex_ranked_components
func.func @complex_ranked_components(%arg0: tensor<?xf32>, %arg1: tensor<?x?xf32>) -> tensor<?x?xcomplex<f32>> {
  %0 = chlo.broadcast_complex %arg0, %arg1 : (tensor<?xf32>, tensor<?x?xf32>) -> tensor<?x?xcomplex<f32>>
  // CHECK: "mhlo_test.return_type_components"(%0) {dims0 = [-1, -1], element_type0 = complex<f32>}
  %1 = "mhlo_test.get_return_type_components"(%0) : (tensor<?x?xcomplex<f32>>) -> tensor<?x?xcomplex<f32>>
  func.return %1 : tensor<?x?xcomplex<f32>>
}

// -----
// CHECK-LABEL: @compare_ranked_components
func.func @compare_ranked_components(%arg0: tensor<?xf32>, %arg1: tensor<?x?xf32>) -> tensor<?x?xi1> {
  %0 = chlo.broadcast_compare %arg0, %arg1 {comparison_direction = #mhlo<"comparison_direction EQ">} : (tensor<?xf32>, tensor<?x?xf32>) -> tensor<?x?xi1>
  // CHECK: "mhlo_test.return_type_components"(%0) {dims0 = [-1, -1], element_type0 = i1}
  %1 = "mhlo_test.get_return_type_components"(%0) : (tensor<?x?xi1>) -> tensor<?x?xi1>
  func.return %0 : tensor<?x?xi1>
}

// -----
// CHECK-LABEL: @broadcast_add_ranked_components_r1
func.func @broadcast_add_ranked_components_r1(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>) -> tensor<?xf32> {
  %0 = chlo.broadcast_add %arg0, %arg1 : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  // CHECK: "mhlo_test.return_type_components"(%0) {dims0 = [-1], element_type0 = f32}
  %1 = "mhlo_test.get_return_type_components"(%0) : (tensor<?xf32>) -> tensor<?xf32>
  func.return %1 : tensor<?xf32>
}

// -----
// CHECK-LABEL: @broadcast_add_ranked_components_r1x2
func.func @broadcast_add_ranked_components_r1x2(%arg0: tensor<?xf32>, %arg1: tensor<?x3xf32>) -> tensor<?x3xf32> {
  %0 = chlo.broadcast_add %arg0, %arg1 : (tensor<?xf32>, tensor<?x3xf32>) -> tensor<?x3xf32>
  // CHECK: "mhlo_test.return_type_components"(%0) {dims0 = [-1, 3], element_type0 = f32}
  %1 = "mhlo_test.get_return_type_components"(%0) : (tensor<?x3xf32>) -> tensor<?x3xf32>
  func.return %1 : tensor<?x3xf32>
}

// -----
// CHECK-LABEL: @broadcast_add_ranked_components_with_zero_r1x2
func.func @broadcast_add_ranked_components_with_zero_r1x2(%arg0: tensor<0xf32>, %arg1: tensor<?x1xf32>) -> tensor<?x0xf32> {
  %0 = chlo.broadcast_add %arg0, %arg1 : (tensor<0xf32>, tensor<?x1xf32>) -> tensor<?x0xf32>
  // CHECK: "mhlo_test.return_type_components"(%0) {dims0 = [-1, 0], element_type0 = f32}
  %1 = "mhlo_test.get_return_type_components"(%0) : (tensor<?x0xf32>) -> tensor<?x0xf32>
  func.return %1 : tensor<?x0xf32>
}
