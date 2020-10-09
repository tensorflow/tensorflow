// RUN: mlir-hlo-opt --transform-unranked-hlo --split-input-file %s | FileCheck %s

// Check the validity of expected IR.
// CHECK-LABEL: @sqr_transform_result
func @sqr_transform_result(%a: tensor<*xf32>) -> tensor<*xf32> {

  // Flatten operand shape.
  %shape = shape.shape_of %a : tensor<*xf32> -> tensor<?xindex>
  %num_elements = shape.num_elements %shape : tensor<?xindex> -> index
  %flat_shape = tensor_from_elements %num_elements : tensor<1xindex>
  %flat_a = "mhlo.dynamic_reshape"(%a, %flat_shape)
      : (tensor<*xf32>, tensor<1xindex>) -> tensor<?xf32>

  // Apply operation.
  %flat_b = "mhlo.sqrt"(%flat_a) : (tensor<?xf32>) -> tensor<?xf32>

  // Restore original shape.
  %b = "mhlo.dynamic_reshape"(%flat_b, %shape)
      : (tensor<?xf32>, tensor<?xindex>) -> tensor<*xf32>

  return %b : tensor<*xf32>
}

// -----

// Check transformation of unranked code.
// CHECK-LABEL: @sqrt
// CHECK-SAME: (%[[A:.*]]: tensor<*xf32>)
func @sqrt(%a: tensor<*xf32>) -> tensor<*xf32> {
  // CHECK-NEXT: %[[SHAPE:.*]] = shape.shape_of %[[A]] : tensor<*xf32> -> tensor<?xindex>
  // CHECK-NEXT: %[[NUM_ELEMENTS:.*]] = shape.num_elements %[[SHAPE]]
  // CHECK-NEXT: %[[FLAT_SHAPE:.*]] = tensor_from_elements %[[NUM_ELEMENTS]] : tensor<1xindex>
  // CHECK-NEXT: %[[FLAT_A:.*]] = "mhlo.dynamic_reshape"(%[[A]], %[[FLAT_SHAPE]]) : (tensor<*xf32>, tensor<1xindex>) -> tensor<?xf32>
  // CHECK-NEXT: %[[FLAT_B:.*]] = "mhlo.sqrt"(%[[FLAT_A]]) : (tensor<?xf32>) -> tensor<?xf32>
  // CHECK-NEXT: %[[B:.*]] = "mhlo.dynamic_reshape"(%[[FLAT_B]], %[[SHAPE]]) : (tensor<?xf32>, tensor<?xindex>) -> tensor<*xf32>
  // CHECK-NEXT: return %[[B]] : tensor<*xf32>
  %b = "mhlo.sqrt"(%a) : (tensor<*xf32>) -> tensor<*xf32>
  return %b : tensor<*xf32>
}

// -----

// Not transformed when ranked.
// CHECK-LABEL: @sqrt_ranked
// CHECK-SAME: (%[[A:.*]]: tensor<3x?xf32>)
func @sqrt_ranked(%a: tensor<3x?xf32>) -> tensor<3x?xf32> {
  // CHECK-NEXT: %[[B:.*]] = "mhlo.sqrt"(%[[A]]) : (tensor<3x?xf32>) -> tensor<3x?xf32>
  // CHECK-NEXT: return %[[B]] : tensor<3x?xf32>
  %b = "mhlo.sqrt"(%a) : (tensor<3x?xf32>) -> tensor<3x?xf32>
  return %b : tensor<3x?xf32>
}

// -----

// Not transformed when statically shaped.
// CHECK-LABEL: @sqrt_static
// CHECK-SAME: (%[[A:.*]]: tensor<2x3xf32>)
func @sqrt_static(%a: tensor<2x3xf32>) -> tensor<2x3xf32> {
  // CHECK-NEXT: %[[B:.*]] = "mhlo.sqrt"(%[[A]]) : (tensor<2x3xf32>) -> tensor<2x3xf32>
  // CHECK-NEXT: return %[[B]] : tensor<2x3xf32>
  %b = "mhlo.sqrt"(%a) : (tensor<2x3xf32>) -> tensor<2x3xf32>
  return %b : tensor<2x3xf32>
}

// -----

// CHECK-LABEL: @add_unranked
// CHECK-SAME:  (%[[A:.*]]: tensor<*xf32>, %[[B:.*]]: tensor<*xf32>) -> tensor<*xf32>
func @add_unranked(%a : tensor<*xf32>, %b : tensor<*xf32>) -> tensor<*xf32> {
  // CHECK: %[[SHAPE_A:.*]] = shape.shape_of %[[A]]
  // CHECK: %[[SHAPE_B:.*]] = shape.shape_of %[[B]]
  // CHECK: %[[SHAPE:.*]] = shape.any %[[SHAPE_A]], %[[SHAPE_B]]
  // CHECK: %[[NUM_ELEMENTS:.*]] = shape.num_elements %[[SHAPE]]
  // CHECK: %[[FLAT_SHAPE:.*]] = tensor_from_elements %[[NUM_ELEMENTS]] : tensor<1xindex>
  // CHECK: %[[FLAT_A:.*]] = "mhlo.dynamic_reshape"(%[[A]], %[[FLAT_SHAPE]]) : (tensor<*xf32>, tensor<1xindex>) -> tensor<?xf32>
  // CHECK: %[[FLAT_B:.*]] = "mhlo.dynamic_reshape"(%[[B]], %[[FLAT_SHAPE]]) : (tensor<*xf32>, tensor<1xindex>) -> tensor<?xf32>
  // CHECK: %[[FLAT_RESULT:.*]] = mhlo.add %[[FLAT_A]], %[[FLAT_B]] : tensor<?xf32>
  // CHECK: %[[RESULT:.*]] = "mhlo.dynamic_reshape"(%[[FLAT_RESULT]], %[[SHAPE]]) : (tensor<?xf32>, tensor<?xindex>) -> tensor<*xf32>
  // CHECK: return %[[RESULT]] : tensor<*xf32>
  %result = mhlo.add %a, %b : tensor<*xf32>
  return %result : tensor<*xf32>
}

// -----

// CHECK-LABEL: @tan
// CHECK-SAME: (%[[A:.*]]: tensor<*xf32>) -> tensor<*xf32>
func @tan(%a : tensor<*xf32>) -> tensor<*xf32> {
  // CHECK: %[[SHAPE:.*]] = shape.shape_of %[[A]] : tensor<*xf32> -> tensor<?xindex>
  // CHECK: %[[NUM_ELEMENTS:.*]] = shape.num_elements %[[SHAPE]]
  // CHECK: %[[FLAT_SHAPE:.*]] = tensor_from_elements %[[NUM_ELEMENTS]] : tensor<1xindex>
  // CHECK: %[[FLAT_A:.*]] = "mhlo.dynamic_reshape"(%[[A]], %[[FLAT_SHAPE]]) : (tensor<*xf32>, tensor<1xindex>) -> tensor<?xf32>
  // CHECK: %[[FLAT_B:.*]] = chlo.tan %[[FLAT_A]] : tensor<?xf32>
  // CHECK: %[[B:.*]] = "mhlo.dynamic_reshape"(%[[FLAT_B]], %[[SHAPE]]) : (tensor<?xf32>, tensor<?xindex>) -> tensor<*xf32>
  // CHECK: return %[[B]] : tensor<*xf32>
  %result = chlo.tan %a : tensor<*xf32>
  return %result : tensor<*xf32>
}
