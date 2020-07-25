// RUN: mlir-hlo-opt -mhlo-test-chlo-legalize-to-hlo -cse -split-input-file -verify-diagnostics %s -o - | FileCheck %s

// Check the non-broadcast case for each registered op, then just check a
// representative op for detailed broadcast semantics.
// CHECK-LABEL: @addWithoutBroadcast
func @addWithoutBroadcast(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: mhlo.add %arg0, %arg1
  %0 = chlo.broadcast_add %arg0, %arg1 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: @dynamicBroadcast
// CHECK-SAME: %[[ARG0:.+]]: tensor<?xf32>
// CHECK-SAME: %[[ARG1:.+]]: tensor<?x?xf32>
func @dynamicBroadcast(%arg0: tensor<?xf32>, %arg1: tensor<?x?xf32>) -> tensor<?x?xf32> {
  // CHECK-DAG:  %[[ARG0_S:.+]] = shape.shape_of %[[ARG0]]
  // CHECK-DAG:  %[[ARG1_S:.+]] = shape.shape_of %[[ARG1]]
  // CHECK-NEXT: %[[WITNESS:.+]] = shape.cstr_broadcastable %[[ARG0_S]], %[[ARG1_S]]
  // CHECK-NEXT: %[[FINAL_RESULT:.+]] = shape.assuming %[[WITNESS]]
  // CHECK-DAG:    %[[RESULT_S:.+]] = shape.broadcast %[[ARG0_S]], %[[ARG1_S]]
  // CHECK:        %[[RESULT_EXTENTS:.+]] = shape.to_extent_tensor %[[RESULT_S]]
  // CHECK-DAG:    %[[ARG0_B:.+]] = "mhlo.dynamic_broadcast_in_dim"(%[[ARG0]], %[[RESULT_EXTENTS]]) {broadcast_dimensions = dense<1> : tensor<1xi64>}
  // CHECK-DAG:    %[[ARG1_B:.+]] = "mhlo.dynamic_broadcast_in_dim"(%[[ARG1]], %[[RESULT_EXTENTS]]) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>}
  // CHECK-NEXT:   %[[RESULT:.+]] = mhlo.add %[[ARG0_B]], %[[ARG1_B]]
  // CHECK-NEXT:   shape.assuming_yield %[[RESULT]]
  // CHECK-NEXT: }
  // CHECK-NEXT:      return %[[FINAL_RESULT]] : tensor<?x?xf32>
  %0 = chlo.broadcast_add %arg0, %arg1 : (tensor<?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// -----
// CHECK-LABEL: @dynamicBroadcastComplex
// CHECK-SAME: %[[ARG0:.+]]: tensor<?xf32>
// CHECK-SAME: %[[ARG1:.+]]: tensor<?x?xf32>
func @dynamicBroadcastComplex(%arg0: tensor<?xf32>, %arg1: tensor<?x?xf32>) -> tensor<?x?xcomplex<f32>> {
  // CHECK-DAG:  %[[ARG0_S:.+]] = shape.shape_of %[[ARG0]]
  // CHECK-DAG:  %[[ARG1_S:.+]] = shape.shape_of %[[ARG1]]
  // CHECK-NEXT: %[[WITNESS:.+]] = shape.cstr_broadcastable %[[ARG0_S]], %[[ARG1_S]]
  // CHECK-NEXT: %[[FINAL_RESULT:.+]] = shape.assuming %[[WITNESS]]
  // CHECK-NEXT:   %[[RESULT_S:.+]] = shape.broadcast %[[ARG0_S]], %[[ARG1_S]]
  // CHECK-NEXT:   %[[RESULT_EXTENTS:.+]] = shape.to_extent_tensor %[[RESULT_S]]
  // CHECK-DAG:    %[[ARG0_B:.+]] = "mhlo.dynamic_broadcast_in_dim"(%[[ARG0]], %[[RESULT_EXTENTS]]) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<?xf32>, tensor<2xindex>) -> tensor<?x?xf32>
  // CHECK-DAG:    %[[ARG1_B:.+]] = "mhlo.dynamic_broadcast_in_dim"(%[[ARG1]], %[[RESULT_EXTENTS]]) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<?x?xf32>, tensor<2xindex>) -> tensor<?x?xf32>
  // CHECK-NEXT:   %[[RESULT:.+]] = "mhlo.complex"(%[[ARG0_B]], %[[ARG1_B]]) : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xcomplex<f32>>
  // CHECK-NEXT:   shape.assuming_yield %[[RESULT]]
  // CHECK-NEXT: }
  // CHECK-NEXT: return %[[FINAL_RESULT]] : tensor<?x?xcomplex<f32>>
  %0 = chlo.broadcast_complex %arg0, %arg1 : (tensor<?xf32>, tensor<?x?xf32>) -> tensor<?x?xcomplex<f32>>
  return %0 : tensor<?x?xcomplex<f32>>
}

// -----
// CHECK-LABEL: @dynamicBroadcastCompare
// CHECK-SAME: %[[ARG0:.+]]: tensor<?xf32>
// CHECK-SAME: %[[ARG1:.+]]: tensor<?x?xf32>
func @dynamicBroadcastCompare(%arg0: tensor<?xf32>, %arg1: tensor<?x?xf32>) -> tensor<?x?xi1> {
  // CHECK-DAG: %[[ARG0_S:.+]] = shape.shape_of %[[ARG0]]
  // CHECK-DAG: %[[ARG1_S:.+]] = shape.shape_of %[[ARG1]]
  // CHECK: %[[WITNESS:.+]] = shape.cstr_broadcastable %[[ARG0_S]], %[[ARG1_S]]
  // CHECK: %[[FINAL_RESULT:.+]] = shape.assuming %[[WITNESS]]
  // CHECK: %[[RESULT_S:.+]] = shape.broadcast %[[ARG0_S]], %[[ARG1_S]]
  // CHECK: %[[RESULT_EXTENTS:.+]] = shape.to_extent_tensor %[[RESULT_S]]
  // CHECK-DAG: %[[ARG0_B:.+]] = "mhlo.dynamic_broadcast_in_dim"(%[[ARG0]], %[[RESULT_EXTENTS]]) {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<?xf32>, tensor<2xindex>) -> tensor<?x?xf32>
  // CHECK-DAG: %[[ARG1_B:.+]] = "mhlo.dynamic_broadcast_in_dim"(%[[ARG1]], %[[RESULT_EXTENTS]]) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<?x?xf32>, tensor<2xindex>) -> tensor<?x?xf32>
  // CHECK: %[[RESULT:.+]] = "mhlo.compare"(%[[ARG0_B]], %[[ARG1_B]]) {comparison_direction = "EQ"} : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xi1>
  // CHECK: shape.assuming_yield %[[RESULT]]
  // CHECK-NEXT: }
  // CHECK: return %[[FINAL_RESULT]] : tensor<?x?xi1>
  %0 = chlo.broadcast_compare %arg0, %arg1 {comparison_direction = "EQ"} : (tensor<?xf32>, tensor<?x?xf32>) -> tensor<?x?xi1>
  return %0 : tensor<?x?xi1>
}

// -----
// Verifies that broadcast_dimensions validity checks are valid.
// CHECK-LABEL: @dynamicNonScalarBroadcastDimensions
func @dynamicNonScalarBroadcastDimensions(%arg0: tensor<1x4xf32>, %arg1: tensor<4xf32>) -> tensor<1x4xf32> {
  // CHECK: mhlo.add
  %0 = chlo.broadcast_add %arg0, %arg1 {broadcast_dimensions = dense<1> : tensor<1xi64>} : (tensor<1x4xf32>, tensor<4xf32>) -> tensor<1x4xf32>
  return %0 : tensor<1x4xf32>
}

// -----
// Verifies that broadcast_dimensions validity checks are valid.
// CHECK-LABEL: @dynamicNonScalarByScalarBroadcastDimensions
func @dynamicNonScalarByScalarBroadcastDimensions(%arg0: tensor<1x4xf32>, %arg1: tensor<f32>) -> tensor<1x4xf32> {
  // CHECK: mhlo.add
  %0 = chlo.broadcast_add %arg0, %arg1 {broadcast_dimensions = dense<[]> : tensor<0xi64>} : (tensor<1x4xf32>, tensor<f32>) -> tensor<1x4xf32>
  return %0 : tensor<1x4xf32>
}

// -----
// Verifies that invalid broadcast dimensions are rejected.
func @dynamicNonScalarBroadcastDimensionsSizeMismatch(%arg0: tensor<1x4xf32>, %arg1: tensor<4xf32>) -> tensor<1x4xf32> {
  // expected-warning @+2 {{unsupported non prefix-padded dynamic rank broadcast_dimensions}}
  // expected-error @+1 {{failed to legalize operation}}
  %0 = chlo.broadcast_add %arg0, %arg1 {broadcast_dimensions = dense<[1, 2]> : tensor<2xi64>} : (tensor<1x4xf32>, tensor<4xf32>) -> tensor<1x4xf32>
  return %0 : tensor<1x4xf32>
}

// -----
// Verifies that invalid broadcast dimensions are rejected.
func @dynamicNonScalarBroadcastDimensionsMismatch(%arg0: tensor<1x4xf32>, %arg1: tensor<4xf32>) -> tensor<1x4xf32> {
  // expected-warning @+2 {{unsupported non prefix-padded dynamic rank broadcast_dimensions}}
  // expected-error @+1 {{failed to legalize operation}}
  %0 = chlo.broadcast_add %arg0, %arg1 {broadcast_dimensions = dense<2> : tensor<1xi64>} : (tensor<1x4xf32>, tensor<4xf32>) -> tensor<1x4xf32>
  return %0 : tensor<1x4xf32>
}

// -----
// Note that broadcast_add is used as a proxy for all of the template
// expansions. Tests below merely verify that the op has an expansion.
// CHECK-LABEL: @andWithoutBroadcast
func @andWithoutBroadcast(%arg0: tensor<4xi1>, %arg1: tensor<4xi1>) -> tensor<4xi1> {
  // CHECK: mhlo.and %arg0, %arg1
  %0 = chlo.broadcast_and %arg0, %arg1 : (tensor<4xi1>, tensor<4xi1>) -> tensor<4xi1>
  return %0 : tensor<4xi1>
}

// -----
// CHECK-LABEL: @atan2WithoutBroadcast
func @atan2WithoutBroadcast(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: mhlo.atan2 %arg0, %arg1
  %0 = chlo.broadcast_atan2 %arg0, %arg1 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: @compareWithoutBroadcast
func @compareWithoutBroadcast(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xi1> {
  // CHECK: "mhlo.compare"(%arg0, %arg1) {comparison_direction = "EQ"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
  %0 = chlo.broadcast_compare %arg0, %arg1 {comparison_direction = "EQ"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xi1>
  return %0 : tensor<4xi1>
}

// -----
// CHECK-LABEL: @complexWithoutBroadcast
func @complexWithoutBroadcast(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xcomplex<f32>> {
  // CHECK: "mhlo.complex"(%arg0, %arg1) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xcomplex<f32>>
  %0 = chlo.broadcast_complex %arg0, %arg1 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xcomplex<f32>>
  return %0 : tensor<4xcomplex<f32>>
}

// -----
// CHECK-LABEL: @divideWithoutBroadcast
func @divideWithoutBroadcast(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: mhlo.divide %arg0, %arg1
  %0 = chlo.broadcast_divide %arg0, %arg1 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: @maximumWithoutBroadcast
func @maximumWithoutBroadcast(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: mhlo.maximum %arg0, %arg1
  %0 = chlo.broadcast_maximum %arg0, %arg1 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: @minimumWithoutBroadcast
func @minimumWithoutBroadcast(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: mhlo.minimum %arg0, %arg1
  %0 = chlo.broadcast_minimum %arg0, %arg1 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: @multiplyWithoutBroadcast
func @multiplyWithoutBroadcast(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: mhlo.multiply %arg0, %arg1
  %0 = chlo.broadcast_multiply %arg0, %arg1 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: @orWithoutBroadcast
func @orWithoutBroadcast(%arg0: tensor<4xi1>, %arg1: tensor<4xi1>) -> tensor<4xi1> {
  // CHECK: mhlo.or %arg0, %arg1
  %0 = chlo.broadcast_or %arg0, %arg1 : (tensor<4xi1>, tensor<4xi1>) -> tensor<4xi1>
  return %0 : tensor<4xi1>
}

// -----
// CHECK-LABEL: @powerWithoutBroadcast
func @powerWithoutBroadcast(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: mhlo.power %arg0, %arg1
  %0 = chlo.broadcast_power %arg0, %arg1 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: @remainderWithoutBroadcast
func @remainderWithoutBroadcast(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: mhlo.remainder %arg0, %arg1
  %0 = chlo.broadcast_remainder %arg0, %arg1 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: @shift_leftWithoutBroadcast
func @shift_leftWithoutBroadcast(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: mhlo.shift_left %arg0, %arg1
  %0 = chlo.broadcast_shift_left %arg0, %arg1 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: @shift_right_arithmeticWithoutBroadcast
func @shift_right_arithmeticWithoutBroadcast(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: mhlo.shift_right_arithmetic %arg0, %arg1
  %0 = chlo.broadcast_shift_right_arithmetic %arg0, %arg1 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: @shift_right_logicalWithoutBroadcast
func @shift_right_logicalWithoutBroadcast(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: mhlo.shift_right_logical %arg0, %arg1
  %0 = chlo.broadcast_shift_right_logical %arg0, %arg1 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: @subWithoutBroadcast
func @subWithoutBroadcast(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: mhlo.subtract %arg0, %arg1
  %0 = chlo.broadcast_subtract %arg0, %arg1 : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----
// CHECK-LABEL: @xorWithoutBroadcast
func @xorWithoutBroadcast(%arg0: tensor<4xi1>, %arg1: tensor<4xi1>) -> tensor<4xi1> {
  // CHECK: mhlo.xor %arg0, %arg1
  %0 = chlo.broadcast_xor %arg0, %arg1 : (tensor<4xi1>, tensor<4xi1>) -> tensor<4xi1>
  return %0 : tensor<4xi1>
}

// -----
func @addScalarUnranked(%arg0: tensor<f32>, %arg1: tensor<*xf32>) -> tensor<*xf32> {
  %0 = chlo.broadcast_add %arg0, %arg1 : (tensor<f32>, tensor<*xf32>)
                                         -> tensor<*xf32>
  return %0 : tensor<*xf32>
}

// CHECK-LABEL:   func @addScalarUnranked(
// CHECK-SAME:                            %[[ARG_0:.*]]: tensor<f32>,
// CHECK-SAME:                            %[[ARG_1:.*]]: tensor<*xf32>
// CHECK-SAME:                            ) -> tensor<*xf32> {
//                  First handle the dynamic reshaping of the unranked operand
//                  to a 1D tensor.
// CHECK:           %[[SHAPE_1:.*]] = shape.shape_of %[[ARG_1]] : tensor<*xf32>
// CHECK:           %[[NUM_ELEMENTS:.*]] = shape.num_elements %[[SHAPE_1]]
// CHECK:           %[[SIZE:.*]] = shape.size_to_index %[[NUM_ELEMENTS]]
// CHECK:           %[[SIZE_TENSOR:.*]] = tensor_from_elements(%[[SIZE]]) : tensor<1xindex>
// CHECK:           %[[RESHAPED:.*]] = "mhlo.dynamic_reshape"(%[[ARG_1]], %[[SIZE_TENSOR]]) : (tensor<*xf32>, tensor<1xindex>) -> tensor<?xf32>
//                  The assuming region is part of the second stage of lowering
//                  with ranked broadcasting logic.
// CHECK:           %[[SHAPE_0:.*]] = shape.shape_of %[[ARG_0]] : tensor<f32>
// CHECK:           %[[SHAPE_RESHAPED:.*]] = shape.shape_of %[[RESHAPED]] : tensor<?xf32>
// CHECK:           %[[WITNESS:.*]] = shape.cstr_broadcastable %[[SHAPE_0]], %[[SHAPE_RESHAPED]]
// CHECK:           %[[ASSUMING_RESULT:.*]] = shape.assuming %[[WITNESS]] -> (tensor<?xf32>) {
// CHECK:             %[[SCALAR_SHAPE:.*]] = shape.const_shape []
// CHECK:             %[[BROADCASTED_SHAPE:.*]] = shape.broadcast %[[SCALAR_SHAPE]], %[[SHAPE_RESHAPED]]
// CHECK:             %[[SHAPE_TENSOR:.*]] = shape.to_extent_tensor %[[BROADCASTED_SHAPE]] : tensor<1xindex>
// CHECK:             %[[BROADCASTED_LHS:.*]] = "mhlo.dynamic_broadcast_in_dim"(%[[ARG_0]], %[[SHAPE_TENSOR]]) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>, tensor<1xindex>) -> tensor<?xf32>
// CHECK:             %[[BROADCASTED_RHS:.*]] = "mhlo.dynamic_broadcast_in_dim"(%[[RESHAPED]], %[[SHAPE_TENSOR]]) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<?xf32>, tensor<1xindex>) -> tensor<?xf32>
// CHECK:             %[[BROADCASTED_RESULT:.*]] = mhlo.add %[[BROADCASTED_LHS]], %[[BROADCASTED_RHS]] : tensor<?xf32>
// CHECK:             shape.assuming_yield %[[BROADCASTED_RESULT]] : tensor<?xf32>
// CHECK:           }
//                  As part of the unranked logic, the result is reshaped back
//                  to an unranked tensor.
// CHECK:           %[[PROPER_SHAPE_TENSOR:.*]] = shape.to_extent_tensor %[[SHAPE_1]] : tensor<?xindex>
// CHECK:           %[[RESHAPED_RESULT:.*]] = "mhlo.dynamic_reshape"(%[[VAL_19:.*]], %[[PROPER_SHAPE_TENSOR]]) : (tensor<?xf32>, tensor<?xindex>) -> tensor<*xf32>
// CHECK:           return %[[RESHAPED_RESULT]] : tensor<*xf32>
// CHECK:         }

// -----
func @addUnrankedScalar(%arg0: tensor<*xf32>, %arg1: tensor<f32>) -> tensor<*xf32> {
  %0 = chlo.broadcast_add %arg0, %arg1 : (tensor<*xf32>, tensor<f32>)
                                         -> tensor<*xf32>
  return %0 : tensor<*xf32>
}
// CHECK-LABEL:   func @addUnrankedScalar(
// CHECK-SAME:                            %[[ARG_0:.*]]: tensor<*xf32>,
// CHECK-SAME:                            %[[ARG_1:.*]]: tensor<f32>) -> tensor<*xf32> {
//                  First handle the dynamic reshaping of the unranked operand
//                  to a 1D tensor.
// CHECK:           %[[SHAPE_0:.*]] = shape.shape_of %[[ARG_0]] : tensor<*xf32>
// CHECK:           %[[NUM_ELEMENTS:.*]] = shape.num_elements %[[SHAPE_0]]
// CHECK:           %[[SIZE:.*]] = shape.size_to_index %[[NUM_ELEMENTS]]
// CHECK:           %[[SIZE_TENSOR:.*]] = tensor_from_elements(%[[SIZE]]) : tensor<1xindex>
// CHECK:           %[[RESHAPED:.*]] = "mhlo.dynamic_reshape"(%[[ARG_0]], %[[SIZE_TENSOR]]) : (tensor<*xf32>, tensor<1xindex>) -> tensor<?xf32>
//                  The assuming region is part of the second stage of lowering
//                  with ranked broadcasting logic.
// CHECK:           %[[SHAPE_RESHAPED:.*]] = shape.shape_of %[[RESHAPED]] : tensor<?xf32>
// CHECK:           %[[SHAPE_1:.*]] = shape.shape_of %[[ARG_1]] : tensor<f32>
// CHECK:           %[[WITNESS:.*]] = shape.cstr_broadcastable %[[SHAPE_RESHAPED]], %[[SHAPE_1]]
// CHECK:           %[[ASSUMING_RESULT:.*]] = shape.assuming %[[WITNESS]] -> (tensor<?xf32>) {
// CHECK:             %[[SHAPE_TENSOR:.*]] = shape.to_extent_tensor %[[SHAPE_RESHAPED]] : tensor<1xindex>
// CHECK:             %[[BROADCASTED_LHS:.*]] = "mhlo.dynamic_broadcast_in_dim"(%[[RESHAPED]], %[[SHAPE_TENSOR]]) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<?xf32>, tensor<1xindex>) -> tensor<?xf32>
// CHECK:             %[[BROADCASTED_RHS:.*]] = "mhlo.dynamic_broadcast_in_dim"(%[[ARG_1]], %[[SHAPE_TENSOR]]) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>, tensor<1xindex>) -> tensor<?xf32>
// CHECK:             %[[BROADCASTED_RESULT:.*]] = mhlo.add %[[BROADCASTED_LHS]], %[[BROADCASTED_RHS]] : tensor<?xf32>
// CHECK:             shape.assuming_yield %[[BROADCASTED_RESULT]] : tensor<?xf32>
// CHECK:           }
//                  As part of the unranked logic, the result is reshaped back
//                  to an unranked tensor.
// CHECK:           %[[PROPER_SHAPE_TENSOR:.*]] = shape.to_extent_tensor %[[SHAPE_0]] : tensor<?xindex>
// CHECK:           %[[RESHAPED_RESULT:.*]] = "mhlo.dynamic_reshape"(%[[VAL_19:.*]], %[[PROPER_SHAPE_TENSOR]]) : (tensor<?xf32>, tensor<?xindex>) -> tensor<*xf32>
// CHECK:           return %[[RESHAPED_RESULT]] : tensor<*xf32>
// CHECK:         }
