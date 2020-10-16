// RUN: mlir-hlo-opt -chlo-legalize-to-hlo -cse -split-input-file -verify-diagnostics %s -o - | FileCheck %s

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
  // CHECK:        %[[RESULT_EXTENTS:.+]] = tensor_cast %[[RESULT_S]] : tensor<?xindex> to tensor<2xindex>
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
  // CHECK-NEXT:   %[[RESULT_EXTENTS:.+]] = tensor_cast %[[RESULT_S]] : tensor<?xindex> to tensor<2xindex>
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
  // CHECK: %[[RESULT_EXTENTS:.+]] = tensor_cast %[[RESULT_S]] : tensor<?xindex> to tensor<2xindex>
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
// CHECK:           %[[NUM_ELEMENTS:.*]] = shape.num_elements %[[SHAPE_1]] : tensor<?xindex> -> index
// CHECK:           %[[SIZE_TENSOR:.*]] = tensor_from_elements %[[NUM_ELEMENTS]] : tensor<1xindex>
// CHECK:           %[[RESHAPED:.*]] = "mhlo.dynamic_reshape"(%[[ARG_1]], %[[SIZE_TENSOR]]) : (tensor<*xf32>, tensor<1xindex>) -> tensor<?xf32>
//                  The assuming region is part of the second stage of lowering
//                  with ranked broadcasting logic.
// CHECK:           %[[SHAPE_0:.*]] = shape.shape_of %[[ARG_0]] : tensor<f32>
// CHECK:           %[[SHAPE_RESHAPED:.*]] = shape.shape_of %[[RESHAPED]] : tensor<?xf32>
// CHECK:           %[[WITNESS:.*]] = shape.cstr_broadcastable %[[SHAPE_0]], %[[SHAPE_RESHAPED]]
// CHECK:           %[[ASSUMING_RESULT:.*]] = shape.assuming %[[WITNESS]] -> (tensor<?xf32>) {
// CHECK:             %[[SCALAR_SHAPE:.*]] = shape.const_shape []
// CHECK:             %[[BROADCASTED_SHAPE:.*]] = shape.broadcast %[[SCALAR_SHAPE]], %[[SHAPE_RESHAPED]]
// CHECK:             %[[SHAPE_TENSOR:.*]] = tensor_cast %[[BROADCASTED_SHAPE]] : tensor<?xindex> to tensor<1xindex>
// CHECK:             %[[BROADCASTED_LHS:.*]] = "mhlo.dynamic_broadcast_in_dim"(%[[ARG_0]], %[[SHAPE_TENSOR]]) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>, tensor<1xindex>) -> tensor<?xf32>
// CHECK:             %[[BROADCASTED_RHS:.*]] = "mhlo.dynamic_broadcast_in_dim"(%[[RESHAPED]], %[[SHAPE_TENSOR]]) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<?xf32>, tensor<1xindex>) -> tensor<?xf32>
// CHECK:             %[[BROADCASTED_RESULT:.*]] = mhlo.add %[[BROADCASTED_LHS]], %[[BROADCASTED_RHS]] : tensor<?xf32>
// CHECK:             shape.assuming_yield %[[BROADCASTED_RESULT]] : tensor<?xf32>
// CHECK:           }
//                  As part of the unranked logic, the result is reshaped back
//                  to an unranked tensor.
// CHECK:           %[[RESHAPED_RESULT:.*]] = "mhlo.dynamic_reshape"(%[[ASSUMING_RESULT:.*]], %[[SHAPE_1]]) : (tensor<?xf32>, tensor<?xindex>) -> tensor<*xf32>
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
// CHECK:           %[[NUM_ELEMENTS:.*]] = shape.num_elements %[[SHAPE_0]] : tensor<?xindex> -> index
// CHECK:           %[[SIZE_TENSOR:.*]] = tensor_from_elements %[[NUM_ELEMENTS]] : tensor<1xindex>
// CHECK:           %[[RESHAPED:.*]] = "mhlo.dynamic_reshape"(%[[ARG_0]], %[[SIZE_TENSOR]]) : (tensor<*xf32>, tensor<1xindex>) -> tensor<?xf32>
//                  The assuming region is part of the second stage of lowering
//                  with ranked broadcasting logic.
// CHECK:           %[[SHAPE_RESHAPED:.*]] = shape.shape_of %[[RESHAPED]] : tensor<?xf32>
// CHECK:           %[[SHAPE_1:.*]] = shape.shape_of %[[ARG_1]] : tensor<f32>
// CHECK:           %[[WITNESS:.*]] = shape.cstr_broadcastable %[[SHAPE_RESHAPED]], %[[SHAPE_1]]
// CHECK:           %[[ASSUMING_RESULT:.*]] = shape.assuming %[[WITNESS]] -> (tensor<?xf32>) {
// CHECK:             %[[ASTENSOR:.*]] = tensor_cast %[[SHAPE_RESHAPED]]
// CHECK:             %[[BROADCASTED_LHS:.*]] = "mhlo.dynamic_broadcast_in_dim"(%[[RESHAPED]], %[[ASTENSOR]]) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<?xf32>, tensor<1xindex>) -> tensor<?xf32>
// CHECK:             %[[BROADCASTED_RHS:.*]] = "mhlo.dynamic_broadcast_in_dim"(%[[ARG_1]], %[[ASTENSOR]]) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>, tensor<1xindex>) -> tensor<?xf32>
// CHECK:             %[[BROADCASTED_RESULT:.*]] = mhlo.add %[[BROADCASTED_LHS]], %[[BROADCASTED_RHS]] : tensor<?xf32>
// CHECK:             shape.assuming_yield %[[BROADCASTED_RESULT]] : tensor<?xf32>
// CHECK:           }
//                  As part of the unranked logic, the result is reshaped back
//                  to an unranked tensor.
// CHECK:           %[[RESHAPED_RESULT:.*]] = "mhlo.dynamic_reshape"(%[[ASSUMING_RESULT:.*]], %[[SHAPE_0]]) : (tensor<?xf32>, tensor<?xindex>) -> tensor<*xf32>
// CHECK:           return %[[RESHAPED_RESULT]] : tensor<*xf32>
// CHECK:         }

// -----
func @addUnrankedUnranked(
      %arg0: tensor<*xf32>, %arg1: tensor<*xf32>) -> tensor<*xf32> {
  %0 = chlo.broadcast_add %arg0, %arg1 : (tensor<*xf32>, tensor<*xf32>)
                                         -> tensor<*xf32>
  return %0 : tensor<*xf32>
}

// CHECK-LABEL:   func @addUnrankedUnranked(
// CHECK-SAME:          %[[LHS:.*]]: tensor<*xf32>,
// CHECK-SAME:          %[[RHS:.*]]: tensor<*xf32>) -> tensor<*xf32> {
// CHECK:           %[[LHS_SHAPE:.*]] = shape.shape_of %[[LHS]] : tensor<*xf32> -> tensor<?xindex>
// CHECK:           %[[RANK_LHS:.*]] = shape.rank %[[LHS_SHAPE]] : tensor<?xindex> -> index
// CHECK:           %[[C0:.*]] = constant 0 : index
// CHECK:           %[[LHS_IS_SCALAR:.*]] = cmpi "eq", %[[RANK_LHS]], %[[C0]] : index
//                  Handle scalar LHS case
// CHECK:           %[[VAL_8:.*]] = scf.if %[[LHS_IS_SCALAR]] -> (tensor<*xf32>) {
// CHECK:             %[[SCALAR_LHS:.*]] = tensor_cast %[[LHS]] : tensor<*xf32> to tensor<f32>
// CHECK:             %[[VAL_10:.*]] = chlo.broadcast_add %[[SCALAR_LHS]], %[[RHS]] : (tensor<f32>, tensor<*xf32>) -> tensor<*xf32>
// CHECK:             scf.yield %[[VAL_10]] : tensor<*xf32>
// CHECK:           } else {
// CHECK:             %[[RHS_SHAPE:.*]] = shape.shape_of %[[RHS]] : tensor<*xf32> -> tensor<?xindex>
// CHECK:             %[[RANK_RHS:.*]] = shape.rank %[[RHS_SHAPE]] : tensor<?xindex> -> index
// CHECK:             %[[RHS_IS_SCALAR:.*]] = cmpi "eq", %[[RANK_RHS]], %[[C0]] : index
  //                  Handle scalar RHS case
// CHECK:             %[[VAL_14:.*]] = scf.if %[[RHS_IS_SCALAR]] -> (tensor<*xf32>) {
// CHECK:               %[[SCALAR_RHS:.*]] = tensor_cast %[[RHS]] : tensor<*xf32> to tensor<f32>
// CHECK:               %[[VAL_16:.*]] = chlo.broadcast_add %[[LHS]], %[[SCALAR_RHS]] : (tensor<*xf32>, tensor<f32>) -> tensor<*xf32>
// CHECK:               scf.yield %[[VAL_16]] : tensor<*xf32>
// CHECK:             } else {
// CHECK:               %[[SHAPES_EQ:.*]] = shape.shape_eq %[[LHS_SHAPE]], %[[RHS_SHAPE]] : tensor<?xindex>, tensor<?xindex>
  //                    Handle scalar RHS case
// CHECK:               %[[VAL_18:.*]] = scf.if %[[SHAPES_EQ]] -> (tensor<*xf32>) {
// CHECK:                 %[[VAL_19:.*]] = mhlo.add %[[LHS]], %[[RHS]] : tensor<*xf32>
// CHECK:                 scf.yield %[[VAL_19]] : tensor<*xf32>
// CHECK:               } else {
// CHECK:                 %[[LHS_RANK:.*]] = rank %[[LHS_SHAPE]] : tensor<?xindex>
// CHECK:                 %[[RHS_RANK:.*]] = rank %[[RHS_SHAPE]] : tensor<?xindex>
// CHECK:                 %[[LHS_RANK_GREATER:.*]] = cmpi "sgt", %[[LHS_RANK]], %[[RHS_RANK]] : index
// CHECK:                 %[[GREATEST_RANK:.*]] = select %[[LHS_RANK_GREATER]], %[[LHS_RANK]], %[[RHS_RANK]] : index
// CHECK:                 %[[C2:.*]] = constant 2 : index
// CHECK:                 %[[GREATEST_RANK_IS_2:.*]] = cmpi "eq", %[[GREATEST_RANK]], %[[C2]] : index
//                        Handle rank 2 specialization
// CHECK:                 %[[VAL_26:.*]] = scf.if %[[GREATEST_RANK_IS_2]] -> (tensor<*xf32>) {
// CHECK:                   %[[CONST_SHAPE_2:.*]] = shape.const_shape [1, 1]
// CHECK:                   %[[BROADCASTED_LHS_2:.*]] = shape.broadcast %[[LHS_SHAPE]], %[[CONST_SHAPE_2]] : tensor<?xindex>, tensor<2xindex> -> tensor<?xindex>
// CHECK:                   %[[CASTED_LHS_2:.*]] = tensor_cast %[[BROADCASTED_LHS_2]] : tensor<?xindex> to tensor<2xindex>
// CHECK:                   %[[BROADCASTED_RHS_2:.*]] = shape.broadcast %[[RHS_SHAPE]], %[[CONST_SHAPE_2]] : tensor<?xindex>, tensor<2xindex> -> tensor<?xindex>
// CHECK:                   %[[CASTED_RHS_2:.*]] = tensor_cast %[[BROADCASTED_RHS_2]] : tensor<?xindex> to tensor<2xindex>
// CHECK:                   %[[RESHAPED_LHS_2:.*]] = "mhlo.dynamic_reshape"(%[[LHS]], %[[CASTED_LHS_2]]) : (tensor<*xf32>, tensor<2xindex>) -> tensor<?x?xf32>
// CHECK:                   %[[RESHAPED_RHS_2:.*]] = "mhlo.dynamic_reshape"(%[[RHS]], %[[CASTED_RHS_2]]) : (tensor<*xf32>, tensor<2xindex>) -> tensor<?x?xf32>
// CHECK:                   %[[RESULT_RANK_2:.*]] = chlo.broadcast_add %[[RESHAPED_LHS_2]], %[[RESHAPED_RHS_2]] : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK:                   %[[RESULT_2:.*]] = tensor_cast %[[RESULT_RANK_2]] : tensor<?x?xf32> to tensor<*xf32>
// CHECK:                   scf.yield %[[RESULT_2]] : tensor<*xf32>
// CHECK:                 } else {
// CHECK:                   %[[C3:.*]] = constant 3 : index
// CHECK:                   %[[GREATEST_RANK_IS_3:.*]] = cmpi "eq", %[[GREATEST_RANK]], %[[C3]] : index
//                          Handle rank 3 specialization
// CHECK:                   %[[VAL_34:.*]] = scf.if %[[GREATEST_RANK_IS_3]] -> (tensor<*xf32>) {
// CHECK:                     %[[CONST_SHAPE_3:.*]] = shape.const_shape [1, 1, 1]
// CHECK:                     %[[BROADCASTED_LHS_3:.*]] = shape.broadcast %[[LHS_SHAPE]], %[[CONST_SHAPE_3]] : tensor<?xindex>, tensor<3xindex> -> tensor<?xindex>
// CHECK:                     %[[CASTED_LHS_3:.*]] = tensor_cast %[[BROADCASTED_LHS_3]] : tensor<?xindex> to tensor<3xindex>
// CHECK:                     %[[BROADCASTED_RHS_3:.*]] = shape.broadcast %[[RHS_SHAPE]], %[[CONST_SHAPE_3]] : tensor<?xindex>, tensor<3xindex> -> tensor<?xindex>
// CHECK:                     %[[CASTED_RHS_3:.*]] = tensor_cast %[[BROADCASTED_RHS_3]] : tensor<?xindex> to tensor<3xindex>
// CHECK:                     %[[RESHAPED_LHS_3:.*]] = "mhlo.dynamic_reshape"(%[[LHS]], %[[CASTED_LHS_3]]) : (tensor<*xf32>, tensor<3xindex>) -> tensor<?x?x?xf32>
// CHECK:                     %[[RESHAPED_RHS_3:.*]] = "mhlo.dynamic_reshape"(%[[RHS]], %[[CASTED_RHS_3]]) : (tensor<*xf32>, tensor<3xindex>) -> tensor<?x?x?xf32>
// CHECK:                     %[[RESULT_RANK_3:.*]] = chlo.broadcast_add %[[RESHAPED_LHS_3]], %[[RESHAPED_RHS_3]] : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
// CHECK:                     %[[RESULT_3:.*]] = tensor_cast %[[RESULT_RANK_3]] : tensor<?x?x?xf32> to tensor<*xf32>
// CHECK:                     scf.yield %[[RESULT_3]] : tensor<*xf32>
// CHECK:                   } else {
// CHECK:                     %[[C4:.*]] = constant 4 : index
// CHECK:                     %[[GREATEST_RANK_IS_4:.*]] = cmpi "eq", %[[GREATEST_RANK]], %[[C4]] : index
//                            Handle rank 4 specialization
// CHECK:                     %[[VAL_42:.*]] = scf.if %[[GREATEST_RANK_IS_4]] -> (tensor<*xf32>) {
// CHECK:                       %[[CONST_SHAPE_4:.*]] = shape.const_shape [1, 1, 1, 1]
// CHECK:                       %[[BROADCASTED_LHS_4:.*]] = shape.broadcast %[[LHS_SHAPE]], %[[CONST_SHAPE_4]] : tensor<?xindex>, tensor<4xindex> -> tensor<?xindex>
// CHECK:                       %[[CASTED_LHS_4:.*]] = tensor_cast %[[BROADCASTED_LHS_4]] : tensor<?xindex> to tensor<4xindex>
// CHECK:                       %[[BROADCASTED_RHS_4:.*]] = shape.broadcast %[[RHS_SHAPE]], %[[CONST_SHAPE_4]] : tensor<?xindex>, tensor<4xindex> -> tensor<?xindex>
// CHECK:                       %[[CASTED_RHS_4:.*]] = tensor_cast %[[BROADCASTED_RHS_4]] : tensor<?xindex> to tensor<4xindex>
// CHECK:                       %[[RESHAPED_LHS_4:.*]] = "mhlo.dynamic_reshape"(%[[LHS]], %[[CASTED_LHS_4]]) : (tensor<*xf32>, tensor<4xindex>) -> tensor<?x?x?x?xf32>
// CHECK:                       %[[RESHAPED_RHS_4:.*]] = "mhlo.dynamic_reshape"(%[[RHS]], %[[CASTED_RHS_4]]) : (tensor<*xf32>, tensor<4xindex>) -> tensor<?x?x?x?xf32>
// CHECK:                       %[[RESULT_RANK_4:.*]] = chlo.broadcast_add %[[RESHAPED_LHS_4]], %[[RESHAPED_RHS_4]] : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
// CHECK:                       %[[RESULT_4:.*]] = tensor_cast %[[RESULT_RANK_4]] : tensor<?x?x?x?xf32> to tensor<*xf32>
// CHECK:                       scf.yield %[[RESULT_4]] : tensor<*xf32>
// CHECK:                     } else {
// CHECK:                       %[[C5:.*]] = constant 5 : index
// CHECK:                       %[[GREATEST_RANK_IS_5:.*]] = cmpi "eq", %[[GREATEST_RANK]], %[[C5]] : index
//                              Handle rank 5 specialization
// CHECK:                       %[[VAL_50:.*]] = scf.if %[[GREATEST_RANK_IS_5]] -> (tensor<*xf32>) {
// CHECK:                         %[[CONST_SHAPE_5:.*]] = shape.const_shape [1, 1, 1, 1, 1]
// CHECK:                         %[[BROADCASTED_LHS_5:.*]] = shape.broadcast %[[LHS_SHAPE]], %[[CONST_SHAPE_5]] : tensor<?xindex>, tensor<5xindex> -> tensor<?xindex>
// CHECK:                         %[[CASTED_LHS_5:.*]] = tensor_cast %[[BROADCASTED_LHS_5]] : tensor<?xindex> to tensor<5xindex>
// CHECK:                         %[[BROADCASTED_RHS_5:.*]] = shape.broadcast %[[RHS_SHAPE]], %[[CONST_SHAPE_5]] : tensor<?xindex>, tensor<5xindex> -> tensor<?xindex>
// CHECK:                         %[[CASTED_RHS_5:.*]] = tensor_cast %[[BROADCASTED_RHS_5]] : tensor<?xindex> to tensor<5xindex>
// CHECK:                         %[[RESHAPED_LHS_5:.*]] = "mhlo.dynamic_reshape"(%[[LHS]], %[[CASTED_LHS_5]]) : (tensor<*xf32>, tensor<5xindex>) -> tensor<?x?x?x?x?xf32>
// CHECK:                         %[[RESHAPED_RHS_5:.*]] = "mhlo.dynamic_reshape"(%[[RHS]], %[[CASTED_RHS_5]]) : (tensor<*xf32>, tensor<5xindex>) -> tensor<?x?x?x?x?xf32>
// CHECK:                         %[[RESULT_RANK_5:.*]] = chlo.broadcast_add %[[RESHAPED_LHS_5]], %[[RESHAPED_RHS_5]] : (tensor<?x?x?x?x?xf32>, tensor<?x?x?x?x?xf32>) -> tensor<?x?x?x?x?xf32>
// CHECK:                         %[[RESULT_5:.*]] = tensor_cast %[[RESULT_RANK_5]] : tensor<?x?x?x?x?xf32> to tensor<*xf32>
// CHECK:                         scf.yield %[[RESULT_5]] : tensor<*xf32>
// CHECK:                       } else {
// CHECK:                         %[[C6:.*]] = constant 6 : index
// CHECK:                         %[[GREATEST_RANK_IS_6:.*]] = cmpi "eq", %[[GREATEST_RANK]], %[[C6]] : index
//                                Handle rank 6 specialization
// CHECK:                         %[[VAL_58:.*]] = scf.if %[[GREATEST_RANK_IS_6]] -> (tensor<*xf32>) {
// CHECK:                           %[[CONST_SHAPE_6:.*]] = shape.const_shape [1, 1, 1, 1, 1, 1]
// CHECK:                           %[[BROADCASTED_LHS_6:.*]] = shape.broadcast %[[LHS_SHAPE]], %[[CONST_SHAPE_6]] : tensor<?xindex>, tensor<6xindex> -> tensor<?xindex>
// CHECK:                           %[[CASTED_LHS_6:.*]] = tensor_cast %[[BROADCASTED_LHS_6]] : tensor<?xindex> to tensor<6xindex>
// CHECK:                           %[[BROADCASTED_RHS_6:.*]] = shape.broadcast %[[RHS_SHAPE]], %[[CONST_SHAPE_6]] : tensor<?xindex>, tensor<6xindex> -> tensor<?xindex>
// CHECK:                           %[[CASTED_RHS_6:.*]] = tensor_cast %[[BROADCASTED_RHS_6]] : tensor<?xindex> to tensor<6xindex>
// CHECK:                           %[[RESHAPED_LHS_6:.*]] = "mhlo.dynamic_reshape"(%[[LHS]], %[[CASTED_LHS_6]]) : (tensor<*xf32>, tensor<6xindex>) -> tensor<?x?x?x?x?x?xf32>
// CHECK:                           %[[RESHAPED_RHS_6:.*]] = "mhlo.dynamic_reshape"(%[[RHS]], %[[CASTED_RHS_6]]) : (tensor<*xf32>, tensor<6xindex>) -> tensor<?x?x?x?x?x?xf32>
// CHECK:                           %[[RESULT_RANK_6:.*]] = chlo.broadcast_add %[[RESHAPED_LHS_6]], %[[RESHAPED_RHS_6]] : (tensor<?x?x?x?x?x?xf32>, tensor<?x?x?x?x?x?xf32>) -> tensor<?x?x?x?x?x?xf32>
// CHECK:                           %[[RESULT_6:.*]] = tensor_cast %[[RESULT_RANK_6]] : tensor<?x?x?x?x?x?xf32> to tensor<*xf32>
// CHECK:                           scf.yield %[[RESULT_6]] : tensor<*xf32>
// CHECK:                         } else {
// CHECK:                           %false = constant false
// CHECK:                           assert %false
// CHECK:                           scf.yield %[[LHS]] : tensor<*xf32>
// CHECK:                         }
// CHECK:                         scf.yield %[[VAL_64:.*]] : tensor<*xf32>
// CHECK:                       }
// CHECK:                       scf.yield %[[VAL_65:.*]] : tensor<*xf32>
// CHECK:                     }
// CHECK:                     scf.yield %[[VAL_66:.*]] : tensor<*xf32>
// CHECK:                   }
// CHECK:                   scf.yield %[[VAL_67:.*]] : tensor<*xf32>
// CHECK:                 }
// CHECK:                 scf.yield %[[VAL_68:.*]] : tensor<*xf32>
// CHECK:               }
// CHECK:               scf.yield %[[VAL_69:.*]] : tensor<*xf32>
// CHECK:             }
// CHECK:             scf.yield %[[VAL_70:.*]] : tensor<*xf32>
// CHECK:           }
// CHECK:           return %[[VAL_71:.*]] : tensor<*xf32>
// CHECK:         }
