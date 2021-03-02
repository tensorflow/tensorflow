// RUN: mlir-hlo-opt --mhlo-transform-unranked-hlo --cse --split-input-file %s | FileCheck %s

// Check the validity of expected IR.
// CHECK-LABEL: @sqr_transform_result
func @sqr_transform_result(%a: tensor<*xf32>) -> tensor<*xf32> {

  // Flatten operand shape.
  %shape = shape.shape_of %a : tensor<*xf32> -> tensor<?xindex>
  %num_elements = shape.num_elements %shape : tensor<?xindex> -> index
  %flat_shape = tensor.from_elements %num_elements : tensor<1xindex>
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
  // CHECK-NEXT: %[[FLAT_SHAPE:.*]] = tensor.from_elements %[[NUM_ELEMENTS]] : tensor<1xindex>
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
  // CHECK: %[[FLAT_SHAPE:.*]] = tensor.from_elements %[[NUM_ELEMENTS]] : tensor<1xindex>
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
  // CHECK: %[[FLAT_SHAPE:.*]] = tensor.from_elements %[[NUM_ELEMENTS]] : tensor<1xindex>
  // CHECK: %[[FLAT_A:.*]] = "mhlo.dynamic_reshape"(%[[A]], %[[FLAT_SHAPE]]) : (tensor<*xf32>, tensor<1xindex>) -> tensor<?xf32>
  // CHECK: %[[FLAT_B:.*]] = chlo.tan %[[FLAT_A]] : tensor<?xf32> -> tensor<?xf32>
  // CHECK: %[[B:.*]] = "mhlo.dynamic_reshape"(%[[FLAT_B]], %[[SHAPE]]) : (tensor<?xf32>, tensor<?xindex>) -> tensor<*xf32>
  // CHECK: return %[[B]] : tensor<*xf32>
  %result = chlo.tan %a : tensor<*xf32> -> tensor<*xf32>
  return %result : tensor<*xf32>
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
// CHECK-NEXT:           %[[SHAPE_1:.*]] = shape.shape_of %[[ARG_1]] : tensor<*xf32>
// CHECK-NEXT:           %[[NUM_ELEMENTS:.*]] = shape.num_elements %[[SHAPE_1]] : tensor<?xindex> -> index
// CHECK-NEXT:           %[[SIZE_TENSOR:.*]] = tensor.from_elements %[[NUM_ELEMENTS]] : tensor<1xindex>
// CHECK-NEXT:           %[[RESHAPED:.*]] = "mhlo.dynamic_reshape"(%[[ARG_1]], %[[SIZE_TENSOR]]) : (tensor<*xf32>, tensor<1xindex>) -> tensor<?xf32>
// CHECK-NEXT:           %[[BROADCASTED_RESULT:.*]] = chlo.broadcast_add %[[ARG_0]], %[[RESHAPED]] : (tensor<f32>, tensor<?xf32>) -> tensor<?xf32>
//                  As part of the unranked logic, the result is reshaped back
//                  to an unranked tensor.
// CHECK-NEXT:           %[[RESHAPED_RESULT:.*]] = "mhlo.dynamic_reshape"(%[[BROADCASTED_RESULT:.*]], %[[SHAPE_1]]) : (tensor<?xf32>, tensor<?xindex>) -> tensor<*xf32>
// CHECK-NEXT:           return %[[RESHAPED_RESULT]] : tensor<*xf32>
// CHECK-NEXT:         }

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
// CHECK-NEXT:           %[[SHAPE_0:.*]] = shape.shape_of %[[ARG_0]] : tensor<*xf32>
// CHECK-NEXT:           %[[NUM_ELEMENTS:.*]] = shape.num_elements %[[SHAPE_0]] : tensor<?xindex> -> index
// CHECK-NEXT:           %[[SIZE_TENSOR:.*]] = tensor.from_elements %[[NUM_ELEMENTS]] : tensor<1xindex>
// CHECK-NEXT:           %[[RESHAPED:.*]] = "mhlo.dynamic_reshape"(%[[ARG_0]], %[[SIZE_TENSOR]]) : (tensor<*xf32>, tensor<1xindex>) -> tensor<?xf32>
//                  The assuming region is part of the second stage of lowering
//                  with ranked broadcasting logic.
// CHECK-NEXT:           %[[BROADCASTED_RESULT:.*]] = chlo.broadcast_add %[[RESHAPED]], %[[ARG_1]] : (tensor<?xf32>, tensor<f32>)  -> tensor<?xf32>
//                  As part of the unranked logic, the result is reshaped back
//                  to an unranked tensor.
// CHECK-NEXT:           %[[RESHAPED_RESULT:.*]] = "mhlo.dynamic_reshape"(%[[BROADCASTED_RESULT:.*]], %[[SHAPE_0]]) : (tensor<?xf32>, tensor<?xindex>) -> tensor<*xf32>
// CHECK-NEXT:           return %[[RESHAPED_RESULT]] : tensor<*xf32>
// CHECK-NEXT:         }

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
// CHECK-NEXT:           %[[LHS_SHAPE:.*]] = shape.shape_of %[[LHS]] : tensor<*xf32> -> tensor<?xindex>
// CHECK-NEXT:           %[[RHS_SHAPE:.*]] = shape.shape_of %[[RHS]] : tensor<*xf32> -> tensor<?xindex>
// CHECK-NEXT:           %[[NUM_LHS:.*]] = shape.num_elements %[[LHS_SHAPE]] : tensor<?xindex> -> index
// CHECK-NEXT:           %[[C1:.*]] = constant 1 : index
// CHECK-NEXT:           %[[LHS_IS_SCALAR:.*]] = cmpi eq, %[[NUM_LHS]], %[[C1]] : index
//                       Handle scalar LHS case
// CHECK-NEXT:           %[[VAL_8:.*]] = scf.if %[[LHS_IS_SCALAR]] -> (tensor<*xf32>) {
// CHECK-NEXT:             %[[SCALAR_LHS:.*]] = "mhlo.reshape"(%[[LHS]]) : (tensor<*xf32>) -> tensor<f32>
// CHECK-NEXT:             %[[NUM_RHS:.*]] = shape.num_elements %[[RHS_SHAPE]] : tensor<?xindex> -> index
// CHECK-NEXT:             %[[NUM_TENS_RHS:.*]] = tensor.from_elements %[[NUM_RHS]] : tensor<1xindex>
// CHECK-NEXT:             %[[RESHAPED_RHS:.*]] = "mhlo.dynamic_reshape"(%[[RHS]], %[[NUM_TENS_RHS]]) : (tensor<*xf32>, tensor<1xindex>) -> tensor<?xf32>
// CHECK-NEXT:             %[[LHS_SCALAR_RESULT:.*]] = chlo.broadcast_add %[[SCALAR_LHS]], %[[RESHAPED_RHS]] : (tensor<f32>, tensor<?xf32>) -> tensor<?xf32>
// CHECK-NEXT:             %[[RESHAPED_LHS_SCALAR_RESULT:.*]] = "mhlo.dynamic_reshape"(%[[LHS_SCALAR_RESULT]], %[[RHS_SHAPE]]) : (tensor<?xf32>, tensor<?xindex>) -> tensor<*xf32>
// CHECK-NEXT:             %[[SHAPE_BROADCAST_LHS:.*]] = shape.broadcast %[[LHS_SHAPE]], %[[RHS_SHAPE]] : tensor<?xindex>, tensor<?xindex> -> tensor<?xindex>
// CHECK-NEXT:             %[[RESHAPED_EXTENDED_LHS_RESULT:.*]] = "mhlo.dynamic_reshape"(%[[RESHAPED_LHS_SCALAR_RESULT]], %[[SHAPE_BROADCAST_LHS]]) : (tensor<*xf32>, tensor<?xindex>) -> tensor<*xf32>
// CHECK-NEXT:             scf.yield %[[RESHAPED_EXTENDED_LHS_RESULT]] : tensor<*xf32>
// CHECK-NEXT:           } else {
// CHECK-NEXT:             %[[NUM_RHS:.*]] = shape.num_elements %[[RHS_SHAPE]] : tensor<?xindex> -> index
// CHECK-NEXT:             %[[RHS_IS_SCALAR:.*]] = cmpi eq, %[[NUM_RHS]], %[[C1]] : index
//                         Handle scalar RHS case
// CHECK-NEXT:             %[[VAL_14:.*]] = scf.if %[[RHS_IS_SCALAR]] -> (tensor<*xf32>) {
// CHECK-NEXT:               %[[SCALAR_RHS:.*]] = "mhlo.reshape"(%[[RHS]]) : (tensor<*xf32>) -> tensor<f32>
// CHECK-NEXT:               %[[NUM_TENS_LHS:.*]] = tensor.from_elements %[[NUM_LHS]] : tensor<1xindex>
// CHECK-NEXT:               %[[RESHAPED_LHS:.*]] = "mhlo.dynamic_reshape"(%[[LHS]], %[[NUM_TENS_LHS]]) : (tensor<*xf32>, tensor<1xindex>) -> tensor<?xf32>
// CHECK-NEXT:               %[[RHS_SCALAR_RESULT:.*]] = chlo.broadcast_add %[[RESHAPED_LHS]], %[[SCALAR_RHS]] : (tensor<?xf32>, tensor<f32>) -> tensor<?xf32>
// CHECK-NEXT:               %[[RESHAPED_RHS_SCALAR_RESULT:.*]] = "mhlo.dynamic_reshape"(%[[RHS_SCALAR_RESULT:.*]], %[[LHS_SHAPE]]) : (tensor<?xf32>, tensor<?xindex>) -> tensor<*xf32>
// CHECK-NEXT:               %[[SHAPE_BROADCAST_RHS:.*]] = shape.broadcast %[[LHS_SHAPE]], %[[RHS_SHAPE]] : tensor<?xindex>, tensor<?xindex> -> tensor<?xindex>
// CHECK-NEXT:               %[[RESHAPED_EXTENDED_RHS_RESULT:.*]] = "mhlo.dynamic_reshape"(%[[RESHAPED_RHS_SCALAR_RESULT]], %[[SHAPE_BROADCAST_RHS]]) : (tensor<*xf32>, tensor<?xindex>) -> tensor<*xf32>
// CHECK-NEXT:               scf.yield %[[RESHAPED_EXTENDED_RHS_RESULT]] : tensor<*xf32>
// CHECK-NEXT:             } else {
// CHECK-NEXT:               %[[SHAPES_EQ:.*]] = shape.shape_eq %[[LHS_SHAPE]], %[[RHS_SHAPE]] : tensor<?xindex>, tensor<?xindex>
//                           Handle equal shapes case
// CHECK-NEXT:               %[[VAL_18:.*]] = scf.if %[[SHAPES_EQ]] -> (tensor<*xf32>) {
// CHECK-NEXT:                 %[[ANY_SHAPE:.*]] = shape.any %[[LHS_SHAPE]], %[[RHS_SHAPE]] : tensor<?xindex>, tensor<?xindex> -> tensor<?xindex>
// CHECK-NEXT:                 %[[ANY_NUM:.*]] = shape.num_elements %[[ANY_SHAPE]] : tensor<?xindex> -> index
// CHECK-NEXT:                 %[[ANY_TENSOR:.*]] = tensor.from_elements %[[ANY_NUM]] : tensor<1xindex>
// CHECK-NEXT:                 %[[FLATTENED_LHS:.*]] = "mhlo.dynamic_reshape"(%[[LHS]], %[[ANY_TENSOR]]) : (tensor<*xf32>, tensor<1xindex>) -> tensor<?xf32>
// CHECK-NEXT:                 %[[FLATTENED_RHS:.*]] = "mhlo.dynamic_reshape"(%[[RHS]], %[[ANY_TENSOR]]) : (tensor<*xf32>, tensor<1xindex>) -> tensor<?xf32>
// CHECK-NEXT:                 %[[FLATTENED_RESULT:.*]] = mhlo.add %[[FLATTENED_LHS]], %[[FLATTENED_RHS]] : tensor<?xf32>
// CHECK-NEXT:                 %[[RESHAPED_SAME_RESULT:.*]] = "mhlo.dynamic_reshape"(%[[FLATTENED_RESULT]], %[[ANY_SHAPE]]) : (tensor<?xf32>, tensor<?xindex>) -> tensor<*xf32>
// CHECK-NEXT:                 scf.yield %[[RESHAPED_SAME_RESULT]] : tensor<*xf32>
// CHECK-NEXT:               } else {
// CHECK-NEXT:                 %[[RESULT_SHAPE:.*]] = shape.broadcast %[[LHS_SHAPE]], %[[RHS_SHAPE]] : tensor<?xindex>, tensor<?xindex> -> tensor<?xindex>
// CHECK-NEXT:                 %[[MINIMUM_SHAPES:.*]]:2 = chlo.minimum_broadcast_shapes %[[LHS_SHAPE]], %[[RHS_SHAPE]] : tensor<?xindex>, tensor<?xindex> -> tensor<?xindex>, tensor<?xindex>
// CHECK-NEXT:                 %[[MINIMUM_RESHAPED_LHS:.*]] = "mhlo.dynamic_reshape"(%[[LHS]], %[[MINIMUM_SHAPES]]#0) : (tensor<*xf32>, tensor<?xindex>) -> tensor<*xf32>
// CHECK-NEXT:                 %[[MINIMUM_RESHAPED_RHS:.*]] = "mhlo.dynamic_reshape"(%[[RHS]], %[[MINIMUM_SHAPES]]#1) : (tensor<*xf32>, tensor<?xindex>) -> tensor<*xf32>
// CHECK-NEXT:                 %[[LHS_RANK:.*]] = shape.rank %[[MINIMUM_SHAPES]]#0 : tensor<?xindex> -> index
// CHECK-NEXT:                 %[[RHS_RANK:.*]] = shape.rank %[[MINIMUM_SHAPES]]#1 : tensor<?xindex> -> index
// CHECK-NEXT:                 %[[LHS_RANK_GREATER:.*]] = cmpi sgt, %[[LHS_RANK]], %[[RHS_RANK]] : index
// CHECK-NEXT:                 %[[GREATEST_RANK:.*]] = select %[[LHS_RANK_GREATER]], %[[LHS_RANK]], %[[RHS_RANK]] : index
//                             Handle rank 1 specialization
// CHECK-NEXT:                 %[[GREATEST_RANK_IS_1:.*]] = cmpi eq, %[[GREATEST_RANK]], %[[C1]] : index
// CHECK-NEXT:                 %[[RESULT_RANK_SPECIALIZATION:.*]] = scf.if %[[GREATEST_RANK_IS_1]] -> (tensor<*xf32>) {
// CHECK-NEXT:                   %[[CONST_SHAPE_1:.*]] = shape.const_shape [1]
// CHECK-NEXT:                   %[[BROADCASTED_LHS_1:.*]] = shape.broadcast %[[MINIMUM_SHAPES]]#0, %[[CONST_SHAPE_1]] : tensor<?xindex>, tensor<1xindex> -> tensor<?xindex>
// CHECK-NEXT:                   %[[CASTED_LHS_1:.*]] = tensor.cast %[[BROADCASTED_LHS_1]] : tensor<?xindex> to tensor<1xindex>
// CHECK-NEXT:                   %[[RESHAPED_LHS_1:.*]] = "mhlo.dynamic_reshape"(%[[MINIMUM_RESHAPED_LHS]], %[[CASTED_LHS_1]]) : (tensor<*xf32>, tensor<1xindex>) -> tensor<?xf32>
// CHECK-NEXT:                   %[[BROADCASTED_RHS_1:.*]] = shape.broadcast %[[MINIMUM_SHAPES]]#1, %[[CONST_SHAPE_1]] : tensor<?xindex>, tensor<1xindex> -> tensor<?xindex>
// CHECK-NEXT:                   %[[CASTED_RHS_1:.*]] = tensor.cast %[[BROADCASTED_RHS_1]] : tensor<?xindex> to tensor<1xindex>
// CHECK-NEXT:                   %[[RESHAPED_RHS_1:.*]] = "mhlo.dynamic_reshape"(%[[MINIMUM_RESHAPED_RHS]], %[[CASTED_RHS_1]]) : (tensor<*xf32>, tensor<1xindex>) -> tensor<?xf32>
// CHECK-NEXT:                   %[[RESULT_RANK_1:.*]] = chlo.broadcast_add %[[RESHAPED_LHS_1]], %[[RESHAPED_RHS_1]] : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
// CHECK-NEXT:                   %[[RESULT_1:.*]] = tensor.cast %[[RESULT_RANK_1]] : tensor<?xf32> to tensor<*xf32>
// CHECK-NEXT:                   scf.yield %[[RESULT_1]] : tensor<*xf32>
// CHECK-NEXT:                 } else {
// CHECK-NEXT:                   %[[C2:.*]] = constant 2 : index
// CHECK-NEXT:                   %[[GREATEST_RANK_IS_2:.*]] = cmpi eq, %[[GREATEST_RANK]], %[[C2]] : index
//                               Handle rank 2 specialization
// CHECK-NEXT:                   %[[VAL_26:.*]] = scf.if %[[GREATEST_RANK_IS_2]] -> (tensor<*xf32>) {
// CHECK-NEXT:                     %[[CONST_SHAPE_2:.*]] = shape.const_shape [1, 1]
// CHECK-NEXT:                     %[[BROADCASTED_LHS_2:.*]] = shape.broadcast %[[MINIMUM_SHAPES]]#0, %[[CONST_SHAPE_2]] : tensor<?xindex>, tensor<2xindex> -> tensor<?xindex>
// CHECK-NEXT:                     %[[CASTED_LHS_2:.*]] = tensor.cast %[[BROADCASTED_LHS_2]] : tensor<?xindex> to tensor<2xindex>
// CHECK-NEXT:                     %[[RESHAPED_LHS_2:.*]] = "mhlo.dynamic_reshape"(%[[MINIMUM_RESHAPED_LHS]], %[[CASTED_LHS_2]]) : (tensor<*xf32>, tensor<2xindex>) -> tensor<?x?xf32>
// CHECK-NEXT:                     %[[BROADCASTED_RHS_2:.*]] = shape.broadcast %[[MINIMUM_SHAPES]]#1, %[[CONST_SHAPE_2]] : tensor<?xindex>, tensor<2xindex> -> tensor<?xindex>
// CHECK-NEXT:                     %[[CASTED_RHS_2:.*]] = tensor.cast %[[BROADCASTED_RHS_2]] : tensor<?xindex> to tensor<2xindex>
// CHECK-NEXT:                     %[[RESHAPED_RHS_2:.*]] = "mhlo.dynamic_reshape"(%[[MINIMUM_RESHAPED_RHS]], %[[CASTED_RHS_2]]) : (tensor<*xf32>, tensor<2xindex>) -> tensor<?x?xf32>
// CHECK-NEXT:                     %[[RESULT_RANK_2:.*]] = chlo.broadcast_add %[[RESHAPED_LHS_2]], %[[RESHAPED_RHS_2]] : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK-NEXT:                     %[[RESULT_2:.*]] = tensor.cast %[[RESULT_RANK_2]] : tensor<?x?xf32> to tensor<*xf32>
// CHECK-NEXT:                     scf.yield %[[RESULT_2]] : tensor<*xf32>
// CHECK-NEXT:                   } else {
// CHECK-NEXT:                     %[[C3:.*]] = constant 3 : index
// CHECK-NEXT:                     %[[GREATEST_RANK_IS_3:.*]] = cmpi eq, %[[GREATEST_RANK]], %[[C3]] : index
//                                 Handle rank 3 specialization
// CHECK-NEXT:                     %[[VAL_34:.*]] = scf.if %[[GREATEST_RANK_IS_3]] -> (tensor<*xf32>) {
// CHECK-NEXT:                       %[[CONST_SHAPE_3:.*]] = shape.const_shape [1, 1, 1]
// CHECK-NEXT:                       %[[BROADCASTED_LHS_3:.*]] = shape.broadcast %[[MINIMUM_SHAPES]]#0, %[[CONST_SHAPE_3]] : tensor<?xindex>, tensor<3xindex> -> tensor<?xindex>
// CHECK-NEXT:                       %[[CASTED_LHS_3:.*]] = tensor.cast %[[BROADCASTED_LHS_3]] : tensor<?xindex> to tensor<3xindex>
// CHECK-NEXT:                       %[[RESHAPED_LHS_3:.*]] = "mhlo.dynamic_reshape"(%[[MINIMUM_RESHAPED_LHS]], %[[CASTED_LHS_3]]) : (tensor<*xf32>, tensor<3xindex>) -> tensor<?x?x?xf32>
// CHECK-NEXT:                       %[[BROADCASTED_RHS_3:.*]] = shape.broadcast %[[MINIMUM_SHAPES]]#1, %[[CONST_SHAPE_3]] : tensor<?xindex>, tensor<3xindex> -> tensor<?xindex>
// CHECK-NEXT:                       %[[CASTED_RHS_3:.*]] = tensor.cast %[[BROADCASTED_RHS_3]] : tensor<?xindex> to tensor<3xindex>
// CHECK-NEXT:                       %[[RESHAPED_RHS_3:.*]] = "mhlo.dynamic_reshape"(%[[MINIMUM_RESHAPED_RHS]], %[[CASTED_RHS_3]]) : (tensor<*xf32>, tensor<3xindex>) -> tensor<?x?x?xf32>
// CHECK-NEXT:                       %[[RESULT_RANK_3:.*]] = chlo.broadcast_add %[[RESHAPED_LHS_3]], %[[RESHAPED_RHS_3]] : (tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
// CHECK-NEXT:                       %[[RESULT_3:.*]] = tensor.cast %[[RESULT_RANK_3]] : tensor<?x?x?xf32> to tensor<*xf32>
// CHECK-NEXT:                       scf.yield %[[RESULT_3]] : tensor<*xf32>
// CHECK-NEXT:                     } else {
// CHECK-NEXT:                       %[[C4:.*]] = constant 4 : index
// CHECK-NEXT:                       %[[GREATEST_RANK_IS_4:.*]] = cmpi eq, %[[GREATEST_RANK]], %[[C4]] : index
//                                   Handle rank 4 specialization
// CHECK-NEXT:                       %[[VAL_42:.*]] = scf.if %[[GREATEST_RANK_IS_4]] -> (tensor<*xf32>) {
// CHECK-NEXT:                         %[[CONST_SHAPE_4:.*]] = shape.const_shape [1, 1, 1, 1]
// CHECK-NEXT:                         %[[BROADCASTED_LHS_4:.*]] = shape.broadcast %[[MINIMUM_SHAPES]]#0, %[[CONST_SHAPE_4]] : tensor<?xindex>, tensor<4xindex> -> tensor<?xindex>
// CHECK-NEXT:                         %[[CASTED_LHS_4:.*]] = tensor.cast %[[BROADCASTED_LHS_4]] : tensor<?xindex> to tensor<4xindex>
// CHECK-NEXT:                         %[[RESHAPED_LHS_4:.*]] = "mhlo.dynamic_reshape"(%[[MINIMUM_RESHAPED_LHS]], %[[CASTED_LHS_4]]) : (tensor<*xf32>, tensor<4xindex>) -> tensor<?x?x?x?xf32>
// CHECK-NEXT:                         %[[BROADCASTED_RHS_4:.*]] = shape.broadcast %[[MINIMUM_SHAPES]]#1, %[[CONST_SHAPE_4]] : tensor<?xindex>, tensor<4xindex> -> tensor<?xindex>
// CHECK-NEXT:                         %[[CASTED_RHS_4:.*]] = tensor.cast %[[BROADCASTED_RHS_4]] : tensor<?xindex> to tensor<4xindex>
// CHECK-NEXT:                         %[[RESHAPED_RHS_4:.*]] = "mhlo.dynamic_reshape"(%[[MINIMUM_RESHAPED_RHS]], %[[CASTED_RHS_4]]) : (tensor<*xf32>, tensor<4xindex>) -> tensor<?x?x?x?xf32>
// CHECK-NEXT:                         %[[RESULT_RANK_4:.*]] = chlo.broadcast_add %[[RESHAPED_LHS_4]], %[[RESHAPED_RHS_4]] : (tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
// CHECK-NEXT:                         %[[RESULT_4:.*]] = tensor.cast %[[RESULT_RANK_4]] : tensor<?x?x?x?xf32> to tensor<*xf32>
// CHECK-NEXT:                         scf.yield %[[RESULT_4]] : tensor<*xf32>
// CHECK-NEXT:                       } else {
// CHECK-NEXT:                         %[[C5:.*]] = constant 5 : index
// CHECK-NEXT:                         %[[GREATEST_RANK_IS_5:.*]] = cmpi eq, %[[GREATEST_RANK]], %[[C5]] : index
// CHECK-NEXT:                         assert %[[GREATEST_RANK_IS_5]]
//                                     Handle rank 5 specialization
// CHECK-NEXT:                         %[[CONST_SHAPE_5:.*]] = shape.const_shape [1, 1, 1, 1, 1]
// CHECK-NEXT:                         %[[BROADCASTED_LHS_5:.*]] = shape.broadcast %[[MINIMUM_SHAPES]]#0, %[[CONST_SHAPE_5]] : tensor<?xindex>, tensor<5xindex> -> tensor<?xindex>
// CHECK-NEXT:                         %[[CASTED_LHS_5:.*]] = tensor.cast %[[BROADCASTED_LHS_5]] : tensor<?xindex> to tensor<5xindex>
// CHECK-NEXT:                         %[[RESHAPED_LHS_5:.*]] = "mhlo.dynamic_reshape"(%[[MINIMUM_RESHAPED_LHS]], %[[CASTED_LHS_5]]) : (tensor<*xf32>, tensor<5xindex>) -> tensor<?x?x?x?x?xf32>
// CHECK-NEXT:                         %[[BROADCASTED_RHS_5:.*]] = shape.broadcast %[[MINIMUM_SHAPES]]#1, %[[CONST_SHAPE_5]] : tensor<?xindex>, tensor<5xindex> -> tensor<?xindex>
// CHECK-NEXT:                         %[[CASTED_RHS_5:.*]] = tensor.cast %[[BROADCASTED_RHS_5]] : tensor<?xindex> to tensor<5xindex>
// CHECK-NEXT:                         %[[RESHAPED_RHS_5:.*]] = "mhlo.dynamic_reshape"(%[[MINIMUM_RESHAPED_RHS]], %[[CASTED_RHS_5]]) : (tensor<*xf32>, tensor<5xindex>) -> tensor<?x?x?x?x?xf32>
// CHECK-NEXT:                         %[[RESULT_RANK_5:.*]] = chlo.broadcast_add %[[RESHAPED_LHS_5]], %[[RESHAPED_RHS_5]] : (tensor<?x?x?x?x?xf32>, tensor<?x?x?x?x?xf32>) -> tensor<?x?x?x?x?xf32>
// CHECK-NEXT:                         %[[RESULT_5:.*]] = tensor.cast %[[RESULT_RANK_5]] : tensor<?x?x?x?x?xf32> to tensor<*xf32>
// CHECK-NEXT:                         scf.yield %[[RESULT_5]] : tensor<*xf32>
// CHECK-NEXT:                       }
// CHECK-NEXT:                       scf.yield %[[VAL_66:.*]] : tensor<*xf32>
// CHECK-NEXT:                     }
// CHECK-NEXT:                     scf.yield %[[VAL_67:.*]] : tensor<*xf32>
// CHECK-NEXT:                   }
// CHECK-NEXT:                   scf.yield %[[VAL_68:.*]] : tensor<*xf32>
// CHECK-NEXT:                 }
// CHECK-NEXT:                 %[[RESHAPED_RESULT:.*]] = "mhlo.dynamic_reshape"(%[[RESULT_RANK_SPECIALIZATION]], %[[RESULT_SHAPE]]) : (tensor<*xf32>, tensor<?xindex>) -> tensor<*xf32>
// CHECK-NEXT:                 scf.yield %[[RESHAPED_RESULT]] : tensor<*xf32>
// CHECK-NEXT:               }
// CHECK-NEXT:               scf.yield %[[VAL_70:.*]] : tensor<*xf32>
// CHECK-NEXT:             }
// CHECK-NEXT:             scf.yield %[[VAL_71:.*]] : tensor<*xf32>
// CHECK-NEXT:           }
// CHECK-NEXT:           return %[[VAL_72:.*]] : tensor<*xf32>
// CHECK-NEXT:         }


// -----

func @selectUnrankedUnrankedUnranked(
    %arg0: tensor<*xi1>, %arg1: tensor<*xf32>, %arg2: tensor<*xf32>)
    -> tensor<*xf32> {
  %0 = chlo.broadcast_select %arg0, %arg1, %arg2
        : (tensor<*xi1>, tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}

// CHECK-LABEL: func @selectUnrankedUnrankedUnranked(
// CHECK-SAME:     %[[PRED:.*]]: tensor<*xi1>,
// CHECK-SAME:     %[[LHS:.*]]: tensor<*xf32>,
// CHECK-SAME:     %[[RHS:.*]]: tensor<*xf32>) -> tensor<*xf32> {
// CHECK-NEXT:    %[[PRED_SHAPE:.*]] = shape.shape_of %[[PRED]] : tensor<*xi1> -> tensor<?xindex>
// CHECK-NEXT:    %[[LHS_SHAPE:.*]] = shape.shape_of %[[LHS]] : tensor<*xf32> -> tensor<?xindex>
// CHECK-NEXT:    %[[RHS_SHAPE:.*]] = shape.shape_of %[[RHS]] : tensor<*xf32> -> tensor<?xindex>
// CHECK-NEXT:    %[[RESULT_SHAPE:.*]] = shape.broadcast %[[PRED_SHAPE]], %[[LHS_SHAPE]], %[[RHS_SHAPE]] : tensor<?xindex>, tensor<?xindex>, tensor<?xindex> -> tensor<?xindex>
// CHECK-NEXT:    %[[MINIMUM_SHAPES:.*]]:3 = chlo.minimum_broadcast_shapes %[[PRED_SHAPE]], %[[LHS_SHAPE]], %[[RHS_SHAPE]] : tensor<?xindex>, tensor<?xindex>, tensor<?xindex> -> tensor<?xindex>, tensor<?xindex>, tensor<?xindex>
// CHECK-NEXT:    %[[MINIMUM_RESHAPED_PRED:.*]] = "mhlo.dynamic_reshape"(%[[PRED]], %[[MINIMUM_SHAPES]]#0) : (tensor<*xi1>, tensor<?xindex>) -> tensor<*xi1>
// CHECK-NEXT:    %[[MINIMUM_RESHAPED_LHS:.*]] = "mhlo.dynamic_reshape"(%[[LHS]], %[[MINIMUM_SHAPES]]#1) : (tensor<*xf32>, tensor<?xindex>) -> tensor<*xf32>
// CHECK-NEXT:    %[[MINIMUM_RESHAPED_RHS:.*]] = "mhlo.dynamic_reshape"(%[[RHS]], %[[MINIMUM_SHAPES]]#2) : (tensor<*xf32>, tensor<?xindex>) -> tensor<*xf32>
// CHECK-NEXT:    %[[PRED_RANK:.*]] = shape.rank %[[MINIMUM_SHAPES]]#0 : tensor<?xindex> -> index
// CHECK-NEXT:    %[[LHS_RANK:.*]] = shape.rank %[[MINIMUM_SHAPES]]#1 : tensor<?xindex> -> index
// CHECK-NEXT:    %[[GREATER_RANK_CMP:.*]] = cmpi sgt, %[[PRED_RANK]], %[[LHS_RANK]] : index
// CHECK-NEXT:    %[[GREATER_RANK:.*]] = select %[[GREATER_RANK_CMP]], %[[PRED_RANK]], %[[LHS_RANK]] : index
// CHECK-NEXT:    %[[RHS_RANK:.*]] = shape.rank %[[MINIMUM_SHAPES]]#2 : tensor<?xindex> -> index
// CHECK-NEXT:    %[[GREATEST_RANK_CMP:.*]] = cmpi sgt, %[[GREATER_RANK]], %[[RHS_RANK]] : index
// CHECK-NEXT:    %[[GREATEST_RANK:.*]] = select %[[GREATEST_RANK_CMP]], %[[GREATER_RANK]], %[[RHS_RANK]] : index
// CHECK-NEXT:    %c1 = constant 1 : index
// CHECK-NEXT:    %[[GREATEST_RANK_IS_1:.*]] = cmpi eq, %[[GREATEST_RANK]], %c1 : index
//                Handle rank 1 specialization
// CHECK-NEXT:    scf.if %[[GREATEST_RANK_IS_1]] -> (tensor<*xf32>) {
// CHECK-NEXT:      %[[CONST_SHAPE_1:.*]] = shape.const_shape [1] : tensor<1xindex>
// CHECK-NEXT:      %[[BROADCASTED_PRED:.*]] = shape.broadcast %[[MINIMUM_SHAPES]]#0, %[[CONST_SHAPE_1]] : tensor<?xindex>, tensor<1xindex> -> tensor<?xindex>
// CHECK-NEXT:      %[[CASTED_PRED:.*]] = tensor.cast %[[BROADCASTED_PRED]] : tensor<?xindex> to tensor<1xindex>
// CHECK-NEXT:      %[[RESHAPED_PRED:.*]] = "mhlo.dynamic_reshape"(%[[MINIMUM_RESHAPED_PRED]], %[[CASTED_PRED]]) : (tensor<*xi1>, tensor<1xindex>) -> tensor<?xi1>
// CHECK-NEXT:      %[[BROADCASTED_LHS:.*]] = shape.broadcast %[[MINIMUM_SHAPES]]#1, %[[CONST_SHAPE_1]] : tensor<?xindex>, tensor<1xindex> -> tensor<?xindex>
// CHECK-NEXT:      %[[CASTED_LHS:.*]] = tensor.cast %[[BROADCASTED_LHS]] : tensor<?xindex> to tensor<1xindex>
// CHECK-NEXT:      %[[RESHAPED_LHS:.*]] = "mhlo.dynamic_reshape"(%[[MINIMUM_RESHAPED_LHS]], %[[CASTED_LHS]]) : (tensor<*xf32>, tensor<1xindex>) -> tensor<?xf32>
// CHECK-NEXT:      %[[BROADCASTED_RHS:.*]] = shape.broadcast %[[MINIMUM_SHAPES]]#2, %[[CONST_SHAPE_1]] : tensor<?xindex>, tensor<1xindex> -> tensor<?xindex>
// CHECK-NEXT:      %[[CASTED_RHS:.*]] = tensor.cast %[[BROADCASTED_RHS]] : tensor<?xindex> to tensor<1xindex>
// CHECK-NEXT:      %[[RESHAPED_RHS:.*]] = "mhlo.dynamic_reshape"(%[[MINIMUM_RESHAPED_RHS]], %[[CASTED_RHS]]) : (tensor<*xf32>, tensor<1xindex>) -> tensor<?xf32>
// CHECK-NEXT:      %[[RESULT_RANK_1:.*]] = chlo.broadcast_select %[[RESHAPED_PRED]], %[[RESHAPED_LHS]], %[[RESHAPED_RHS]] : (tensor<?xi1>, tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
// CHECK-NEXT:      %[[RESULT_1:.*]] = tensor.cast %[[RESULT_RANK_1:.*]] : tensor<?xf32> to tensor<*xf32>
// CHECK-NEXT:      scf.yield %[[RESULT_1]] : tensor<*xf32>
// CHECK-NEXT:      }

// CHECK:      chlo.broadcast_select {{.*}} : (tensor<?x?xi1>, tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
// CHECK:      chlo.broadcast_select {{.*}} : (tensor<?x?x?xi1>, tensor<?x?x?xf32>, tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
// CHECK:      chlo.broadcast_select {{.*}} : (tensor<?x?x?x?xi1>, tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
// CHECK:      chlo.broadcast_select {{.*}} : (tensor<?x?x?x?x?xi1>, tensor<?x?x?x?x?xf32>, tensor<?x?x?x?x?xf32>) -> tensor<?x?x?x?x?xf32>
