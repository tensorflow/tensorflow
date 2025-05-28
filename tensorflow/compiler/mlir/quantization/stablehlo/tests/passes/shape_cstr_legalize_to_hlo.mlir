// RUN: stablehlo-quant-opt %s -split-input-file -tf-stablehlo-convert-shape-to-stablehlo-with-constraints --verify-diagnostics | FileCheck %s

// CHECK-LABEL: func.func @shape_cstr_broadcastable
func.func @shape_cstr_broadcastable(%arg0: tensor<2xindex>, %arg1: tensor<2xindex>) {
  %0 = shape.cstr_broadcastable %arg0, %arg1 : tensor<2xindex>, tensor<2xindex>
  shape.assuming %0 {
  }
  func.return
  //      CHECK: %[[DIMS1:.*]] = builtin.unrealized_conversion_cast %arg0 : tensor<2xindex> to tensor<2xi32>
  // CHECK-NEXT: %[[DIMS2:.*]] = builtin.unrealized_conversion_cast %arg1 : tensor<2xindex> to tensor<2xi32>
  // CHECK-NEXT: %[[ONES:.*]] = stablehlo.constant dense<1> : tensor<2xi32>
  // CHECK-NEXT: %[[DIMS1_IS_1:.*]] = stablehlo.compare  EQ, %[[DIMS1]], %[[ONES:.*]] : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1>
  // CHECK-NEXT: %[[DIMS2_IS_1:.*]] = stablehlo.compare  EQ, %[[DIMS2]], %[[ONES:.*]] : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1>
  // CHECK-NEXT: %[[EITHER_DIM_IS_1:.*]] = stablehlo.or %[[DIMS1_IS_1]], %[[DIMS2_IS_1]] : tensor<2xi1>
  // CHECK-NEXT: %[[DIMS_EQ:.*]] = stablehlo.compare  EQ, %[[DIMS1]], %[[DIMS2]] : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1>
  // CHECK-NEXT: %[[DIMS_BROADCASTABLE:.*]] = stablehlo.or %[[EITHER_DIM_IS_1]], %[[DIMS_EQ]] : tensor<2xi1>
  // CHECK-NEXT: %[[TRUE:.*]] = stablehlo.constant dense<true> : tensor<1xi1>
  // CHECK-NEXT: %[[DIM1_BROADCASTABLE:.*]] = stablehlo.slice %[[DIMS_BROADCASTABLE]] [0:1] : (tensor<2xi1>) -> tensor<1xi1>
  // CHECK-NEXT: %[[BROADCASTABLE_TEMP:.*]] = stablehlo.and %[[TRUE]], %[[DIM1_BROADCASTABLE]] : tensor<1xi1>
  // CHECK-NEXT: %[[DIM2_BROADCASTABLE:.*]] = stablehlo.slice %[[DIMS_BROADCASTABLE]] [1:2] : (tensor<2xi1>) -> tensor<1xi1>
  // CHECK-NEXT: %[[ALL_BROADCASTABLE:.*]] = stablehlo.and %[[BROADCASTABLE_TEMP]], %[[DIM2_BROADCASTABLE]] : tensor<1xi1>
  // CHECK-NEXT: %[[ALL_BROADCASTABLE_SCALAR:.*]] = stablehlo.reshape %[[ALL_BROADCASTABLE]] : (tensor<1xi1>) -> tensor<i1>
  // CHECK-NEXT: stablehlo.custom_call @shape_assertion(%[[ALL_BROADCASTABLE_SCALAR]]) {error_message = "Shape assertion failed", has_side_effect = true} : (tensor<i1>) -> ()
  // CHECK-NEXT: %[[WITNESS:.*]] = shape.const_witness true
  // CHECK-NEXT: shape.assuming %[[WITNESS]] {
  // CHECK-NEXT: }
  // CHECK-NEXT: return
}

// -----

// CHECK-LABEL: func @shape_cstr_broadcastable_different_dims_1
func.func @shape_cstr_broadcastable_different_dims_1(%arg0: tensor<2xindex>, %arg1: tensor<1xindex>) {
  %0 = shape.cstr_broadcastable %arg0, %arg1 : tensor<2xindex>, tensor<1xindex>
  shape.assuming %0 {
  }
  func.return
  //      CHECK: %[[DIMS1:.*]] = builtin.unrealized_conversion_cast %arg0 : tensor<2xindex> to tensor<2xi32>
  // CHECK-NEXT: %[[DIMS2:.*]] = builtin.unrealized_conversion_cast %arg1 : tensor<1xindex> to tensor<1xi32>
  // CHECK-NEXT: %[[PAD:.*]] = stablehlo.constant dense<1> : tensor<1xi32>
  // CHECK-NEXT: %[[DIMS2_PAD:.*]] = stablehlo.concatenate %[[PAD]], %[[DIMS2]], dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
  // CHECK-NEXT: %[[ONES:.*]] = stablehlo.constant dense<1> : tensor<2xi32>
  // CHECK-NEXT: %[[DIMS1_IS_1:.*]] = stablehlo.compare  EQ, %[[DIMS1]], %[[ONES:.*]] : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1>
  // CHECK-NEXT: %[[DIMS2_IS_1:.*]] = stablehlo.compare  EQ, %[[DIMS2_PAD]], %[[ONES:.*]] : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1>
  // CHECK-NEXT: %[[EITHER_DIM_IS_1:.*]] = stablehlo.or %[[DIMS1_IS_1]], %[[DIMS2_IS_1]] : tensor<2xi1>
  // CHECK-NEXT: %[[DIMS_EQ:.*]] = stablehlo.compare  EQ, %[[DIMS1]], %[[DIMS2_PAD]] : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1>
  // CHECK-NEXT: %[[DIMS_BROADCASTABLE:.*]] = stablehlo.or %[[EITHER_DIM_IS_1]], %[[DIMS_EQ]] : tensor<2xi1>
  // CHECK-NEXT: %[[TRUE:.*]] = stablehlo.constant dense<true> : tensor<1xi1>
  // CHECK-NEXT: %[[DIM1_BROADCASTABLE:.*]] = stablehlo.slice %[[DIMS_BROADCASTABLE]] [0:1] : (tensor<2xi1>) -> tensor<1xi1>
  // CHECK-NEXT: %[[BROADCASTABLE_TEMP:.*]] = stablehlo.and %[[TRUE]], %[[DIM1_BROADCASTABLE]] : tensor<1xi1>
  // CHECK-NEXT: %[[DIM2_BROADCASTABLE:.*]] = stablehlo.slice %[[DIMS_BROADCASTABLE]] [1:2] : (tensor<2xi1>) -> tensor<1xi1>
  // CHECK-NEXT: %[[ALL_BROADCASTABLE:.*]] = stablehlo.and %[[BROADCASTABLE_TEMP]], %[[DIM2_BROADCASTABLE]] : tensor<1xi1>
  // CHECK-NEXT: %[[ALL_BROADCASTABLE_SCALAR:.*]] = stablehlo.reshape %[[ALL_BROADCASTABLE]] : (tensor<1xi1>) -> tensor<i1>
  // CHECK-NEXT: stablehlo.custom_call @shape_assertion(%[[ALL_BROADCASTABLE_SCALAR]]) {error_message = "Shape assertion failed", has_side_effect = true} : (tensor<i1>) -> ()
  // CHECK-NEXT: %[[WITNESS:.*]] = shape.const_witness true
  // CHECK-NEXT: shape.assuming %[[WITNESS]] {
  // CHECK-NEXT: }
  // CHECK-NEXT: return
}

// -----

// CHECK-LABEL: func @shape_cstr_broadcastable_different_dims_2
func.func @shape_cstr_broadcastable_different_dims_2(%arg0: tensor<1xindex>, %arg1: tensor<2xindex>) {
  %0 = shape.cstr_broadcastable %arg0, %arg1 : tensor<1xindex>, tensor<2xindex>
  shape.assuming %0 {
  }
  func.return
  //      CHECK: %[[DIMS1:.*]] = builtin.unrealized_conversion_cast %arg0 : tensor<1xindex> to tensor<1xi32>
  // CHECK-NEXT: %[[DIMS2:.*]] = builtin.unrealized_conversion_cast %arg1 : tensor<2xindex> to tensor<2xi32>
  // CHECK-NEXT: %[[PAD:.*]] = stablehlo.constant dense<1> : tensor<1xi32>
  // CHECK-NEXT: %[[DIMS1_PAD:.*]] = stablehlo.concatenate %[[PAD]], %[[DIMS1]], dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
  // CHECK-NEXT: %[[ONES:.*]] = stablehlo.constant dense<1> : tensor<2xi32>
  // CHECK-NEXT: %[[DIMS1_IS_1:.*]] = stablehlo.compare  EQ, %[[DIMS1_PAD]], %[[ONES:.*]] : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1>
  // CHECK-NEXT: %[[DIMS2_IS_1:.*]] = stablehlo.compare  EQ, %[[DIMS2]], %[[ONES:.*]] : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1>
  // CHECK-NEXT: %[[EITHER_DIM_IS_1:.*]] = stablehlo.or %[[DIMS1_IS_1]], %[[DIMS2_IS_1]] : tensor<2xi1>
  // CHECK-NEXT: %[[DIMS_EQ:.*]] = stablehlo.compare  EQ, %[[DIMS1_PAD]], %[[DIMS2]] : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1>
  // CHECK-NEXT: %[[DIMS_BROADCASTABLE:.*]] = stablehlo.or %[[EITHER_DIM_IS_1]], %[[DIMS_EQ]] : tensor<2xi1>
  // CHECK-NEXT: %[[TRUE:.*]] = stablehlo.constant dense<true> : tensor<1xi1>
  // CHECK-NEXT: %[[DIM1_BROADCASTABLE:.*]] = stablehlo.slice %[[DIMS_BROADCASTABLE]] [0:1] : (tensor<2xi1>) -> tensor<1xi1>
  // CHECK-NEXT: %[[BROADCASTABLE_TEMP:.*]] = stablehlo.and %[[TRUE]], %[[DIM1_BROADCASTABLE]] : tensor<1xi1>
  // CHECK-NEXT: %[[DIM2_BROADCASTABLE:.*]] = stablehlo.slice %[[DIMS_BROADCASTABLE]] [1:2] : (tensor<2xi1>) -> tensor<1xi1>
  // CHECK-NEXT: %[[ALL_BROADCASTABLE:.*]] = stablehlo.and %[[BROADCASTABLE_TEMP]], %[[DIM2_BROADCASTABLE]] : tensor<1xi1>
  // CHECK-NEXT: %[[ALL_BROADCASTABLE_SCALAR:.*]] = stablehlo.reshape %[[ALL_BROADCASTABLE]] : (tensor<1xi1>) -> tensor<i1>
  // CHECK-NEXT: stablehlo.custom_call @shape_assertion(%[[ALL_BROADCASTABLE_SCALAR]]) {error_message = "Shape assertion failed", has_side_effect = true} : (tensor<i1>) -> ()
  // CHECK-NEXT: %[[WITNESS:.*]] = shape.const_witness true
  // CHECK-NEXT: shape.assuming %[[WITNESS]] {
  // CHECK-NEXT: }
  // CHECK-NEXT: return
}

// -----

func.func @shape_cstr_broadcast_too_many_operands(%arg0: tensor<4xindex>, %arg1: tensor<4xindex>, %arg2: tensor<4xindex>) {
  // expected-error@+1 {{failed to legalize operation 'shape.cstr_broadcastable' that was explicitly marked illegal}}
  %0 = shape.cstr_broadcastable %arg0, %arg1, %arg2 : tensor<4xindex>, tensor<4xindex>, tensor<4xindex>
  shape.assuming %0 {
  }
  func.return
}

// -----

func.func @shape_cstr_broadcastable_input_shape(%arg0: !shape.shape, %arg1: !shape.shape) {
  // expected-error@+1 {{failed to legalize operation 'shape.cstr_broadcastable' that was explicitly marked illegal}}
  %0 = shape.cstr_broadcastable %arg0, %arg1 : !shape.shape, !shape.shape
  shape.assuming %0 {
  }
  func.return
}
