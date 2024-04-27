// RUN: mlir-hlo-opt --shape-legalize-to-hlo=legalize-constraints=true --split-input-file --verify-diagnostics %s | FileCheck %s

// -----

// CHECK-LABEL: func.func @shape_cstr_broadcastable
func.func @shape_cstr_broadcastable(%arg0: tensor<2xindex>, %arg1: tensor<2xindex>) {
  %0 = shape.cstr_broadcastable %arg0, %arg1 : tensor<2xindex>, tensor<2xindex>
  shape.assuming %0 {
  }
  func.return
  //      CHECK: %[[DIMS1:.*]] = builtin.unrealized_conversion_cast %arg0 : tensor<2xindex> to tensor<2xi32>
  // CHECK-NEXT: %[[DIMS2:.*]] = builtin.unrealized_conversion_cast %arg1 : tensor<2xindex> to tensor<2xi32>
  // CHECK-NEXT: %[[ONES:.*]] = mhlo.constant dense<1> : tensor<2xi32>
  // CHECK-NEXT: %[[DIMS1_IS_1:.*]] = mhlo.compare  EQ, %[[DIMS1]], %[[ONES:.*]],  NOTYPE : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1>
  // CHECK-NEXT: %[[DIMS2_IS_1:.*]] = mhlo.compare  EQ, %[[DIMS2]], %[[ONES:.*]],  NOTYPE : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1>
  // CHECK-NEXT: %[[EITHER_DIM_IS_1:.*]] = mhlo.or %[[DIMS1_IS_1]], %[[DIMS2_IS_1]] : tensor<2xi1>
  // CHECK-NEXT: %[[DIMS_EQ:.*]] = mhlo.compare  EQ, %[[DIMS1]], %[[DIMS2]],  NOTYPE : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1>
  // CHECK-NEXT: %[[DIMS_BROADCASTABLE:.*]] = mhlo.or %[[EITHER_DIM_IS_1]], %[[DIMS_EQ]] : tensor<2xi1>
  // CHECK-NEXT: %[[TRUE:.*]] = mhlo.constant dense<true> : tensor<1xi1>
  // CHECK-NEXT: %[[DIM1_BROADCASTABLE:.*]] = "mhlo.slice"(%[[DIMS_BROADCASTABLE]]) <{limit_indices = dense<1> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>}> : (tensor<2xi1>) -> tensor<1xi1>
  // CHECK-NEXT: %[[BROADCASTABLE_TEMP:.*]] = mhlo.and %[[TRUE]], %[[DIM1_BROADCASTABLE]] : tensor<1xi1>
  // CHECK-NEXT: %[[DIM2_BROADCASTABLE:.*]] = "mhlo.slice"(%[[DIMS_BROADCASTABLE]]) <{limit_indices = dense<2> : tensor<1xi64>, start_indices = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>}> : (tensor<2xi1>) -> tensor<1xi1>
  // CHECK-NEXT: %[[ALL_BROADCASTABLE:.*]] = mhlo.and %[[BROADCASTABLE_TEMP]], %[[DIM2_BROADCASTABLE]] : tensor<1xi1>
  // CHECK-NEXT: %[[ALL_BROADCASTABLE_SCALAR:.*]] = mhlo.reshape %[[ALL_BROADCASTABLE]] : (tensor<1xi1>) -> tensor<i1>
  // CHECK-NEXT: mhlo.custom_call @shape_assertion(%[[ALL_BROADCASTABLE_SCALAR]]) {error_message = "Shape assertion failed", has_side_effect = true} : (tensor<i1>) -> ()
  // CHECK-NEXT: %[[WITNESS:.*]] = shape.const_witness true
  // CHECK-NEXT: shape.assuming %[[WITNESS]] {
  // CHECK-NEXT: }
  // CHECK-NEXT: return
}

// -----

func.func @shape_cstr_broadcastable_input_shape(%arg0: !shape.shape, %arg1: !shape.shape) {
  // expected-error@+1 {{failed to legalize operation 'shape.cstr_broadcastable' that was explicitly marked illegal}}
  %0 = shape.cstr_broadcastable %arg0, %arg1 : !shape.shape, !shape.shape
  shape.assuming %0 {
  }
  func.return
}

// -----

func.func @shape_cstr_broadcastable_different_dims_1(%arg0: tensor<2xindex>, %arg1: tensor<1xindex>) {
  %0 = shape.cstr_broadcastable %arg0, %arg1 : tensor<2xindex>, tensor<1xindex>
  shape.assuming %0 {
  }
  func.return
  //      CHECK: %[[DIMS1:.*]] = builtin.unrealized_conversion_cast %arg0 : tensor<2xindex> to tensor<2xi32>
  // CHECK-NEXT: %[[DIMS2:.*]] = builtin.unrealized_conversion_cast %arg1 : tensor<1xindex> to tensor<1xi32>
  // CHECK-NEXT: %[[PAD:.*]] = mhlo.constant dense<1> : tensor<1xi32>
  // CHECK-NEXT: %[[DIMS2_PAD:.*]] = "mhlo.concatenate"(%[[PAD]], %[[DIMS2]]) <{dimension = 0 : i64}> : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
  // CHECK-NEXT: %[[ONES:.*]] = mhlo.constant dense<1> : tensor<2xi32>
  // CHECK-NEXT: %[[DIMS1_IS_1:.*]] = mhlo.compare  EQ, %[[DIMS1]], %[[ONES:.*]],  NOTYPE : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1>
  // CHECK-NEXT: %[[DIMS2_IS_1:.*]] = mhlo.compare  EQ, %[[DIMS2_PAD]], %[[ONES:.*]],  NOTYPE : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1>
  // CHECK-NEXT: %[[EITHER_DIM_IS_1:.*]] = mhlo.or %[[DIMS1_IS_1]], %[[DIMS2_IS_1]] : tensor<2xi1>
  // CHECK-NEXT: %[[DIMS_EQ:.*]] = mhlo.compare  EQ, %[[DIMS1]], %[[DIMS2_PAD]],  NOTYPE : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1>
  // CHECK-NEXT: %[[DIMS_BROADCASTABLE:.*]] = mhlo.or %[[EITHER_DIM_IS_1]], %[[DIMS_EQ]] : tensor<2xi1>
  // CHECK-NEXT: %[[TRUE:.*]] = mhlo.constant dense<true> : tensor<1xi1>
  // CHECK-NEXT: %[[DIM1_BROADCASTABLE:.*]] = "mhlo.slice"(%[[DIMS_BROADCASTABLE]]) <{limit_indices = dense<1> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>}> : (tensor<2xi1>) -> tensor<1xi1>
  // CHECK-NEXT: %[[BROADCASTABLE_TEMP:.*]] = mhlo.and %[[TRUE]], %[[DIM1_BROADCASTABLE]] : tensor<1xi1>
  // CHECK-NEXT: %[[DIM2_BROADCASTABLE:.*]] = "mhlo.slice"(%[[DIMS_BROADCASTABLE]]) <{limit_indices = dense<2> : tensor<1xi64>, start_indices = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>}> : (tensor<2xi1>) -> tensor<1xi1>
  // CHECK-NEXT: %[[ALL_BROADCASTABLE:.*]] = mhlo.and %[[BROADCASTABLE_TEMP]], %[[DIM2_BROADCASTABLE]] : tensor<1xi1>
  // CHECK-NEXT: %[[ALL_BROADCASTABLE_SCALAR:.*]] = mhlo.reshape %[[ALL_BROADCASTABLE]] : (tensor<1xi1>) -> tensor<i1>
  // CHECK-NEXT: mhlo.custom_call @shape_assertion(%[[ALL_BROADCASTABLE_SCALAR]]) {error_message = "Shape assertion failed", has_side_effect = true} : (tensor<i1>) -> ()
  // CHECK-NEXT: %[[WITNESS:.*]] = shape.const_witness true
  // CHECK-NEXT: shape.assuming %[[WITNESS]] {
  // CHECK-NEXT: }
  // CHECK-NEXT: return
}

// -----

func.func @shape_cstr_broadcastable_different_dims_2(%arg0: tensor<1xindex>, %arg1: tensor<2xindex>) {
  %0 = shape.cstr_broadcastable %arg0, %arg1 : tensor<1xindex>, tensor<2xindex>
  shape.assuming %0 {
  }
  func.return
  //      CHECK: %[[DIMS1:.*]] = builtin.unrealized_conversion_cast %arg0 : tensor<1xindex> to tensor<1xi32>
  // CHECK-NEXT: %[[DIMS2:.*]] = builtin.unrealized_conversion_cast %arg1 : tensor<2xindex> to tensor<2xi32>
  // CHECK-NEXT: %[[PAD:.*]] = mhlo.constant dense<1> : tensor<1xi32>
  // CHECK-NEXT: %[[DIMS1_PAD:.*]] = "mhlo.concatenate"(%[[PAD]], %[[DIMS1]]) <{dimension = 0 : i64}> : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
  // CHECK-NEXT: %[[ONES:.*]] = mhlo.constant dense<1> : tensor<2xi32>
  // CHECK-NEXT: %[[DIMS1_IS_1:.*]] = mhlo.compare  EQ, %[[DIMS1_PAD]], %[[ONES:.*]],  NOTYPE : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1>
  // CHECK-NEXT: %[[DIMS2_IS_1:.*]] = mhlo.compare  EQ, %[[DIMS2]], %[[ONES:.*]],  NOTYPE : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1>
  // CHECK-NEXT: %[[EITHER_DIM_IS_1:.*]] = mhlo.or %[[DIMS1_IS_1]], %[[DIMS2_IS_1]] : tensor<2xi1>
  // CHECK-NEXT: %[[DIMS_EQ:.*]] = mhlo.compare  EQ, %[[DIMS1_PAD]], %[[DIMS2]],  NOTYPE : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1>
  // CHECK-NEXT: %[[DIMS_BROADCASTABLE:.*]] = mhlo.or %[[EITHER_DIM_IS_1]], %[[DIMS_EQ]] : tensor<2xi1>
  // CHECK-NEXT: %[[TRUE:.*]] = mhlo.constant dense<true> : tensor<1xi1>
  // CHECK-NEXT: %[[DIM1_BROADCASTABLE:.*]] = "mhlo.slice"(%[[DIMS_BROADCASTABLE]]) <{limit_indices = dense<1> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>}> : (tensor<2xi1>) -> tensor<1xi1>
  // CHECK-NEXT: %[[BROADCASTABLE_TEMP:.*]] = mhlo.and %[[TRUE]], %[[DIM1_BROADCASTABLE]] : tensor<1xi1>
  // CHECK-NEXT: %[[DIM2_BROADCASTABLE:.*]] = "mhlo.slice"(%[[DIMS_BROADCASTABLE]]) <{limit_indices = dense<2> : tensor<1xi64>, start_indices = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>}> : (tensor<2xi1>) -> tensor<1xi1>
  // CHECK-NEXT: %[[ALL_BROADCASTABLE:.*]] = mhlo.and %[[BROADCASTABLE_TEMP]], %[[DIM2_BROADCASTABLE]] : tensor<1xi1>
  // CHECK-NEXT: %[[ALL_BROADCASTABLE_SCALAR:.*]] = mhlo.reshape %[[ALL_BROADCASTABLE]] : (tensor<1xi1>) -> tensor<i1>
  // CHECK-NEXT: mhlo.custom_call @shape_assertion(%[[ALL_BROADCASTABLE_SCALAR]]) {error_message = "Shape assertion failed", has_side_effect = true} : (tensor<i1>) -> ()
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

func.func @mhlo_cstr_reshapable(%arg0: index, %arg1: tensor<2xindex>) {
  %0 = mhlo.cstr_reshapable %arg0, %arg1 : (index, tensor<2xindex>) -> !shape.witness
  func.return
  //  CHECK-DAG: %[[NUM_ELEMENTS:.*]] = builtin.unrealized_conversion_cast %arg0 : index to tensor<i32>
  //  CHECK-DAG: %[[DYNAMIC_SHAPE:.*]] = builtin.unrealized_conversion_cast %arg1 : tensor<2xindex> to tensor<2xi32>
  //  CHECK-DAG: %[[MINUS_ONE:.*]] = mhlo.constant dense<-1> : tensor<i32>
  //  CHECK-DAG: %[[ONE:.*]] = mhlo.constant dense<1> : tensor<i32>
  //  CHECK-DAG: %[[ZERO:.*]] = mhlo.constant dense<0> : tensor<i32>
  // CHECK-NEXT: %[[DIM_SIZE_1:.*]] = "mhlo.slice"(%[[DYNAMIC_SHAPE]]) <{limit_indices = dense<1> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>}> : (tensor<2xi32>) -> tensor<1xi32>
  // CHECK-NEXT: %[[DIM_SIZE_SCALAR_1:.*]] = mhlo.reshape %[[DIM_SIZE_1]] : (tensor<1xi32>) -> tensor<i32>
  // CHECK-NEXT: %[[ALL_DIMS_PRODUCT_1:.*]] = mhlo.multiply %[[ONE]], %[[DIM_SIZE_SCALAR_1]] : tensor<i32>
  // CHECK-NEXT: %[[EQ_MINUS_ONE_1:.*]] = mhlo.compare  EQ, %[[DIM_SIZE_SCALAR_1]], %[[MINUS_ONE]],  NOTYPE : (tensor<i32>, tensor<i32>) -> tensor<i1>
  // CHECK-NEXT: %[[DYNAMIC_DIM_1:.*]] = mhlo.select %[[EQ_MINUS_ONE_1]], %[[ONE]], %[[ZERO]] : tensor<i1>, tensor<i32>
  // CHECK-NEXT: %[[NUM_DYNAMIC_DIM_1:.*]] = mhlo.add %[[ZERO]], %[[DYNAMIC_DIM_1]] : tensor<i32>
  // CHECK-NEXT: %[[DIM_SIZE_2:.*]] = "mhlo.slice"(%[[DYNAMIC_SHAPE]]) <{limit_indices = dense<2> : tensor<1xi64>, start_indices = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>}> : (tensor<2xi32>) -> tensor<1xi32>
  // CHECK-NEXT: %[[DIM_SIZE_SCALAR_2:.*]] = mhlo.reshape %[[DIM_SIZE_2]] : (tensor<1xi32>) -> tensor<i32>
  // CHECK-NEXT: %[[ALL_DIMS_PRODUCT:.*]] = mhlo.multiply %[[ALL_DIMS_PRODUCT_1]], %[[DIM_SIZE_SCALAR_2]] : tensor<i32>
  // CHECK-NEXT: %[[EQ_MINUS_ONE_2:.*]] = mhlo.compare  EQ, %[[DIM_SIZE_SCALAR_2]], %[[MINUS_ONE]],  NOTYPE : (tensor<i32>, tensor<i32>) -> tensor<i1>
  // CHECK-NEXT: %[[DYNAMIC_DIM_2:.*]] = mhlo.select %[[EQ_MINUS_ONE_2]], %[[ONE]], %[[ZERO]] : tensor<i1>, tensor<i32>
  // CHECK-NEXT: %[[NUM_DYNAMIC_DIM:.*]] = mhlo.add %[[NUM_DYNAMIC_DIM_1]], %[[DYNAMIC_DIM_2]] : tensor<i32>
  // CHECK-NEXT: %[[ONLY_ONE_DYNAMIC_DIM:.*]] = mhlo.compare  EQ, %[[NUM_DYNAMIC_DIM]], %[[ONE]],  NOTYPE : (tensor<i32>, tensor<i32>) -> tensor<i1>
  // CHECK-NEXT: %[[STATIC_DIMS_PRODUCT:.*]] = mhlo.multiply %[[ALL_DIMS_PRODUCT]], %[[MINUS_ONE]] : tensor<i32>
  // CHECK-NEXT: %[[REM:.*]] = mhlo.remainder %[[NUM_ELEMENTS]], %[[STATIC_DIMS_PRODUCT]] : tensor<i32>
  // CHECK-NEXT: %[[NO_RESIDUAL:.*]] = mhlo.compare  EQ, %[[REM]], %[[ZERO]],  NOTYPE : (tensor<i32>, tensor<i32>) -> tensor<i1>
  // CHECK-NEXT: %[[DYNAMIC_RESHAPABLE:.*]] = mhlo.and %[[NO_RESIDUAL]], %[[ONLY_ONE_DYNAMIC_DIM]] : tensor<i1>
  // CHECK-NEXT: %[[NO_DYNAMIC_DIM:.*]] = mhlo.compare EQ, %16, %[[ZERO]], NOTYPE : (tensor<i32>, tensor<i32>) -> tensor<i1>
  // CHECK-NEXT: %[[NUM_ELEMENTS_EQUALS:.*]] = mhlo.compare EQ, %[[ALL_DIMS_PRODUCT]], %[[NUM_ELEMENTS]], NOTYPE : (tensor<i32>, tensor<i32>) -> tensor<i1>
  // CHECK-NEXT: %[[STATIC_RESHAPABLE:.*]] = mhlo.and %[[NO_DYNAMIC_DIM]], %[[NUM_ELEMENTS_EQUALS]] : tensor<i1>
  // CHECK-NEXT: %[[RESHAPABLE:.*]] = mhlo.or %[[DYNAMIC_RESHAPABLE]], %[[STATIC_RESHAPABLE]] : tensor<i1>
  // CHECK-NEXT: mhlo.custom_call @shape_assertion(%[[RESHAPABLE]]) {error_message = "Shape assertion failed", has_side_effect = true} : (tensor<i1>) -> ()
}

// -----

// CHECK-LABEL: func.func @mhlo_cstr_reshapable_const
func.func @mhlo_cstr_reshapable_const(%arg0: tensor<?x2xf32>) {
  %0 = arith.constant 20 : index
  %1 = mhlo.constant dense<[-1, 4]> : tensor<2xi32>
  %2 = mhlo.cstr_reshapable %0, %1 : (index, tensor<2xi32>) -> !shape.witness
  func.return
  //  CHECK-DAG: %[[DYNAMIC_SHAPE:.*]] = mhlo.constant dense<[-1, 4]> : tensor<2xi32>
  //  CHECK-DAG: %[[NUM_ELEMENTS:.*]] = mhlo.constant dense<20> : tensor<i32>
  //  CHECK-DAG: %[[MINUS_ONE:.*]] = mhlo.constant dense<-1> : tensor<i32>
  //  CHECK-DAG: %[[ONE:.*]] = mhlo.constant dense<1> : tensor<i32>
  //  CHECK-DAG: %[[ZERO:.*]] = mhlo.constant dense<0> : tensor<i32>
  // CHECK-NEXT: %[[DIM_SIZE_1:.*]] = "mhlo.slice"(%[[DYNAMIC_SHAPE]]) <{limit_indices = dense<1> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>}> : (tensor<2xi32>) -> tensor<1xi32>
  // CHECK-NEXT: %[[DIM_SIZE_SCALAR_1:.*]] = mhlo.reshape %[[DIM_SIZE_1]] : (tensor<1xi32>) -> tensor<i32>
  // CHECK-NEXT: %[[ALL_DIMS_PRODUCT_1:.*]] = mhlo.multiply %[[ONE]], %[[DIM_SIZE_SCALAR_1]] : tensor<i32>
  // CHECK-NEXT: %[[EQ_MINUS_ONE_1:.*]] = mhlo.compare  EQ, %[[DIM_SIZE_SCALAR_1]], %[[MINUS_ONE]],  NOTYPE : (tensor<i32>, tensor<i32>) -> tensor<i1>
  // CHECK-NEXT: %[[DYNAMIC_DIM_1:.*]] = mhlo.select %[[EQ_MINUS_ONE_1]], %[[ONE]], %[[ZERO]] : tensor<i1>, tensor<i32>
  // CHECK-NEXT: %[[NUM_DYNAMIC_DIM_1:.*]] = mhlo.add %[[ZERO]], %[[DYNAMIC_DIM_1]] : tensor<i32>
  // CHECK-NEXT: %[[DIM_SIZE_2:.*]] = "mhlo.slice"(%[[DYNAMIC_SHAPE]]) <{limit_indices = dense<2> : tensor<1xi64>, start_indices = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>}> : (tensor<2xi32>) -> tensor<1xi32>
  // CHECK-NEXT: %[[DIM_SIZE_SCALAR_2:.*]] = mhlo.reshape %[[DIM_SIZE_2]] : (tensor<1xi32>) -> tensor<i32>
  // CHECK-NEXT: %[[ALL_DIMS_PRODUCT:.*]] = mhlo.multiply %[[ALL_DIMS_PRODUCT_1]], %[[DIM_SIZE_SCALAR_2]] : tensor<i32>
  // CHECK-NEXT: %[[EQ_MINUS_ONE_2:.*]] = mhlo.compare  EQ, %[[DIM_SIZE_SCALAR_2]], %[[MINUS_ONE]],  NOTYPE : (tensor<i32>, tensor<i32>) -> tensor<i1>
  // CHECK-NEXT: %[[DYNAMIC_DIM_2:.*]] = mhlo.select %[[EQ_MINUS_ONE_2]], %[[ONE]], %[[ZERO]] : tensor<i1>, tensor<i32>
  // CHECK-NEXT: %[[NUM_DYNAMIC_DIM:.*]] = mhlo.add %[[NUM_DYNAMIC_DIM_1]], %[[DYNAMIC_DIM_2]] : tensor<i32>
  // CHECK-NEXT: %[[ONLY_ONE_DYNAMIC_DIM:.*]] = mhlo.compare  EQ, %[[NUM_DYNAMIC_DIM]], %[[ONE]],  NOTYPE : (tensor<i32>, tensor<i32>) -> tensor<i1>
  // CHECK-NEXT: %[[STATIC_DIMS_PRODUCT:.*]] = mhlo.multiply %[[ALL_DIMS_PRODUCT]], %[[MINUS_ONE]] : tensor<i32>
  // CHECK-NEXT: %[[REM:.*]] = mhlo.remainder %[[NUM_ELEMENTS]], %[[STATIC_DIMS_PRODUCT]] : tensor<i32>
  // CHECK-NEXT: %[[NO_RESIDUAL:.*]] = mhlo.compare  EQ, %[[REM]], %[[ZERO]],  NOTYPE : (tensor<i32>, tensor<i32>) -> tensor<i1>
  // CHECK-NEXT: %[[DYNAMIC_RESHAPABLE:.*]] = mhlo.and %[[NO_RESIDUAL]], %[[ONLY_ONE_DYNAMIC_DIM]] : tensor<i1>
  // CHECK-NEXT: %[[NO_DYNAMIC_DIM:.*]] = mhlo.compare EQ, %16, %[[ZERO]], NOTYPE : (tensor<i32>, tensor<i32>) -> tensor<i1>
  // CHECK-NEXT: %[[NUM_ELEMENTS_EQUALS:.*]] = mhlo.compare EQ, %[[ALL_DIMS_PRODUCT]], %[[NUM_ELEMENTS]], NOTYPE : (tensor<i32>, tensor<i32>) -> tensor<i1>
  // CHECK-NEXT: %[[STATIC_RESHAPABLE:.*]] = mhlo.and %[[NO_DYNAMIC_DIM]], %[[NUM_ELEMENTS_EQUALS]] : tensor<i1>
  // CHECK-NEXT: %[[RESHAPABLE:.*]] = mhlo.or %[[DYNAMIC_RESHAPABLE]], %[[STATIC_RESHAPABLE]] : tensor<i1>
  // CHECK-NEXT: mhlo.custom_call @shape_assertion(%[[RESHAPABLE]]) {error_message = "Shape assertion failed", has_side_effect = true} : (tensor<i1>) -> ()
}

// -----

func.func @mhlo_cstr_reshapable_i8(%arg0: index, %arg1: tensor<2xi8>) {
  // expected-error@+1 {{failed to legalize operation 'mhlo.cstr_reshapable' that was explicitly marked illegal}}
  %0 = mhlo.cstr_reshapable %arg0, %arg1 : (index, tensor<2xi8>) -> !shape.witness
  func.return
}
