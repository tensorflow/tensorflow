// RUN: mlir-hlo-opt --shape-legalize-to-hlo --split-input-file --verify-diagnostics %s | FileCheck %s

// CHECK-LABEL: func.func @compute_reshape_shape
func.func @compute_reshape_shape(%arg0: index, %arg1: tensor<2xi32>) -> tensor<2xi32> {
  %0 = mhlo.compute_reshape_shape %arg0, %arg1 : (index, tensor<2xi32>) -> tensor<2xi32>
  func.return %0 : tensor<2xi32>
  //      CHECK: %[[ARG0_I32:.*]] = builtin.unrealized_conversion_cast %arg0 : index to tensor<i32>
  // CHECK-NEXT: %[[TMP0:.*]] = mhlo.constant dense<-1> : tensor<i32>
  // CHECK-NEXT: %[[INPUT_SIZE0x1:.*]] = "mhlo.slice"(%arg1) <{limit_indices = dense<1> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>}> : (tensor<2xi32>) -> tensor<1xi32>
  // CHECK-NEXT: %[[INPUT_SIZE0:.*]] = mhlo.reshape %[[INPUT_SIZE0x1]] : (tensor<1xi32>) -> tensor<i32>
  // CHECK-NEXT: %[[TMP1:.*]] = mhlo.multiply %[[TMP0]], %[[INPUT_SIZE0]] : tensor<i32>
  // CHECK-NEXT: %[[INPUT_SIZE1x1:.*]] = "mhlo.slice"(%arg1) <{limit_indices = dense<2> : tensor<1xi64>, start_indices = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>}> : (tensor<2xi32>) -> tensor<1xi32>
  // CHECK-NEXT: %[[INPUT_SIZE1:.*]] = mhlo.reshape %[[INPUT_SIZE1x1]] : (tensor<1xi32>) -> tensor<i32>
  // CHECK-NEXT: %[[INPUT_SIZE_PRODUCT:.*]] = mhlo.multiply %[[TMP1]], %[[INPUT_SIZE1]] : tensor<i32>
  // CHECK-NEXT: %[[COMPUTED_SIZE:.*]] = mhlo.divide %[[ARG0_I32]], %[[INPUT_SIZE_PRODUCT]] : tensor<i32>
  // CHECK-NEXT: %[[M1:.*]] = mhlo.constant dense<-1> : tensor<i32>
  // CHECK-NEXT: %[[INPUT_SIZE0_EQ_M1:.*]] = mhlo.compare  EQ, %3, %[[M1]],  NOTYPE : (tensor<i32>, tensor<i32>) -> tensor<i1>
  // CHECK-NEXT: %[[RESULT_SIZE0:.*]] = mhlo.select %[[INPUT_SIZE0_EQ_M1]], %[[COMPUTED_SIZE]], %3 : tensor<i1>, tensor<i32>
  // CHECK-NEXT: %[[RESULT_SIZE0x1:.*]] = mhlo.reshape %[[RESULT_SIZE0]] : (tensor<i32>) -> tensor<1xi32>
  // CHECK-NEXT: %[[INPUT_SIZE1_EQ_M1:.*]] = mhlo.compare  EQ, %6, %[[M1]],  NOTYPE : (tensor<i32>, tensor<i32>) -> tensor<i1>
  // CHECK-NEXT: %[[RESULT_SIZE1:.*]] = mhlo.select %[[INPUT_SIZE1_EQ_M1]], %[[COMPUTED_SIZE]], %6 : tensor<i1>, tensor<i32>
  // CHECK-NEXT: %[[RESULT_SIZE1x1:.*]] = mhlo.reshape %[[RESULT_SIZE1]] : (tensor<i32>) -> tensor<1xi32>
  // CHECK-NEXT: %[[RESULT:.*]] = "mhlo.concatenate"(%[[RESULT_SIZE0x1]], %[[RESULT_SIZE1x1]]) <{dimension = 0 : i64}> : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
  // CHECK-NEXT: return %[[RESULT]] : tensor<2xi32>
}

// -----

// CHECK-LABEL: func.func @num_elements_tensor_to_index
func.func @num_elements_tensor_to_index(%arg0: tensor<2xindex>) -> index {
  %0 = shape.num_elements %arg0 : tensor<2xindex> -> index
  func.return %0 : index
  //      CHECK: %[[ARG0_I32:.*]] = builtin.unrealized_conversion_cast %arg0 : tensor<2xindex> to tensor<2xi32>
  // CHECK-NEXT: %[[TMP0:.*]] = mhlo.constant dense<1> : tensor<i32>
  // CHECK-NEXT: %[[SIZE0x1:.*]] = "mhlo.slice"(%[[ARG0_I32]]) <{limit_indices = dense<1> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>}> : (tensor<2xi32>) -> tensor<1xi32>
  // CHECK-NEXT: %[[SIZE0:.*]] = mhlo.reshape %[[SIZE0x1]] : (tensor<1xi32>) -> tensor<i32>
  // CHECK-NEXT: %[[TMP1:.*]] = mhlo.multiply %[[TMP0]], %[[SIZE0]] : tensor<i32>
  // CHECK-NEXT: %[[SIZE1x1:.*]] = "mhlo.slice"(%[[ARG0_I32]]) <{limit_indices = dense<2> : tensor<1xi64>, start_indices = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>}> : (tensor<2xi32>) -> tensor<1xi32>
  // CHECK-NEXT: %[[SIZE1:.*]] = mhlo.reshape %[[SIZE1x1]] : (tensor<1xi32>) -> tensor<i32>
  // CHECK-NEXT: %[[RESULT_I32:.*]] = mhlo.multiply %[[TMP1]], %[[SIZE1]] : tensor<i32>
  // CHECK-NEXT: %[[RESULT_INDEX:.*]] = builtin.unrealized_conversion_cast %[[RESULT_I32]] : tensor<i32> to index
  // CHECK-NEXT: return %[[RESULT_INDEX]] : index
}

// -----

func.func @num_elements_shape_to_xxx(%arg0: !shape.shape) -> !shape.size {
  // expected-error@+1 {{failed to legalize operation 'shape.num_elements' that was explicitly marked illegal}}
  %0 = shape.num_elements %arg0 : !shape.shape -> !shape.size
  func.return %0 : !shape.size
}

// -----

func.func @num_elements_xxx_to_size(%arg0: tensor<2xindex>) -> !shape.size {
  // expected-error@+1 {{failed to legalize operation 'shape.num_elements' that was explicitly marked illegal}}
  %0 = shape.num_elements %arg0 : tensor<2xindex> -> !shape.size
  func.return %0 : !shape.size
}

// -----

// CHECK-LABEL: func.func @shape_of_ranked
func.func @shape_of_ranked_to_index(%arg0: tensor<?x1xf32>) -> tensor<2xindex> {
  %0 = shape.shape_of %arg0 : tensor<?x1xf32> -> tensor<2xindex>
  func.return %0 : tensor<2xindex>
  //      CHECK: %[[SIZE0x1:.*]] = "mhlo.get_dimension_size"(%arg0) <{dimension = 0 : i64}> : (tensor<?x1xf32>) -> tensor<i32>
  // CHECK-NEXT: %[[SIZE0:.*]] = mhlo.reshape %[[SIZE0x1]] : (tensor<i32>) -> tensor<1xi32>
  // CHECK-NEXT: %[[SIZE1x1:.*]] = "mhlo.get_dimension_size"(%arg0) <{dimension = 1 : i64}> : (tensor<?x1xf32>) -> tensor<i32>
  // CHECK-NEXT: %[[SIZE1:.*]] = mhlo.reshape %[[SIZE1x1]] : (tensor<i32>) -> tensor<1xi32>
  // CHECK-NEXT: %[[RESULT_I32:.*]] = "mhlo.concatenate"(%[[SIZE0]], %[[SIZE1]]) <{dimension = 0 : i64}> : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
  // CHECK-NEXT: %[[RESULT_INDEX:.*]] = builtin.unrealized_conversion_cast %[[RESULT_I32]] : tensor<2xi32> to tensor<2xindex>
  // CHECK-NEXT: return %[[RESULT_INDEX]] : tensor<2xindex>
}

// -----

func.func @shape_of_unranked_to_xxx(%arg0: tensor<*xf32>) -> tensor<?xindex> {
  // expected-error@+1 {{failed to legalize operation 'shape.shape_of' that was explicitly marked illegal}}
  %0 = shape.shape_of %arg0 : tensor<*xf32> -> tensor<?xindex>
  func.return %0 : tensor<?xindex>
}

// -----

func.func @shape_of_ranked_to_shape(%arg0: tensor<?x1xf32>) -> !shape.shape {
  // expected-error@+1 {{failed to legalize operation 'shape.shape_of' that was explicitly marked illegal}}
  %0 = shape.shape_of %arg0 : tensor<?x1xf32> -> !shape.shape
  func.return %0 : !shape.shape
}

// -----

// CHECK-LABEL: func.func @tensor_dim
func.func @tensor_dim(%arg0: tensor<?x?xf32>) -> index {
  %c0 = arith.constant 0 : index
  %dim = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  func.return %dim : index
  //      CHECK: %[[DIM_SIZE:.*]] = "mhlo.get_dimension_size"(%arg0) <{dimension = 0 : i64}> : (tensor<?x?xf32>) -> tensor<i32>
  // CHECK-NEXT: %[[DIM_SIZE_INDEX:.*]] = builtin.unrealized_conversion_cast %[[DIM_SIZE]] : tensor<i32> to index
  // CHECK-NEXT: return %[[DIM_SIZE_INDEX]] : index
}

// -----

func.func @tensor_dim_dynamic(%arg0: tensor<?x?xf32>, %arg1: index) -> index {
  // expected-error@+1 {{failed to legalize operation 'tensor.dim' that was explicitly marked illegal}}
  %dim = tensor.dim %arg0, %arg1 : tensor<?x?xf32>
  func.return %dim : index
}

// -----

// CHECK-LABEL: func.func @tensor_from_elements
func.func @tensor_from_elements(%arg0: index) -> tensor<2xindex> {
  %c0 = arith.constant 0 : index
  %0 = tensor.from_elements %arg0, %c0 : tensor<2xindex>
  func.return %0 : tensor<2xindex>
  //      CHECK: %[[ELEMENT1_SCALAR:.*]] = builtin.unrealized_conversion_cast %arg0 : index to tensor<i32>
  // CHECK-NEXT: %[[ELEMENT1:.*]] = mhlo.reshape %[[ELEMENT1_SCALAR]] : (tensor<i32>) -> tensor<1xi32>
  // CHECK-NEXT: %[[ELEMENT2:.*]] = mhlo.constant dense<0> : tensor<1xi32>
  // CHECK-NEXT: %[[CONCAT:.*]] = "mhlo.concatenate"(%[[ELEMENT1]], %[[ELEMENT2]]) <{dimension = 0 : i64}> : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
  // CHECK-NEXT: %[[CONCAT_INDEX:.*]] = builtin.unrealized_conversion_cast %[[CONCAT]] : tensor<2xi32> to tensor<2xindex>
  // CHECK-NEXT: return %[[CONCAT_INDEX]] : tensor<2xindex>
}

// -----

func.func @tensor_from_elements_i8(%arg0: i8) -> tensor<2xi8> {
  %c0 = arith.constant 0 : i8
  // expected-error@+1 {{failed to legalize operation 'tensor.from_elements' that was explicitly marked illegal}}
  %0 = tensor.from_elements %arg0, %c0 : tensor<2xi8>
  func.return %0 : tensor<2xi8>
}

// -----

// CHECK-LABEL: func.func @tensor_from_elements_scalar
func.func @tensor_from_elements_scalar(%arg0: i64) -> tensor<i64> {
  %0 = tensor.from_elements %arg0 : tensor<i64>
  func.return %0 : tensor<i64>
  //      CHECK: %[[RESULT:.*]] = builtin.unrealized_conversion_cast %arg0 : i64 to tensor<i64>
  // CHECK-NEXT: return %[[RESULT]] : tensor<i64>
}

// -----

func.func @tensor_from_elements_rank2(%arg0: index) -> tensor<2x1xindex> {
  %c0 = arith.constant 0 : index
  // expected-error@+1 {{failed to legalize operation 'tensor.from_elements' that was explicitly marked illegal}}
  %0 = tensor.from_elements %arg0, %c0 : tensor<2x1xindex>
  func.return %0 : tensor<2x1xindex>
}

// -----

// CHECK-LABEL: func.func @shape_broadcast
func.func @shape_broadcast(%arg0: tensor<4xindex>, %arg1: tensor<4xindex>) -> tensor<4xindex> {
  %0 = shape.broadcast %arg0, %arg1 : tensor<4xindex>, tensor<4xindex> -> tensor<4xindex>
  func.return %0 : tensor<4xindex>
  //      CHECK: %[[LHS:.*]] = builtin.unrealized_conversion_cast %arg0 : tensor<4xindex> to tensor<4xi32>
  // CHECK-NEXT: %[[RHS:.*]] = builtin.unrealized_conversion_cast %arg1 : tensor<4xindex> to tensor<4xi32>
  // CHECK-NEXT: %[[BROADCAST:.*]] = mhlo.maximum %[[LHS]], %[[RHS]] : tensor<4xi32>
  // CHECK-NEXT: %[[BROADCAST_INDEX:.*]] = builtin.unrealized_conversion_cast %[[BROADCAST]] : tensor<4xi32> to tensor<4xindex>
  // CHECK-NEXT: return %[[BROADCAST_INDEX]] : tensor<4xindex>
}

// -----

func.func @shape_broadcast_different_dims(%arg0: tensor<4xindex>, %arg1: tensor<6xindex>) -> tensor<6xindex> {
  %0 = shape.broadcast %arg0, %arg1 : tensor<4xindex>, tensor<6xindex> -> tensor<6xindex>
  func.return %0 : tensor<6xindex>
  //      CHECK: %[[LHS:.*]] = builtin.unrealized_conversion_cast %arg0 : tensor<4xindex> to tensor<4xi32>
  // CHECK-NEXT: %[[RHS:.*]] = builtin.unrealized_conversion_cast %arg1 : tensor<6xindex> to tensor<6xi32>
  // CHECK-NEXT: %[[PAD:.*]] = mhlo.constant dense<1> : tensor<2xi32>
  // CHECK-NEXT: %[[LHS_PAD:.*]] = "mhlo.concatenate"(%[[PAD]], %[[LHS]]) <{dimension = 0 : i64}> : (tensor<2xi32>, tensor<4xi32>) -> tensor<6xi32>
  // CHECK-NEXT: %[[BROADCAST:.*]] = mhlo.maximum %[[LHS_PAD]], %[[RHS]] : tensor<6xi32>
  // CHECK-NEXT: %[[BROADCAST_INDEX:.*]] = builtin.unrealized_conversion_cast %[[BROADCAST]] : tensor<6xi32> to tensor<6xindex>
  // CHECK-NEXT: return %[[BROADCAST_INDEX]] : tensor<6xindex>
}

// -----

func.func @shape_broadcast_result_shape(%arg0: tensor<4xindex>, %arg1: tensor<4xindex>) -> !shape.shape {
  // expected-error@+1 {{failed to legalize operation 'shape.broadcast' that was explicitly marked illegal}}
  %0 = shape.broadcast %arg0, %arg1 : tensor<4xindex>, tensor<4xindex> -> !shape.shape
  func.return %0 : !shape.shape
}

// -----

func.func @shape_broadcast_input_shape(%arg0: !shape.shape, %arg1: !shape.shape) -> !shape.shape {
  // expected-error@+1 {{failed to legalize operation 'shape.broadcast' that was explicitly marked illegal}}
  %0 = shape.broadcast %arg0, %arg1 : !shape.shape, !shape.shape -> !shape.shape
  func.return %0 : !shape.shape
}

// -----

func.func @shape_broadcast_too_many_operands(%arg0: tensor<4xindex>, %arg1: tensor<4xindex>, %arg2: tensor<4xindex>) -> tensor<4xindex> {
  // expected-error@+1 {{failed to legalize operation 'shape.broadcast' that was explicitly marked illegal}}
  %0 = shape.broadcast %arg0, %arg1, %arg2 : tensor<4xindex>, tensor<4xindex>, tensor<4xindex> -> tensor<4xindex>
  func.return %0 : tensor<4xindex>
}

// -----

func.func @shape_cstr_broadcastable(%arg0: tensor<2xindex>, %arg1: tensor<2xindex>) -> !shape.witness {
  // expected-error@+1 {{failed to legalize operation 'shape.cstr_broadcastable' that was explicitly marked illegal}}
  %0 = shape.cstr_broadcastable %arg0, %arg1 : tensor<2xindex>, tensor<2xindex>
  func.return %0 : !shape.witness
}

// -----

func.func @mhlo_cstr_reshapable(%arg0: index, %arg1: tensor<2xindex>, %arg2: tensor<?x2xf32>) -> tensor<?x4xf32> {
  // expected-error@+1 {{failed to legalize operation 'mhlo.cstr_reshapable' that was explicitly marked illegal}}
  %0 = mhlo.cstr_reshapable %arg0, %arg1 : (index, tensor<2xindex>) -> !shape.witness
  %1 = shape.assuming %0 -> (tensor<?x4xf32>) {
    %2 = mhlo.dynamic_reshape %arg2, %arg1 : (tensor<?x2xf32>, tensor<2xindex>) -> tensor<?x4xf32>
    shape.assuming_yield %2 : tensor<?x4xf32>
  }
  func.return %1 : tensor<?x4xf32>
}

// -----

// CHECK-LABEL: func @const_shape
func.func @const_shape() -> tensor<2xindex> {
  %0 = shape.const_shape [6, 4] : tensor<2xindex>
  return %0 : tensor<2xindex>
  //      CHECK: %[[CST:.*]] = mhlo.constant dense<[6, 4]> : tensor<2xi32>
  // CHECK-NEXT: %[[CST_INDEX:.*]] = builtin.unrealized_conversion_cast %[[CST]] : tensor<2xi32> to tensor<2xindex>
  // CHECK-NEXT: return %[[CST_INDEX]] : tensor<2xindex>
}

// -----

// CHECK-LABEL: func @index_cast_index_to_i32
func.func @index_cast_index_to_i32(%arg0: tensor<2xindex>) -> tensor<2xi32> {
  %0 = arith.index_cast %arg0 : tensor<2xindex> to tensor<2xi32>
  return %0 : tensor<2xi32>
  // CHECK-NEXT: %[[CST_I32:.*]] = builtin.unrealized_conversion_cast %arg0 : tensor<2xindex> to tensor<2xi32>
  // CHECK-NEXT: return %[[CST_I32]] : tensor<2xi32>
}

// -----

// CHECK-LABEL: func @index_cast_i32_to_index
func.func @index_cast_i32_to_index(%arg0: tensor<2xi32>) -> tensor<2xindex> {
  %0 = arith.index_cast %arg0 : tensor<2xi32> to tensor<2xindex>
  return %0 : tensor<2xindex>
  // CHECK-NEXT: %[[CST_INDEX:.*]] = builtin.unrealized_conversion_cast %arg0 : tensor<2xi32> to tensor<2xindex>
  // CHECK-NEXT: return %[[CST_INDEX]] : tensor<2xindex>
}

// -----

// CHECK-LABEL: func @index_cast_scalar_index_to_i32
func.func @index_cast_scalar_index_to_i32(%arg0: index) -> i32 {
  //      CHECK: %[[CAST_I32:.*]] = builtin.unrealized_conversion_cast %arg0 : index to tensor<i32>
  // CHECK-NEXT: %[[CAST_INDEX:.*]] = builtin.unrealized_conversion_cast %[[CAST_I32]] : tensor<i32> to i32
  // CHECK-NEXT: return %[[CAST_INDEX]] : i32
  %0 = arith.index_cast %arg0 : index to i32
  return %0 : i32
}

// -----

// CHECK-LABEL: func @index_cast_scalar_index_to_i64
func.func @index_cast_scalar_index_to_i64(%arg0: index) -> i64 {
  //      CHECK: %[[CAST_I32:.*]] = builtin.unrealized_conversion_cast %arg0 : index to tensor<i32>
  // CHECK-NEXT: %[[CONVERT:.*]] = mhlo.convert %[[CAST_I32]] : (tensor<i32>) -> tensor<i64>
  // CHECK-NEXT: %[[CAST_INDEX:.*]] = builtin.unrealized_conversion_cast %[[CONVERT]] : tensor<i64> to i64
  // CHECK-NEXT: return %[[CAST_INDEX]] : i64
  %0 = arith.index_cast %arg0 : index to i64
  return %0 : i64
}

// -----

func.func @index_cast_scalar_i32_to_index(%arg0: i32) -> index {
  //      CHECK: %[[CAST_I32:.*]] = builtin.unrealized_conversion_cast %arg0 : i32 to tensor<i32>
  // CHECK-NEXT: %[[CAST_INDEX:.*]] = builtin.unrealized_conversion_cast %[[CAST_I32]] : tensor<i32> to index
  // CHECK-NEXT: return %[[CAST_INDEX]] : index
  %0 = arith.index_cast %arg0 : i32 to index
  return %0 : index
}

// -----

func.func @index_cast_index_to_i8(%arg0: tensor<2xindex>) -> tensor<2xi8> {
  // expected-error@+1 {{failed to legalize operation 'arith.index_cast' that was explicitly marked illegal}}
  %0 = arith.index_cast %arg0 : tensor<2xindex> to tensor<2xi8>
  return %0 : tensor<2xi8>
}

// -----

func.func @index_cast_i8_to_index(%arg0: tensor<2xi8>) -> tensor<2xindex> {
  // expected-error@+1 {{failed to legalize operation 'arith.index_cast' that was explicitly marked illegal}}
  %0 = arith.index_cast %arg0 : tensor<2xi8> to tensor<2xindex>
  return %0 : tensor<2xindex>
}


// -----

// CHECK-LABEL: func @muli
func.func @muli(%arg0: index, %arg1: index) -> index {
  %0 = arith.muli %arg0, %arg1 : index
  return %0 : index
  //      CHECK: %[[LHS:.*]] = builtin.unrealized_conversion_cast %arg0 : index to tensor<i32>
  // CHECK-NEXT: %[[RHS:.*]] = builtin.unrealized_conversion_cast %arg1 : index to tensor<i32>
  // CHECK-NEXT: %[[RES:.*]] = mhlo.multiply %[[LHS]], %[[RHS]] : tensor<i32>
  // CHECK-NEXT: %[[RES_INDEX:.*]] = builtin.unrealized_conversion_cast %[[RES]] : tensor<i32> to index
  // CHECK-NEXT: return %[[RES_INDEX]] : index
}

// -----

// CHECK-LABEL: func @muli_const
func.func @muli_const() -> index {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %0 = arith.muli %c1, %c2 : index
  return %0 : index
  //      CHECK: %[[LHS:.*]] = mhlo.constant dense<1> : tensor<i32>
  // CHECK-NEXT: %[[RHS:.*]] = mhlo.constant dense<2> : tensor<i32>
  // CHECK-NEXT: %[[RES:.*]] = mhlo.multiply %[[LHS]], %[[RHS]] : tensor<i32>
  // CHECK-NEXT: %[[RES_INDEX:.*]] = builtin.unrealized_conversion_cast %[[RES]] : tensor<i32> to index
  // CHECK-NEXT: return %[[RES_INDEX]] : index
}

// -----

func.func @muli_i32(%arg0: i32, %arg1: i32) -> i32 {
  // expected-error@+1 {{failed to legalize operation 'arith.muli' that was explicitly marked illegal}}
  %0 = arith.muli %arg0, %arg1 : i32
  return %0 : i32
}

// -----

// CHECK-LABEL: func @tensor_extract
func.func @tensor_extract(%arg0: tensor<3x3xindex>) -> index {
  %c1 = arith.constant 0 : index
  %c2 = arith.constant 1 : index
  %0 = tensor.extract %arg0[%c1, %c2] : tensor<3x3xindex>
  return %0 : index
  //      CHECK: %[[CAST:.*]] = builtin.unrealized_conversion_cast %arg0 : tensor<3x3xindex> to tensor<3x3xi32>
  // CHECK-NEXT: %[[SLICE:.*]] = "mhlo.slice"(%[[CAST]])
  // CHECK-SAME: limit_indices = dense<[1, 2]> : tensor<2xi64>
  // CHECK-SAME: start_indices = dense<[0, 1]> : tensor<2xi64>
  // CHECK-SAME: strides = dense<1> : tensor<2xi64>
  // CHECK-SAME: (tensor<3x3xi32>) -> tensor<1x1xi32>
  // CHECK-NEXT: %[[RESHAPE:.*]] = mhlo.reshape %[[SLICE]] : (tensor<1x1xi32>) -> tensor<i32>
  // CHECK-NEXT: %[[RES_INDEX:.*]] = builtin.unrealized_conversion_cast %[[RESHAPE]] : tensor<i32> to index
  // CHECK-NEXT: return %[[RES_INDEX]] : index
}

// -----

// CHECK-LABEL: func @tensor_extract_i32
func.func @tensor_extract_i32(%arg0: tensor<3x3xi32>) -> i32 {
  %c1 = arith.constant 0 : index
  %c2 = arith.constant 1 : index
  %0 = tensor.extract %arg0[%c1, %c2] : tensor<3x3xi32>
  return %0 : i32
  //      CHECK: %[[SLICE:.*]] = "mhlo.slice"(%arg0)
  // CHECK-SAME: limit_indices = dense<[1, 2]> : tensor<2xi64>
  // CHECK-SAME: start_indices = dense<[0, 1]> : tensor<2xi64>
  // CHECK-SAME: strides = dense<1> : tensor<2xi64>
  // CHECK-SAME: (tensor<3x3xi32>) -> tensor<1x1xi32>
  // CHECK-NEXT: %[[RESHAPE:.*]] = mhlo.reshape %[[SLICE]] : (tensor<1x1xi32>) -> tensor<i32>
  // CHECK-NEXT: %[[RES_I32:.*]] = builtin.unrealized_conversion_cast %[[RESHAPE]] : tensor<i32> to i32
  // CHECK-NEXT: return %[[RES_I32]] : i32
}

// -----

func.func @tensor_extract_out_of_range(%arg0: tensor<3x3xindex>) -> index {
  %c1 = arith.constant 4 : index
  %c2 = arith.constant 4 : index
  // expected-error@+1 {{failed to legalize operation 'tensor.extract' that was explicitly marked illegal}}
  %0 = tensor.extract %arg0[%c1, %c2] : tensor<3x3xindex>
  return %0 : index
}

// -----

func.func @tensor_extract_dynamic(%arg0: tensor<?x3xindex>) -> index {
  %c1 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  // expected-error@+1 {{failed to legalize operation 'tensor.extract' that was explicitly marked illegal}}
  %0 = tensor.extract %arg0[%c1, %c2] : tensor<?x3xindex>
  return %0 : index
}
