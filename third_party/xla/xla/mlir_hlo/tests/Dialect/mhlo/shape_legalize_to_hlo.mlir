// RUN: mlir-hlo-opt --shape-legalize-to-hlo --split-input-file --verify-diagnostics %s | FileCheck %s

// CHECK-LABEL: func.func @compute_reshape_shape
func.func @compute_reshape_shape(%arg0: index, %arg1: tensor<2xi32>) -> tensor<2xi32> {
  %0 = mhlo.compute_reshape_shape %arg0, %arg1 : (index, tensor<2xi32>) -> tensor<2xi32>
  func.return %0 : tensor<2xi32>
  //      CHECK: %[[ARG0_I32:.*]] = builtin.unrealized_conversion_cast %arg0 : index to tensor<i32>
  // CHECK-NEXT: %[[TMP0:.*]] = mhlo.constant dense<-1> : tensor<i32>
  // CHECK-NEXT: %[[INPUT_SIZE0x1:.*]] = "mhlo.slice"(%arg1) {limit_indices = dense<1> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<2xi32>) -> tensor<1xi32>
  // CHECK-NEXT: %[[INPUT_SIZE0:.*]] = mhlo.reshape %[[INPUT_SIZE0x1]] : (tensor<1xi32>) -> tensor<i32>
  // CHECK-NEXT: %[[TMP1:.*]] = mhlo.multiply %[[TMP0]], %[[INPUT_SIZE0]] : tensor<i32>
  // CHECK-NEXT: %[[INPUT_SIZE1x1:.*]] = "mhlo.slice"(%arg1) {limit_indices = dense<2> : tensor<1xi64>, start_indices = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<2xi32>) -> tensor<1xi32>
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
  // CHECK-NEXT: %[[RESULT:.*]] = "mhlo.concatenate"(%[[RESULT_SIZE0x1]], %[[RESULT_SIZE1x1]]) {dimension = 0 : i64} : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
  // CHECK-NEXT: return %[[RESULT]] : tensor<2xi32>
}

// -----

// CHECK-LABEL: func.func @num_elements_tensor_to_index
func.func @num_elements_tensor_to_index(%arg0: tensor<2xindex>) -> index {
  %0 = shape.num_elements %arg0 : tensor<2xindex> -> index
  func.return %0 : index
  //      CHECK: %[[ARG0_I32:.*]] = builtin.unrealized_conversion_cast %arg0 : tensor<2xindex> to tensor<2xi32>
  // CHECK-NEXT: %[[TMP0:.*]] = mhlo.constant dense<1> : tensor<i32>
  // CHECK-NEXT: %[[SIZE0x1:.*]] = "mhlo.slice"(%[[ARG0_I32]]) {limit_indices = dense<1> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<2xi32>) -> tensor<1xi32>
  // CHECK-NEXT: %[[SIZE0:.*]] = mhlo.reshape %[[SIZE0x1]] : (tensor<1xi32>) -> tensor<i32>
  // CHECK-NEXT: %[[TMP1:.*]] = mhlo.multiply %[[TMP0]], %[[SIZE0]] : tensor<i32>
  // CHECK-NEXT: %[[SIZE1x1:.*]] = "mhlo.slice"(%[[ARG0_I32]]) {limit_indices = dense<2> : tensor<1xi64>, start_indices = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<2xi32>) -> tensor<1xi32>
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
  //      CHECK: %[[SIZE0x1:.*]] = "mhlo.get_dimension_size"(%arg0) {dimension = 0 : i64} : (tensor<?x1xf32>) -> tensor<i32>
  // CHECK-NEXT: %[[SIZE0:.*]] = mhlo.reshape %[[SIZE0x1]] : (tensor<i32>) -> tensor<1xi32>
  // CHECK-NEXT: %[[SIZE1x1:.*]] = "mhlo.get_dimension_size"(%arg0) {dimension = 1 : i64} : (tensor<?x1xf32>) -> tensor<i32>
  // CHECK-NEXT: %[[SIZE1:.*]] = mhlo.reshape %[[SIZE1x1]] : (tensor<i32>) -> tensor<1xi32>
  // CHECK-NEXT: %[[RESULT_I32:.*]] = "mhlo.concatenate"(%[[SIZE0]], %[[SIZE1]]) {dimension = 0 : i64} : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
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
  //      CHECK: %[[DIM_SIZE:.*]] = "mhlo.get_dimension_size"(%arg0) {dimension = 0 : i64} : (tensor<?x?xf32>) -> tensor<i32>
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
  // CHECK-NEXT: %[[CONCAT:.*]] = "mhlo.concatenate"(%[[ELEMENT1]], %[[ELEMENT2]]) {dimension = 0 : i64} : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
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

func.func @shape_broadcast_different_dims(%arg0: tensor<4xindex>, %arg1: tensor<6xindex>) -> tensor<6xindex> {
  // expected-error@+1 {{failed to legalize operation 'shape.broadcast' that was explicitly marked illegal}}
  %0 = shape.broadcast %arg0, %arg1 : tensor<4xindex>, tensor<6xindex> -> tensor<6xindex>
  func.return %0 : tensor<6xindex>
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
