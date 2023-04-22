// RUN: mlir-hlo-opt -hlo-legalize-to-memref -cse --split-input-file %s | FileCheck %s

// CHECK: #[[MAP:.*]] = affine_map<(d0, d1, d2)[s0, s1, s2] -> (d0 * s0 + d1 * s1 + d2 * s2)>

// CHECK-LABEL: func @dyn_broadcast
func @dyn_broadcast(%operand: tensor<?x?xf32>) -> tensor<?x?x?xf32> {
  // CHECK-SAME: %[[ARG:.*]]: tensor<?x?xf32>
  %c1 = constant 1 : i64
  %shape = tensor.from_elements %c1, %c1, %c1 : tensor<3xi64>
  %result = "mhlo.dynamic_broadcast_in_dim"(%operand, %shape) {
    broadcast_dimensions = dense<[1, 2]> : tensor<2xi64>
  } : (tensor<?x?xf32>, tensor<3xi64>) -> tensor<?x?x?xf32>
  return %result : tensor<?x?x?xf32>
}
// CHECK: %[[SHAPE:.*]] = tensor.from_elements
// CHECK: %[[OPERAND:.*]] = memref.buffer_cast %[[ARG]]

// CHECK: %[[C0:.*]] = constant 0 : index
// CHECK: %[[C1:.*]] = constant 1 : index
// CHECK: %[[OPER_DIM_1:.*]] = memref.dim %[[OPERAND]], %[[C1]] : memref<?x?xf32>
// CHECK: %[[OP_STRIDE_0:.*]] = muli %[[C1]], %[[OPER_DIM_1]] : index
// CHECK: %[[OPER_DIM_0:.*]] = memref.dim %[[OPERAND]], %[[C0]] : memref<?x?xf32>

// CHECK: %[[EL0:.*]] = tensor.extract %[[SHAPE]][%[[C0]]] : tensor<3xi64>
// CHECK: %[[SIZE_0:.*]] = index_cast %[[EL0]] : i64 to index
// CHECK: %[[EL1:.*]] = tensor.extract %[[SHAPE]][%[[C1]]] : tensor<3xi64>

// CHECK: %[[SIZE_1:.*]] = index_cast %[[EL1]] : i64 to index
// CHECK: %[[EXPAND_1:.*]] = cmpi slt, %[[OPER_DIM_0]], %[[SIZE_1]] : index
// CHECK: %[[STRIDE_1:.*]] = select %[[EXPAND_1]], %[[C0]], %[[OP_STRIDE_0]] : index

// CHECK: %[[C2:.*]] = constant 2 : index
// CHECK: %[[EL2:.*]] = tensor.extract %[[SHAPE]][%[[C2]]] : tensor<3xi64>
// CHECK: %[[SIZE_2:.*]] = index_cast %[[EL2]] : i64 to index
// CHECK: %[[EXPAND_2:.*]] = cmpi slt, %[[OPER_DIM_1]], %[[SIZE_2]] : index
// CHECK: %[[STRIDE_2:.*]] = select %[[EXPAND_2]], %[[C0]], %[[C1]] : index

// CHECK: %[[TRANSFORMED_MEMREF:.*]] = memref.reinterpret_cast %[[OPERAND]] to offset: [0], sizes: [%[[SIZE_0]], %[[SIZE_1]], %[[SIZE_2]]], strides: [%[[C0]], %[[STRIDE_1]], %[[STRIDE_2]]] : memref<?x?xf32> to memref<?x?x?xf32, #map>
// CHECK: %[[ALLOC:.*]] = memref.alloc
// CHECK: memref.copy %[[TRANSFORMED_MEMREF]], %[[ALLOC]] : memref<?x?x?xf32, #map> to memref<?x?x?xf32>

// CHECK: %[[RESULT:.*]] = memref.tensor_load %[[ALLOC]]

// CHECK: return %[[RESULT]]

// -----

// CHECK-LABEL: func @dyn_broadcast_unsigned
func @dyn_broadcast_unsigned(%operand: tensor<?x?xui32>) -> tensor<?x?x?xui32> {
  // CHECK-SAME: %[[ARG:.*]]: tensor<?x?xui32>
  %c1 = constant 1 : i64
  %shape = tensor.from_elements %c1, %c1, %c1 : tensor<3xi64>
  %result = "mhlo.dynamic_broadcast_in_dim"(%operand, %shape) {
    broadcast_dimensions = dense<[1, 2]> : tensor<2xi64>
  } : (tensor<?x?xui32>, tensor<3xi64>) -> tensor<?x?x?xui32>
  return %result : tensor<?x?x?xui32>
}

// CHECK: %[[SHAPE:.*]] = tensor.from_elements
// CHECK: %[[OPERAND:.*]] = memref.buffer_cast %[[ARG]]
// CHECK: %[[COPERAND:.*]] = unrealized_conversion_cast %[[OPERAND]] : memref<?x?xui32> to memref<?x?xi32>

// CHECK: %[[C0:.*]] = constant 0 : index
// CHECK: %[[C1:.*]] = constant 1 : index
// CHECK: %[[OPER_DIM_1:.*]] = memref.dim %[[COPERAND]], %[[C1]] : memref<?x?xi32>
// CHECK: %[[OP_STRIDE_0:.*]] = muli %[[C1]], %[[OPER_DIM_1]] : index
// CHECK: %[[OPER_DIM_0:.*]] = memref.dim %[[COPERAND]], %[[C0]] : memref<?x?xi32>

// CHECK: %[[EL0:.*]] = tensor.extract %[[SHAPE]][%[[C0]]] : tensor<3xi64>
// CHECK: %[[SIZE_0:.*]] = index_cast %[[EL0]] : i64 to index
// CHECK: %[[EL1:.*]] = tensor.extract %[[SHAPE]][%[[C1]]] : tensor<3xi64>

// CHECK: %[[SIZE_1:.*]] = index_cast %[[EL1]] : i64 to index
// CHECK: %[[EXPAND_1:.*]] = cmpi slt, %[[OPER_DIM_0]], %[[SIZE_1]] : index
// CHECK: %[[STRIDE_1:.*]] = select %[[EXPAND_1]], %[[C0]], %[[OP_STRIDE_0]] : index

// CHECK: %[[C2:.*]] = constant 2 : index
// CHECK: %[[EL2:.*]] = tensor.extract %[[SHAPE]][%[[C2]]] : tensor<3xi64>
// CHECK: %[[SIZE_2:.*]] = index_cast %[[EL2]] : i64 to index
// CHECK: %[[EXPAND_2:.*]] = cmpi slt, %[[OPER_DIM_1]], %[[SIZE_2]] : index
// CHECK: %[[STRIDE_2:.*]] = select %[[EXPAND_2]], %[[C0]], %[[C1]] : index

// CHECK: %[[TRANSFORMED_MEMREF:.*]] = memref.reinterpret_cast %[[COPERAND]] to offset: [0], sizes: [%[[SIZE_0]], %[[SIZE_1]], %[[SIZE_2]]], strides: [%[[C0]], %[[STRIDE_1]], %[[STRIDE_2]]] : memref<?x?xi32> to memref<?x?x?xi32, #map>
// CHECK: %[[ALLOC:.*]] = memref.alloc
// CHECK: memref.copy %[[TRANSFORMED_MEMREF]], %[[ALLOC]] : memref<?x?x?xi32, #map> to memref<?x?x?xi32>
// CHECK: %[[CALLOC:.*]] = unrealized_conversion_cast %[[ALLOC]] : memref<?x?x?xi32> to memref<?x?x?xui32>

// CHECK: %[[RESULT:.*]] = memref.tensor_load %[[CALLOC]]

// CHECK: return %[[RESULT]]

// -----

// CHECK-LABEL: func @dyn_reshape_unsigned
func @dyn_reshape_unsigned(%operand: tensor<?x?xui32>) -> tensor<?x?x?xui32> {
  // CHECK-SAME: %[[ARG:.*]]: tensor<?x?xui32>
  %c1 = constant 1 : i64
  %shape = tensor.from_elements %c1, %c1, %c1 : tensor<3xi64>
  %result = "mhlo.dynamic_reshape"(%operand, %shape) : (tensor<?x?xui32>, tensor<3xi64>) -> tensor<?x?x?xui32>
  return %result : tensor<?x?x?xui32>
}

// CHECK: %[[SHAPE:.*]] = tensor.from_elements
// CHECK: %[[OPERAND:.*]] = memref.buffer_cast %[[ARG]]
// CHECK: %[[BSHAPE:.*]] = memref.buffer_cast %[[SHAPE]]
// CHECK: %[[COPERAND:.*]] = unrealized_conversion_cast %[[OPERAND]] : memref<?x?xui32> to memref<?x?xi32>

// CHECK: %[[RESHAPED:.*]] = memref.reshape %[[COPERAND]](%[[BSHAPE]]) : (memref<?x?xi32>, memref<3xi64>) -> memref<?x?x?xi32>
// CHECK: %[[RESULT:.*]] = unrealized_conversion_cast %[[RESHAPED]] : memref<?x?x?xi32> to memref<?x?x?xui32>
// CHECK: %[[TRESULT:.*]] = memref.tensor_load %[[RESULT]] : memref<?x?x?xui32>
// CHECK: return %[[TRESULT]]

// -----

// CHECK-LABEL: func @reshape_unsigned
func @reshape_unsigned(%operand: tensor<*xui32>) -> tensor<4x3xui32> {
  // CHECK-SAME: %[[ARG:.*]]: tensor<*xui32>
  %result = "mhlo.reshape"(%operand) : (tensor<*xui32>) -> tensor<4x3xui32>
  return %result : tensor<4x3xui32>
}

// CHECK: %[[OPERAND:.*]] = memref.buffer_cast %[[ARG]]
// CHECK: %[[COPERAND:.*]] = unrealized_conversion_cast %[[OPERAND]] : memref<*xui32> to memref<*xi32>
// CHECK: %[[RESHAPED:.*]] = memref.cast %[[COPERAND]] : memref<*xi32> to memref<4x3xi32>
// CHECK: %[[RESULT:.*]] = unrealized_conversion_cast %[[RESHAPED]] : memref<4x3xi32> to memref<4x3xui32>
// CHECK: %[[TRESULT:.*]] = memref.tensor_load %[[RESULT]] : memref<4x3xui32>
// CHECK: return %[[TRESULT]]
