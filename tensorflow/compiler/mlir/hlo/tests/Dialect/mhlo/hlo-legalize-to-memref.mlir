// RUN: mlir-hlo-opt -hlo-legalize-to-memref -cse --split-input-file %s | FileCheck %s

// CHECK: #[[MAP:.*]] = affine_map<(d0, d1, d2)[s0, s1, s2] -> (d0 * s0 + d1 * s1 + d2 * s2)>

// CHECK-LABEL: func @dyn_broadcast
func @dyn_broadcast(%operand: tensor<?x?xf32>) -> tensor<?x?x?xf32> {
  // CHECK-SAME: %[[ARG:.*]]: tensor<?x?xf32>
  %c1 = arith.constant 1 : i64
  %shape = tensor.from_elements %c1, %c1, %c1 : tensor<3xi64>
  %result = "mhlo.dynamic_broadcast_in_dim"(%operand, %shape) {
    broadcast_dimensions = dense<[1, 2]> : tensor<2xi64>
  } : (tensor<?x?xf32>, tensor<3xi64>) -> tensor<?x?x?xf32>
  return %result : tensor<?x?x?xf32>
}
// CHECK: %[[OPERAND:.*]] = bufferization.to_memref %[[ARG]]
// CHECK: %[[SHAPE:.*]] = tensor.from_elements

// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
// CHECK: %[[OPER_DIM_1:.*]] = memref.dim %[[OPERAND]], %[[C1]] : memref<?x?xf32>
// CHECK: %[[OP_STRIDE_0:.*]] = arith.muli %[[C1]], %[[OPER_DIM_1]] : index
// CHECK: %[[OPER_DIM_0:.*]] = memref.dim %[[OPERAND]], %[[C0]] : memref<?x?xf32>

// CHECK: %[[EL0:.*]] = tensor.extract %[[SHAPE]][%[[C0]]] : tensor<3xi64>
// CHECK: %[[SIZE_0:.*]] = arith.index_cast %[[EL0]] : i64 to index
// CHECK: %[[EL1:.*]] = tensor.extract %[[SHAPE]][%[[C1]]] : tensor<3xi64>

// CHECK: %[[SIZE_1:.*]] = arith.index_cast %[[EL1]] : i64 to index
// CHECK: %[[EXPAND_1:.*]] = arith.cmpi slt, %[[OPER_DIM_0]], %[[SIZE_1]] : index
// CHECK: %[[STRIDE_1:.*]] = arith.select %[[EXPAND_1]], %[[C0]], %[[OP_STRIDE_0]] : index

// CHECK: %[[C2:.*]] = arith.constant 2 : index
// CHECK: %[[EL2:.*]] = tensor.extract %[[SHAPE]][%[[C2]]] : tensor<3xi64>
// CHECK: %[[SIZE_2:.*]] = arith.index_cast %[[EL2]] : i64 to index
// CHECK: %[[EXPAND_2:.*]] = arith.cmpi slt, %[[OPER_DIM_1]], %[[SIZE_2]] : index
// CHECK: %[[STRIDE_2:.*]] = arith.select %[[EXPAND_2]], %[[C0]], %[[C1]] : index

// CHECK: %[[TRANSFORMED_MEMREF:.*]] = memref.reinterpret_cast %[[OPERAND]] to offset: [0], sizes: [%[[SIZE_0]], %[[SIZE_1]], %[[SIZE_2]]], strides: [%[[C0]], %[[STRIDE_1]], %[[STRIDE_2]]] : memref<?x?xf32> to memref<?x?x?xf32, #map>
// CHECK: %[[ALLOC:.*]] = memref.alloc
// CHECK: memref.copy %[[TRANSFORMED_MEMREF]], %[[ALLOC]] : memref<?x?x?xf32, #map> to memref<?x?x?xf32>

// CHECK: %[[RESULT:.*]] = bufferization.to_tensor %[[ALLOC]]

// CHECK: return %[[RESULT]]

// -----

// CHECK-LABEL: func @dyn_broadcast_unsigned
func @dyn_broadcast_unsigned(%operand: tensor<?x?xui32>) -> tensor<?x?x?xui32> {
  // CHECK-SAME: %[[ARG:.*]]: tensor<?x?xui32>
  %c1 = arith.constant 1 : i64
  %shape = tensor.from_elements %c1, %c1, %c1 : tensor<3xi64>
  %result = "mhlo.dynamic_broadcast_in_dim"(%operand, %shape) {
    broadcast_dimensions = dense<[1, 2]> : tensor<2xi64>
  } : (tensor<?x?xui32>, tensor<3xi64>) -> tensor<?x?x?xui32>
  return %result : tensor<?x?x?xui32>
}

// CHECK: %[[OPERAND:.*]] = bufferization.to_memref %[[ARG]]
// CHECK: %[[SHAPE:.*]] = tensor.from_elements
// CHECK: %[[COPERAND:.*]] = builtin.unrealized_conversion_cast %[[OPERAND]] : memref<?x?xui32> to memref<?x?xi32>

// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
// CHECK: %[[OPER_DIM_1:.*]] = memref.dim %[[COPERAND]], %[[C1]] : memref<?x?xi32>
// CHECK: %[[OP_STRIDE_0:.*]] = arith.muli %[[C1]], %[[OPER_DIM_1]] : index
// CHECK: %[[OPER_DIM_0:.*]] = memref.dim %[[COPERAND]], %[[C0]] : memref<?x?xi32>

// CHECK: %[[EL0:.*]] = tensor.extract %[[SHAPE]][%[[C0]]] : tensor<3xi64>
// CHECK: %[[SIZE_0:.*]] = arith.index_cast %[[EL0]] : i64 to index
// CHECK: %[[EL1:.*]] = tensor.extract %[[SHAPE]][%[[C1]]] : tensor<3xi64>

// CHECK: %[[SIZE_1:.*]] = arith.index_cast %[[EL1]] : i64 to index
// CHECK: %[[EXPAND_1:.*]] = arith.cmpi slt, %[[OPER_DIM_0]], %[[SIZE_1]] : index
// CHECK: %[[STRIDE_1:.*]] = arith.select %[[EXPAND_1]], %[[C0]], %[[OP_STRIDE_0]] : index

// CHECK: %[[C2:.*]] = arith.constant 2 : index
// CHECK: %[[EL2:.*]] = tensor.extract %[[SHAPE]][%[[C2]]] : tensor<3xi64>
// CHECK: %[[SIZE_2:.*]] = arith.index_cast %[[EL2]] : i64 to index
// CHECK: %[[EXPAND_2:.*]] = arith.cmpi slt, %[[OPER_DIM_1]], %[[SIZE_2]] : index
// CHECK: %[[STRIDE_2:.*]] = arith.select %[[EXPAND_2]], %[[C0]], %[[C1]] : index

// CHECK: %[[TRANSFORMED_MEMREF:.*]] = memref.reinterpret_cast %[[COPERAND]] to offset: [0], sizes: [%[[SIZE_0]], %[[SIZE_1]], %[[SIZE_2]]], strides: [%[[C0]], %[[STRIDE_1]], %[[STRIDE_2]]] : memref<?x?xi32> to memref<?x?x?xi32, #map>
// CHECK: %[[ALLOC:.*]] = memref.alloc
// CHECK: memref.copy %[[TRANSFORMED_MEMREF]], %[[ALLOC]] : memref<?x?x?xi32, #map> to memref<?x?x?xi32>
// CHECK: %[[CALLOC:.*]] = builtin.unrealized_conversion_cast %[[ALLOC]] : memref<?x?x?xi32> to memref<?x?x?xui32>

// CHECK: %[[RESULT:.*]] = bufferization.to_tensor %[[CALLOC]]

// CHECK: return %[[RESULT]]

// -----

// CHECK-LABEL: func @dyn_reshape_unsigned
func @dyn_reshape_unsigned(%operand: tensor<?x?xui32>) -> tensor<?x?x?xui32> {
  // CHECK-SAME: %[[ARG:.*]]: tensor<?x?xui32>
  %c1 = arith.constant 1 : i64
  %shape = tensor.from_elements %c1, %c1, %c1 : tensor<3xi64>
  %result = "mhlo.dynamic_reshape"(%operand, %shape) : (tensor<?x?xui32>, tensor<3xi64>) -> tensor<?x?x?xui32>
  return %result : tensor<?x?x?xui32>
}

// CHECK: %[[OPERAND:.*]] = bufferization.to_memref %[[ARG]]
// CHECK: %[[SHAPE:.*]] = tensor.from_elements
// CHECK: %[[BSHAPE:.*]] = bufferization.to_memref %[[SHAPE]]
// CHECK: %[[COPERAND:.*]] = builtin.unrealized_conversion_cast %[[OPERAND]] : memref<?x?xui32> to memref<?x?xi32>

// CHECK: %[[RESHAPED:.*]] = memref.reshape %[[COPERAND]](%[[BSHAPE]]) : (memref<?x?xi32>, memref<3xi64>) -> memref<?x?x?xi32>
// CHECK: %[[RESULT:.*]] =  builtin.unrealized_conversion_cast %[[RESHAPED]] : memref<?x?x?xi32> to memref<?x?x?xui32>
// CHECK: %[[TRESULT:.*]] = bufferization.to_tensor %[[RESULT]] : memref<?x?x?xui32>
// CHECK: return %[[TRESULT]]

// -----

// CHECK-LABEL: func @reshape_unsigned
func @reshape_unsigned(%operand: tensor<*xui32>) -> tensor<4x3xui32> {
  // CHECK-SAME: %[[ARG:.*]]: tensor<*xui32>
  %result = "mhlo.reshape"(%operand) : (tensor<*xui32>) -> tensor<4x3xui32>
  return %result : tensor<4x3xui32>
}

// CHECK: %[[OPERAND:.*]] = bufferization.to_memref %[[ARG]]
// CHECK: %[[COPERAND:.*]] = builtin.unrealized_conversion_cast %[[OPERAND]] : memref<*xui32> to memref<*xi32>
// CHECK: %[[RESHAPED:.*]] = memref.cast %[[COPERAND]] : memref<*xi32> to memref<4x3xi32>
// CHECK: %[[RESULT:.*]] = builtin.unrealized_conversion_cast %[[RESHAPED]] : memref<4x3xi32> to memref<4x3xui32>
// CHECK: %[[TRESULT:.*]] = bufferization.to_tensor %[[RESULT]] : memref<4x3xui32>
// CHECK: return %[[TRESULT]]

// -----

func @reshape_unsigned() -> tensor<2xui64> {
  %result = mhlo.xla.rng_get_and_update_state {delta = 1 : i64}
  return %result : tensor<2xui64>
}

// CHECK:           memref.global "private" @rng_state : memref<i128>
// CHECK-LABEL:     func @reshape_unsigned
// CHECK:             %[[GLOBAL:.*]] = memref.get_global @rng_state : memref<i128>
// CHECK:             %[[OLD_SEED:.*]] = memref.load %[[GLOBAL]][] : memref<i128>
// CHECK:             %[[DELTA:.*]] = arith.constant 1 : i128
// CHECK:             %[[NEW_SEED:.*]] = arith.addi %[[OLD_SEED]], %[[DELTA]] : i128
// CHECK:             memref.store %[[NEW_SEED]], %[[GLOBAL]][] : memref<i128>
// CHECK:             %[[C64:.*]] = arith.constant 64 : i128
// CHECK:             %[[UPPER_BITS:.*]] = arith.shrui %[[OLD_SEED]], %[[C64]] : i128
// CHECK:             %[[UPPER_WORD:.*]] = arith.trunci %[[UPPER_BITS]] : i128 to i64
// CHECK:             %[[C0:.*]] = arith.constant 0 : i128
// CHECK:             %[[LOWER_BITS:.*]] = arith.shrui %[[OLD_SEED]], %[[C0]] : i128
// CHECK:             %[[LOWER_WORD:.*]] = arith.trunci %[[LOWER_BITS]] : i128 to i64
// CHECK:             %[[PACKED_WORDS:.*]] = tensor.from_elements %[[UPPER_WORD]], %[[LOWER_WORD]] : tensor<2xi64>
// CHECK:             %[[CASTED_RESULT:.*]] = builtin.unrealized_conversion_cast %[[PACKED_WORDS]] : tensor<2xi64> to tensor<2xui64>
// CHECK:             return %[[CASTED_RESULT]] : tensor<2xui64>
