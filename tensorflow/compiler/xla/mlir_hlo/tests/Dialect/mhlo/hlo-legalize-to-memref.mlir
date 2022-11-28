// RUN: mlir-hlo-opt -hlo-legalize-to-memref -canonicalize -cse --split-input-file %s | FileCheck %s

// CHECK: #[[MAP:.*]] = affine_map<(d0, d1, d2)[s0, s1, s2] -> (d0 * s0 + d1 * s1 + d2 * s2)>

// CHECK-LABEL: func @dyn_broadcast
func.func @dyn_broadcast(%operand: tensor<?x?xf32>) -> tensor<?x?x?xf32> {
  // CHECK-SAME: %[[ARG:.*]]: tensor<?x?xf32>
  %c1 = arith.constant 1 : i64
  %shape = tensor.from_elements %c1, %c1, %c1 : tensor<3xi64>
  %result = "mhlo.dynamic_broadcast_in_dim"(%operand, %shape) {
    broadcast_dimensions = dense<[1, 2]> : tensor<2xi64>
  } : (tensor<?x?xf32>, tensor<3xi64>) -> tensor<?x?x?xf32>
  func.return %result : tensor<?x?x?xf32>
}

// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[OPERAND:.*]] = bufferization.to_memref %[[ARG]]

// CHECK: %[[OPER_DIM_1:.*]] = tensor.dim %[[ARG]], %[[C1]] : tensor<?x?xf32>
// CHECK: %[[OPER_DIM_0:.*]] = tensor.dim %[[ARG]], %[[C0]] : tensor<?x?xf32>
// CHECK: %[[EXPAND_1:.*]] = arith.cmpi slt, %[[OPER_DIM_0]], %[[C1]] : index
// CHECK: %[[STRIDE_1:.*]] = arith.select %[[EXPAND_1]], %[[C0]], %[[OPER_DIM_1]] : index
// CHECK: %[[EXPAND_2:.*]] = arith.cmpi slt, %[[OPER_DIM_1]], %[[C1]] : index
// CHECK: %[[STRIDE_2:.*]] = arith.select %[[EXPAND_2]], %[[C0]], %[[C1]] : index

// CHECK: %[[TRANSFORMED_MEMREF:.*]] = memref.reinterpret_cast %[[OPERAND]] to offset: [0], sizes: [%[[C1]], %[[C1]], %[[C1]]], strides: [%[[C0]], %[[STRIDE_1]], %[[STRIDE_2]]] : memref<?x?xf32> to memref<?x?x?xf32, #map{{[0-9]*}}>
// CHECK: %[[RESULT:.*]] = bufferization.to_tensor %[[TRANSFORMED_MEMREF]]

// CHECK: return %[[RESULT]]

// -----

// CHECK-LABEL: func @dyn_broadcast_unsigned
func.func @dyn_broadcast_unsigned(%operand: tensor<?x?xi32>, %shape: tensor<3xi64>) -> tensor<?x?x?xi32> {
  // CHECK-SAME: %[[ARG:.*]]: tensor<?x?xi32>, %[[SHAPE:.*]]: tensor<3xi64>
  %c1 = arith.constant 1 : i64
  %result = "mhlo.dynamic_broadcast_in_dim"(%operand, %shape) {
    broadcast_dimensions = dense<[1, 2]> : tensor<2xi64>
  } : (tensor<?x?xi32>, tensor<3xi64>) -> tensor<?x?x?xi32>
  func.return %result : tensor<?x?x?xi32>
}

// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[C2:.*]] = arith.constant 2 : index

// CHECK-DAG: %[[OPERAND:.*]] = bufferization.to_memref %[[ARG]]

// CHECK: %[[OPER_DIM_1:.*]] = tensor.dim %[[ARG]], %[[C1]] : tensor<?x?xi32>
// CHECK: %[[OPER_DIM_0:.*]] = tensor.dim %[[ARG]], %[[C0]] : tensor<?x?xi32>

// CHECK: %[[EL0:.*]] = tensor.extract %[[SHAPE]][%[[C0]]] : tensor<3xi64>
// CHECK: %[[SIZE_0:.*]] = arith.index_cast %[[EL0]] : i64 to index
// CHECK: %[[EL1:.*]] = tensor.extract %[[SHAPE]][%[[C1]]] : tensor<3xi64>

// CHECK: %[[SIZE_1:.*]] = arith.index_cast %[[EL1]] : i64 to index
// CHECK: %[[EXPAND_1:.*]] = arith.cmpi slt, %[[OPER_DIM_0]], %[[SIZE_1]] : index
// CHECK: %[[STRIDE_1:.*]] = arith.select %[[EXPAND_1]], %[[C0]], %[[OPER_DIM_1]] : index

// CHECK: %[[EL2:.*]] = tensor.extract %[[SHAPE]][%[[C2]]] : tensor<3xi64>
// CHECK: %[[SIZE_2:.*]] = arith.index_cast %[[EL2]] : i64 to index
// CHECK: %[[EXPAND_2:.*]] = arith.cmpi slt, %[[OPER_DIM_1]], %[[SIZE_2]] : index
// CHECK: %[[STRIDE_2:.*]] = arith.select %[[EXPAND_2]], %[[C0]], %[[C1]] : index

// CHECK: %[[TRANSFORMED_MEMREF:.*]] = memref.reinterpret_cast %[[OPERAND]] to offset: [0], sizes: [%[[SIZE_0]], %[[SIZE_1]], %[[SIZE_2]]], strides: [%[[C0]], %[[STRIDE_1]], %[[STRIDE_2]]] : memref<?x?xi32> to memref<?x?x?xi32, #map{{[0-9]*}}>

// CHECK: %[[RESULT:.*]] = bufferization.to_tensor %[[TRANSFORMED_MEMREF]]

// CHECK: return %[[RESULT]]

// -----

// CHECK-LABEL: func @dyn_reshape_unsigned
func.func @dyn_reshape_unsigned(%operand: tensor<?x?xi32>, %shape: tensor<3xi64>) -> tensor<?x?x?xi32> {
  // CHECK-SAME: %[[ARG:.*]]: tensor<?x?xi32>, %[[SHAPE:.*]]: tensor<3xi64>
  %c1 = arith.constant 1 : i64
  %result = "mhlo.dynamic_reshape"(%operand, %shape) : (tensor<?x?xi32>, tensor<3xi64>) -> tensor<?x?x?xi32>
  func.return %result : tensor<?x?x?xi32>
}

// CHECK-DAG: %[[OPERAND:.*]] = bufferization.to_memref %[[ARG]]
// CHECK-DAG: %[[BSHAPE:.*]] = bufferization.to_memref %[[SHAPE]]

// CHECK: %[[RESHAPED:.*]] = memref.reshape %[[OPERAND]](%[[BSHAPE]]) : (memref<?x?xi32>, memref<3xi64>) -> memref<?x?x?xi32>
// CHECK: %[[TRESULT:.*]] = bufferization.to_tensor %[[RESHAPED]] : memref<?x?x?xi32>
// CHECK: return %[[TRESULT]]

// -----

// CHECK-LABEL: func @reshape_unsigned
func.func @reshape_unsigned(%operand: tensor<*xi32>) -> tensor<4x3xi32> {
  // CHECK-SAME: %[[ARG:.*]]: tensor<*xi32>
  %result = "mhlo.reshape"(%operand) : (tensor<*xi32>) -> tensor<4x3xi32>
  func.return %result : tensor<4x3xi32>
}

// CHECK: %[[OPERAND:.*]] = bufferization.to_memref %[[ARG]]
// CHECK: %[[RESHAPED:.*]] = memref.cast %[[OPERAND]] : memref<*xi32> to memref<4x3xi32>
// CHECK: %[[TRESULT:.*]] = bufferization.to_tensor %[[RESHAPED]] : memref<4x3xi32>
// CHECK: return %[[TRESULT]]

// -----

// CHECK-LABEL: func @custom_call_multiple_inputs_outputs
// CHECK-SAME: %[[ARG0:.*]]: tensor<2xf32>
// CHECK-SAME: %[[ARG1:.*]]: tensor<5xi32>
func.func @custom_call_multiple_inputs_outputs(%x: tensor<2xf32>,
    %y: tensor<5xi32>) -> (tensor<2xf32>, tensor<2xf32>) {
  %0:3 = "mhlo.custom_call"(%x, %y) {
    backend_config="",
    call_target_name = "foo",
    has_side_effect = false
  } : (tensor<2xf32>, tensor<5xi32>)
    -> (tensor<2xf32>, tensor<2xf32>, tensor<5xi32>)
  func.return %0#0, %0#1 : tensor<2xf32>, tensor<2xf32>
}

// CHECK-DAG: %[[I0:.+]] = bufferization.to_memref %[[ARG0]] : memref<2xf32>
// CHECK-DAG: %[[I1:.+]] = bufferization.to_memref %[[ARG1]] : memref<5xi32>
// CHECK-DAG: %[[O0:.*]] = memref.alloc() {alignment = 128 : i64} : memref<2xf32>
// CHECK-DAG: %[[O1:.*]] = memref.alloc() {alignment = 128 : i64} : memref<2xf32>
// CHECK-DAG: %[[O2:.*]] = memref.alloc() {alignment = 128 : i64} : memref<5xi32>
// CHECK: "lmhlo.custom_call"(%[[I0]], %[[I1]], %[[O0]], %[[O1]], %[[O2]]) {backend_config = "", call_target_name = "foo", has_side_effect = false, operand_segment_sizes = array<i32: 2, 3>} : (memref<2xf32>, memref<5xi32>, memref<2xf32>, memref<2xf32>, memref<5xi32>) -> ()
// CHECK-DAG: %[[T0:.+]] = bufferization.to_tensor %[[O0]] : memref<2xf32>
// CHECK-DAG: %[[T1:.+]] = bufferization.to_tensor %[[O1]] : memref<2xf32>
// CHECK: return %[[T0]], %[[T1]] : tensor<2xf32>, tensor<2xf32>

// -----

// CHECK-LABEL: func @custom_call_side_effect
// CHECK-SAME: %[[ARG0:.*]]: tensor<2xf32>
// CHECK-SAME: %[[ARG1:.*]]: tensor<5xi32>
func.func @custom_call_side_effect(%x: tensor<2xf32>,
                                   %y: tensor<5xi32>) -> !mhlo.token {
  %token = mhlo.create_token : !mhlo.token
  %0:2 = "mhlo.custom_call"(%x, %y, %token) {
    backend_config = "",
    call_target_name = "bar",
    has_side_effect = true
  } : (tensor<2xf32>, tensor<5xi32>, !mhlo.token)
    -> (!mhlo.token, tensor<2xi32>)
  func.return %0#0 : !mhlo.token
}

// CHECK-DAG: %[[TOKEN:.*]] = mhlo.create_token
// CHECK-DAG: %[[I0:.+]] = bufferization.to_memref %[[ARG0]] : memref<2xf32>
// CHECK-DAG: %[[I1:.+]] = bufferization.to_memref %[[ARG1]] : memref<5xi32>
// CHECK-DAG: %[[ALLOC:.+]] = memref.alloc
// CHECK: "lmhlo.custom_call"(%[[I0]], %[[I1]], %[[ALLOC]]) {backend_config = "", call_target_name = "bar", has_side_effect = true, operand_segment_sizes = array<i32: 2, 1>, target_arg_mapping = #lmhlo.custom_call_target_arg_mapping<num_args = 3, num_results = 2, args_to_target_args = [0, 1], results_to_target_results = [1]>} : (memref<2xf32>, memref<5xi32>, memref<2xi32>)
// CHECK: return %[[TOKEN]] : !mhlo.token

// -----

// CHECK-LABEL: func @infeed_outfeed
func.func @infeed_outfeed(%arg0: tensor<f32>) {
  %0 = mhlo.create_token : !mhlo.token
  %1:2 = "mhlo.infeed"(%0) {infeed_config = "", layout = [[1, 0]]} : (!mhlo.token) -> (tensor<3x4xf32>, !mhlo.token)
// CHECK: %[[ALLOC:.*]] = memref.alloc() {alignment = 128 : i64} : memref<3x4xf32>
// CHECK: "lmhlo.infeed"(%[[ALLOC]]) {config = "", infeed_config = "", layout = {{\[}}[1, 0]]} : (memref<3x4xf32>) -> ()
  %2 = "mhlo.outfeed"(%1#0, %1#1) {outfeed_config = ""} : (tensor<3x4xf32>, !mhlo.token) -> !mhlo.token
// CHECK: "lmhlo.outfeed"(%[[ALLOC]]) {config = "", outfeed_config = ""} : (memref<3x4xf32>) -> ()
  func.return
}
