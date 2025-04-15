// RUN: mlir-hlo-opt -resolve-shaped-type-result-dims -canonicalize \
// RUN: -split-input-file %s -o - | FileCheck %s

// CHECK-LABEL: @dynamic_broadcast_i32_shape
func.func @dynamic_broadcast_i32_shape(%arg0 : tensor<3xi32>, %arg1 : tensor<?xf32>)
     -> index {
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %[[EXTRACT:.*]] = tensor.extract %arg0[%[[C0]]] : tensor<3xi32>
  // CHECK: %[[CAST:.*]] = arith.index_cast %[[EXTRACT]] : i32 to index
  // CHECK: return %[[CAST]]
  %c0 = arith.constant 0 : index
  %0 = "mhlo.dynamic_broadcast_in_dim"(%arg1, %arg0)
       { broadcast_dimensions = dense<0> : tensor<1xi64> }
     : (tensor<?xf32>, tensor<3xi32>) -> tensor<?x?x?xf32>
  %1 = tensor.dim %0, %c0 : tensor<?x?x?xf32>
  func.return %1 : index
}

// -----

// CHECK-LABEL: @dynamic_iota_i32_shape
func.func @dynamic_iota_i32_shape(%arg0 : tensor<3xi32>) -> index {
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %[[EXTRACT:.*]] = tensor.extract %arg0[%[[C0]]] : tensor<3xi32>
  // CHECK: %[[CAST:.*]] = arith.index_cast %[[EXTRACT]] : i32 to index
  // CHECK: return %[[CAST]]
  %c0 = arith.constant 0 : index
  %0 = "mhlo.dynamic_iota"(%arg0)
       {iota_dimension = 0 : i64}
     : (tensor<3xi32>) -> tensor<?x?x?xi32>
  %1 = tensor.dim %0, %c0 : tensor<?x?x?xi32>
  func.return %1 : index
}

// -----

// CHECK-LABEL: @dynamic_reshape_i32_shape
func.func @dynamic_reshape_i32_shape(%arg0 : tensor<3xi32>, %arg1 : tensor<?xf32>)
     -> index {
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %[[EXTRACT:.*]] = tensor.extract %arg0[%[[C0]]] : tensor<3xi32>
  // CHECK: %[[CAST:.*]] = arith.index_cast %[[EXTRACT]] : i32 to index
  // CHECK: return %[[CAST]]
  %c0 = arith.constant 0 : index
  %0 = "mhlo.dynamic_reshape"(%arg1, %arg0)
       { broadcast_dimensions = dense<0> : tensor<1xi64> }
     : (tensor<?xf32>, tensor<3xi32>) -> tensor<?x?x?xf32>
  %1 = tensor.dim %0, %c0 : tensor<?x?x?xf32>
  func.return %1 : index
}
