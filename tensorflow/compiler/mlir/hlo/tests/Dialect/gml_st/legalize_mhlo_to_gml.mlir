// RUN: mlir-hlo-opt %s --legalize-mhlo-to-gml | FileCheck %s

// CHECK:      @dynamic_broadcast_in_dim
// CHECK-SAME: %[[ARG:.*]]: tensor<?x?xf32>, %[[SHAPE:.*]]: tensor<3xindex>
func.func @dynamic_broadcast_in_dim(%arg : tensor<?x?xf32>, %shape : tensor<3xindex>) -> tensor<?x?x?xf32> {
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0
  // CHECK-DAG: %[[C1:.*]] = arith.constant 1
  // CHECK-DAG: %[[C2:.*]] = arith.constant 2
  // CHECK-DAG: %[[SHAPE_D0:.*]] = tensor.extract %[[SHAPE]][%[[C0]]]
  // CHECK-DAG: %[[SHAPE_D1:.*]] = tensor.extract %[[SHAPE]][%[[C1]]]
  // CHECK-DAG: %[[SHAPE_D2:.*]] = tensor.extract %[[SHAPE]][%[[C2]]]
  // CHECK-DAG: %[[INIT:.*]] = linalg.init_tensor [%[[SHAPE_D0]], %[[SHAPE_D1]], %[[SHAPE_D2]]] : tensor<?x?x?xf32>
  // CHECK-NEXT: %[[BCAST:.*]] = gml_st.dynamic_broadcast_in_dim
  // CHECK-SAME: ins(%[[ARG]] : tensor<?x?xf32>)
  // CHECK-SAME: outs(%[[INIT]] : tensor<?x?x?xf32>)
  // CHECK-SAME: {broadcast_dimensions = dense<[0, 2]> : tensor<2xi64>}
  // CHECK:     return %[[BCAST]]
  %0 = "mhlo.dynamic_broadcast_in_dim"(%arg, %shape) 
      { broadcast_dimensions = dense<[0, 2]> : tensor<2xi64> } 
      : (tensor<?x?xf32>, tensor<3xindex>) -> tensor<?x?x?xf32>
  func.return %0 : tensor<?x?x?xf32>
} 
