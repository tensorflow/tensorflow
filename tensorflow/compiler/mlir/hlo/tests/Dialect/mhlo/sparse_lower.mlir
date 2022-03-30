// RUN: mlir-hlo-opt %s -verify-diagnostics --hlo-legalize-to-linalg --canonicalize | FileCheck %s

// Verifies that different sparse input and output types are
// properly dealt with while lowering mhlo ops to linalg ops.

#CSR = #sparse_tensor.encoding<{
  dimLevelType = ["dense", "compressed"]
}>

#DCSR = #sparse_tensor.encoding<{
  dimLevelType = ["compressed", "compressed"]
}>

// CHECK-LABEL: func @sparse_abs_eltwise(
// CHECK-SAME:    %[[VAL_0:.*]]: tensor<10x20xf32, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], {{.*}} }>>) -> tensor<10x20xf32, #sparse_tensor.encoding<{ dimLevelType = [ "compressed", "compressed" ], {{.*}} }>> {
// CHECK-DAG:     %[[VAL_1:.*]] = arith.constant 10 : index
// CHECK-DAG:     %[[VAL_2:.*]] = arith.constant 20 : index
// CHECK-DAG:     %[[VAL_3:.*]] = sparse_tensor.init{{\[}}%[[VAL_1]], %[[VAL_2]]] : tensor<10x20xf32, #sparse_tensor.encoding<{ dimLevelType = [ "compressed", "compressed" ], pointerBitWidth = 0, indexBitWidth = 0 }>>
// CHECK:         %[[VAL_4:.*]] = linalg.generic {{{.*}} ins(%[[VAL_0]] : tensor<10x20xf32, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], {{.*}} }>>) outs(%[[VAL_3]] : tensor<10x20xf32, #sparse_tensor.encoding<{ dimLevelType = [ "compressed", "compressed" ], {{.*}} }>>) {
// CHECK:         ^bb0(%[[VAL_5:.*]]: f32, %[[VAL_6:.*]]: f32):
// CHECK:           %[[VAL_7:.*]] = math.abs %[[VAL_5]] : f32
// CHECK:           linalg.yield %[[VAL_7]] : f32
// CHECK:         } -> tensor<10x20xf32, #sparse_tensor.encoding<{ dimLevelType = [ "compressed", "compressed" ], {{.*}} }>>
// CHECK:         return %[[VAL_8:.*]] : tensor<10x20xf32, #sparse_tensor.encoding<{ dimLevelType = [ "compressed", "compressed" ], {{.*}} }>>
// CHECK:       }
func.func @sparse_abs_eltwise(%arg0: tensor<10x20xf32, #CSR>)
                                  -> tensor<10x20xf32, #DCSR> {
  %0 = "mhlo.abs"(%arg0) : (tensor<10x20xf32, #CSR>)
                         -> tensor<10x20xf32, #DCSR>
  func.return %0 : tensor<10x20xf32, #DCSR>
}

// CHECK-LABEL:   func @sparse_add_eltwise(
// CHECK-SAME:      %[[VAL_0:.*]]: tensor<10x20xf32, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], {{.*}} }>>,
// CHECK-SAME:      %[[VAL_1:.*]]: tensor<10x20xf32, #sparse_tensor.encoding<{ dimLevelType = [ "compressed", "compressed" ], {{.*}} }>>) -> tensor<10x20xf32, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], {{.*}} }>> {
// CHECK-DAG:       %[[VAL_2:.*]] = arith.constant 20 : index
// CHECK-DAG:       %[[VAL_3:.*]] = arith.constant 10 : index
// CHECK-DAG:       %[[VAL_4:.*]] = sparse_tensor.init{{\[}}%[[VAL_3]], %[[VAL_2]]] : tensor<10x20xf32, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], {{.*}} }>>
// CHECK:           %[[VAL_5:.*]] = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_0]], %[[VAL_1]] : tensor<10x20xf32, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], {{.*}} }>>, tensor<10x20xf32, #sparse_tensor.encoding<{ dimLevelType = [ "compressed", "compressed" ], {{.*}} }>>) outs(%[[VAL_4]] : tensor<10x20xf32, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], {{.*}} }>>) {
// CHECK:           ^bb0(%[[VAL_6:.*]]: f32, %[[VAL_7:.*]]: f32, %[[VAL_8:.*]]: f32):
// CHECK:             %[[VAL_9:.*]] = arith.addf %[[VAL_6]], %[[VAL_7]] : f32
// CHECK:             linalg.yield %[[VAL_9]] : f32
// CHECK:           } -> tensor<10x20xf32, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], {{.*}} }>>
// CHECK:           return %[[VAL_10:.*]] : tensor<10x20xf32, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], {{.*}} }>>
// CHECK:         }
func.func @sparse_add_eltwise(%arg0: tensor<10x20xf32, #CSR>,
                              %arg1: tensor<10x20xf32, #DCSR>)
                                  -> tensor<10x20xf32, #CSR> {
  %0 = mhlo.add (%arg0, %arg1) : (tensor<10x20xf32, #CSR>,
                                  tensor<10x20xf32, #DCSR>)
                               -> tensor<10x20xf32, #CSR>
  func.return %0 : tensor<10x20xf32, #CSR>
}
