// RUN: mlir-hlo-opt %s \
// RUN:   --verify-diagnostics \
// RUN:   --hlo-legalize-to-linalg \
// RUN:   --canonicalize | FileCheck %s

// Verifies that different sparse input and output types are
// properly dealt with while lowering mhlo ops to linalg ops.

#CSR = #sparse_tensor.encoding<{
  dimLevelType = ["dense", "compressed"]
}>

#DCSR = #sparse_tensor.encoding<{
  dimLevelType = ["compressed", "compressed"]
}>

#ST = #sparse_tensor.encoding<{
  dimLevelType = ["compressed", "compressed", "compressed"]
}>

// CHECK-LABEL: func @sparse_abs_eltwise(
// CHECK-SAME:    %[[ARG0:.*]]: tensor<10x20xf32, #{{.*}}>) -> tensor<10x20xf32, #{{.*}}> {
// CHECK-DAG:     %[[C10:.*]] = arith.constant 10 : index
// CHECK-DAG:     %[[C20:.*]] = arith.constant 20 : index
// CHECK:         %[[OUT:.*]] = sparse_tensor.init{{\[}}%[[C10]], %[[C20]]] : tensor<10x20xf32, #{{.*}}>
// CHECK:         %[[VAL:.*]] = linalg.generic {{{.*}} ins(%[[ARG0]] : tensor<10x20xf32, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], {{.*}} }>>) outs(%[[OUT]] : tensor<10x20xf32, #sparse_tensor.encoding<{ dimLevelType = [ "compressed", "compressed" ], {{.*}} }>>)
// CHECK:         ^bb0(%[[A:.*]]: f32, %[[B:.*]]: f32):
// CHECK:           %[[ABS:.*]] = math.abs %[[A]] : f32
// CHECK:           linalg.yield %[[ABS]] : f32
// CHECK:         } -> tensor<10x20xf32, #{{.*}}>
// CHECK:         return %[[VAL:.*]] : tensor<10x20xf32, #{{.*}}>
// CHECK:       }
func.func @sparse_abs_eltwise(%arg0: tensor<10x20xf32, #CSR>)
                                  -> tensor<10x20xf32, #DCSR> {
  %0 = "mhlo.abs"(%arg0) : (tensor<10x20xf32, #CSR>)
                         -> tensor<10x20xf32, #DCSR>
  func.return %0 : tensor<10x20xf32, #DCSR>
}

// CHECK-LABEL: func @sparse_add_eltwise(
// CHECK-SAME:    %[[ARG0:.*]]: tensor<10x20xf32, #{{.*}}>,
// CHECK-SAME:    %[[ARG1:.*]]: tensor<10x20xf32, #{{.*}}>) -> tensor<10x20xf32, #{{.*}}> {
// CHECK-DAG:     %[[C10:.*]] = arith.constant 10 : index
// CHECK-DAG:     %[[C20:.*]] = arith.constant 20 : index
// CHECK:         %[[OUT:.*]] = sparse_tensor.init{{\[}}%[[C10]], %[[C20]]] : tensor<10x20xf32, #{{.*}}>
// CHECK:         %[[VAL:.*]] = linalg.generic {{{.*}}} ins(%[[ARG0]], %[[ARG1]] : tensor<10x20xf32, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], {{.*}} }>>, tensor<10x20xf32, #sparse_tensor.encoding<{ dimLevelType = [ "compressed", "compressed" ], {{.*}} }>>) outs(%[[OUT]] : tensor<10x20xf32, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], {{.*}} }>>) {
// CHECK:           ^bb0(%[[A:.*]]: f32, %[[B:.*]]: f32, %[[C:.*]]: f32):
// CHECK:             %[[ADD:.*]] = arith.addf %[[A]], %[[B]] : f32
// CHECK:             linalg.yield %[[ADD]] : f32
// CHECK:         } -> tensor<10x20xf32, #{{.*}}>
// CHECK:         return %[[VAL:.*]] : tensor<10x20xf32, #{{.*}}>
// CHECK:       }
func.func @sparse_add_eltwise(%arg0: tensor<10x20xf32, #CSR>,
                              %arg1: tensor<10x20xf32, #DCSR>)
                                  -> tensor<10x20xf32, #CSR> {
  %0 = mhlo.add (%arg0, %arg1) : (tensor<10x20xf32, #CSR>,
                                  tensor<10x20xf32, #DCSR>)
                               -> tensor<10x20xf32, #CSR>
  func.return %0 : tensor<10x20xf32, #CSR>
}

// CHECK-LABEL: func @sparse_math(
// CHECK-SAME:    %[[ARG0:.*]]: tensor<10x20x30xf64, #{{.*}}>) -> tensor<10x20x30xf64, #{{.*}}> {
// CHECK:         %[[T0:.*]] = linalg.generic {{{.*}}} ins(%[[ARG0]] : tensor<10x20x30xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed", "compressed", "compressed" ]
// CHECK:            math.abs
// CHECK:         }
// CHECK:         %[[T1:.*]] = linalg.generic {{{.*}}} ins(%[[T0]] : tensor<10x20x30xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed", "compressed", "compressed" ]
// CHECK:            math.expm1
// CHECK:         }
// CHECK:         %[[T2:.*]] = linalg.generic {{{.*}}} ins(%[[T1]] : tensor<10x20x30xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed", "compressed", "compressed" ]
// CHECK:           math.log1p
// CHECK:         }
// CHECK:         %[[T3:.*]] = linalg.generic {{{.*}}} ins(%[[T2]] : tensor<10x20x30xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed", "compressed", "compressed" ]
// CHECK:           arith.negf
// CHECK:         }
// CHECK:         %[[T4:.*]] = linalg.generic {{{.*}}} ins(%[[T3]] : tensor<10x20x30xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed", "compressed", "compressed" ]
// CHECK:           math.copysign
// CHECK:         }
// CHECK:         %[[T5:.*]] = linalg.generic {{{.*}}} ins(%[[T4]] : tensor<10x20x30xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed", "compressed", "compressed" ]
// CHECK:           math.sin
// CHECK:         }
// CHECK:         %[[T6:.*]] = linalg.generic {{{.*}}} ins(%[[T5]] : tensor<10x20x30xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed", "compressed", "compressed" ]
// CHECK:           math.sqrt
// CHECK:         }
// CHECK:         %[[T7:.*]] = linalg.generic {{{.*}}} ins(%[[T6]] : tensor<10x20x30xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed", "compressed", "compressed" ]
// CHECK:           math.tanh
// CHECK:         }
// CHECK:         return %[[T7]] : tensor<10x20x30xf64, #{{.*}}>
// CHECK:       }
func.func @sparse_math(%arg0: tensor<10x20x30xf64, #ST>) -> tensor<10x20x30xf64, #ST> {
  %0 = mhlo.abs(%arg0) : (tensor<10x20x30xf64, #ST>) -> tensor<10x20x30xf64, #ST>
  %1 = mhlo.exponential_minus_one(%0) : (tensor<10x20x30xf64, #ST>) -> tensor<10x20x30xf64, #ST>
  %2 = mhlo.log_plus_one(%1) : (tensor<10x20x30xf64, #ST>) -> tensor<10x20x30xf64, #ST>
  %3 = mhlo.negate(%2) : (tensor<10x20x30xf64, #ST>) -> tensor<10x20x30xf64, #ST>
  %4 = mhlo.sign(%3) : (tensor<10x20x30xf64, #ST>) -> tensor<10x20x30xf64, #ST>
  %5 = mhlo.sine(%4) : (tensor<10x20x30xf64, #ST>) -> tensor<10x20x30xf64, #ST>
  %6 = mhlo.sqrt(%5) : (tensor<10x20x30xf64, #ST>) -> tensor<10x20x30xf64, #ST>
  %7 = mhlo.tanh(%6) : (tensor<10x20x30xf64, #ST>) -> tensor<10x20x30xf64, #ST>
  return %7 : tensor<10x20x30xf64, #ST>
}
