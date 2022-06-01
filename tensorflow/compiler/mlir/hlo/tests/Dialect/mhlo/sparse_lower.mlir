// RUN: mlir-hlo-opt %s \
// RUN:   --verify-diagnostics \
// RUN:   --hlo-legalize-to-linalg \
// RUN:   --canonicalize | FileCheck %s

// Verifies that different sparse input and output types are
// properly dealt with while lowering mhlo ops to linalg ops.

#SV = #sparse_tensor.encoding<{
  dimLevelType = ["compressed"]
}>

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

// CHECK-LABEL: func @sparse_mul_eltwise(
// CHECK-SAME:    %[[ARG0:.*]]: tensor<10x20xf32, #{{.*}}>,
// CHECK-SAME:    %[[ARG1:.*]]: tensor<10x20xf32, #{{.*}}>) -> tensor<10x20xf32, #{{.*}}> {
// CHECK-DAG:     %[[C10:.*]] = arith.constant 10 : index
// CHECK-DAG:     %[[C20:.*]] = arith.constant 20 : index
// CHECK:         %[[OUT:.*]] = sparse_tensor.init{{\[}}%[[C10]], %[[C20]]] : tensor<10x20xf32, #{{.*}}>
// CHECK:         %[[VAL:.*]] = linalg.generic {{{.*}}} ins(%[[ARG0]], %[[ARG1]] : tensor<10x20xf32, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], {{.*}} }>>, tensor<10x20xf32, #sparse_tensor.encoding<{ dimLevelType = [ "compressed", "compressed" ], {{.*}} }>>) outs(%[[OUT]] : tensor<10x20xf32, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], {{.*}} }>>) {
// CHECK:           ^bb0(%[[A:.*]]: f32, %[[B:.*]]: f32, %[[C:.*]]: f32):
// CHECK:             %[[ADD:.*]] = arith.mulf %[[A]], %[[B]] : f32
// CHECK:             linalg.yield %[[ADD]] : f32
// CHECK:         } -> tensor<10x20xf32, #{{.*}}>
// CHECK:         return %[[VAL:.*]] : tensor<10x20xf32, #{{.*}}>
// CHECK:       }
func.func @sparse_mul_eltwise(%arg0: tensor<10x20xf32, #CSR>,
                              %arg1: tensor<10x20xf32, #DCSR>)
                                  -> tensor<10x20xf32, #CSR> {
  %0 = mhlo.multiply (%arg0, %arg1) : (tensor<10x20xf32, #CSR>,
                                       tensor<10x20xf32, #DCSR>)
                                    -> tensor<10x20xf32, #CSR>
  func.return %0 : tensor<10x20xf32, #CSR>
}

// CHECK-LABEL: func @sparse_math(
// CHECK-SAME:    %[[ARG0:.*]]: tensor<10x20x30xf64, #{{.*}}>) -> tensor<10x20x30xf64, #{{.*}}> {
// CHECK:         %[[T0:.*]] = linalg.generic {{{.*}}} ins(%[[ARG0]] : tensor<10x20x30xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed", "compressed", "compressed" ], {{.*}} }>>) outs
// CHECK:            math.abs
// CHECK:         }
// CHECK:         %[[T1:.*]] = linalg.generic {{{.*}}} ins(%[[T0]] : tensor<10x20x30xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed", "compressed", "compressed" ], {{.*}} }>>) outs
// CHECK:            math.expm1
// CHECK:         }
// CHECK:         %[[T2:.*]] = linalg.generic {{{.*}}} ins(%[[T1]] : tensor<10x20x30xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed", "compressed", "compressed" ], {{.*}} }>>) outs
// CHECK:           math.log1p
// CHECK:         }
// CHECK:         %[[T3:.*]] = linalg.generic {{{.*}}} ins(%[[T2]] : tensor<10x20x30xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed", "compressed", "compressed" ], {{.*}} }>>) outs
// CHECK:           arith.negf
// CHECK:         }
// CHECK:         %[[T4:.*]] = linalg.generic {{{.*}}} ins(%[[T3]] : tensor<10x20x30xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed", "compressed", "compressed" ], {{.*}} }>>) outs
// CHECK:           sparse_tensor.unary %{{.*}} : f64 to f64
// CHECK:           present = {
// CHECK:             math.copysign
// CHECK:             sparse_tensor.yield %{{.*}} : f64
// CHECK:           }
// CHECK:           absent = {
// CHECK:           }
// CHECK:         }
// CHECK:         %[[T5:.*]] = linalg.generic {{{.*}}} ins(%[[T4]] : tensor<10x20x30xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed", "compressed", "compressed" ], {{.*}} }>>) outs
// CHECK:           math.sin
// CHECK:         }
// CHECK:         %[[T6:.*]] = linalg.generic {{{.*}}} ins(%[[T5]] : tensor<10x20x30xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed", "compressed", "compressed" ], {{.*}} }>>) outs
// CHECK:           math.sqrt
// CHECK:         }
// CHECK:         %[[T7:.*]] = linalg.generic {{{.*}}} ins(%[[T6]] : tensor<10x20x30xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed", "compressed", "compressed" ], {{.*}} }>>) outs
// CHECK:           math.tanh
// CHECK:         }
// CHECK:         %[[T8:.*]] = linalg.generic {{{.*}}} ins(%[[T7]] : tensor<10x20x30xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed", "compressed", "compressed" ], {{.*}} }>>) outs
// CHECK:           math.ceil
// CHECK:         }
// CHECK:         %[[T9:.*]] = linalg.generic {{{.*}}} ins(%[[T8]] : tensor<10x20x30xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed", "compressed", "compressed" ], {{.*}} }>>) outs
// CHECK:           math.floor
// CHECK:         }
// CHECK:         return %[[T9]] : tensor<10x20x30xf64, #{{.*}}>
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
  %8 = mhlo.ceil(%7) : (tensor<10x20x30xf64, #ST>) -> tensor<10x20x30xf64, #ST>
  %9 = mhlo.floor(%8) : (tensor<10x20x30xf64, #ST>) -> tensor<10x20x30xf64, #ST>
  func.return %9 : tensor<10x20x30xf64, #ST>
}

// CHECK-LABEL: func @sparse_sign(
// CHECK-SAME:    %[[A:.*]]: tensor<100xi32, #{{.*}}>) -> tensor<100xi32> {
// CHECK:         %[[T:.*]] = linalg.generic {{{.*}}} ins(%[[A]] : tensor<100xi32, #{{.*}}>)
// CHECK:           %[[U:.*]] = sparse_tensor.unary %{{.*}} : i32 to i32
// CHECK:           present = {
// CHECK:             arith.cmpi eq
// CHECK:             sparse_tensor.yield %{{.*}} : i32
// CHECK:           }
// CHECK:           absent = {
// CHECK:           }
// CHECK:           linalg.yield %[[U]] : i32
// CHECK:         } -> tensor<100xi32>
// CHECK:         return %[[T]] : tensor<100xi32>
// CHECK:       }
func.func @sparse_sign(%arg0: tensor<100xi32, #SV>) -> tensor<100xi32> {
  %0 = mhlo.sign(%arg0) : (tensor<100xi32, #SV>) -> tensor<100xi32>
  func.return %0 : tensor<100xi32>
}

// CHECK-LABEL: func @sparse_int_abs(
// CHECK-SAME:    %[[A:.*]]: tensor<100xi64, #{{.*}}>) -> tensor<100xi64> {
// CHECK:         %[[T:.*]] = linalg.generic {{{.*}}} ins(%[[A]] : tensor<100xi64, #{{.*}}>)
// CHECK:           %[[U:.*]] = sparse_tensor.unary %{{.*}} : i64 to i64
// CHECK:           present = {
// CHECK:             arith.cmpi sge
// CHECK:             arith.subi
// CHECK:             arith.select
// CHECK:             sparse_tensor.yield %{{.*}} : i64
// CHECK:           }
// CHECK:           absent = {
// CHECK:           }
// CHECK:           linalg.yield %[[U]] : i64
// CHECK:         } -> tensor<100xi64>
// CHECK:         return %[[T]] : tensor<100xi64>
// CHECK:       }
func.func @sparse_int_abs(%arg0: tensor<100xi64, #SV>) -> tensor<100xi64> {
  %0 = mhlo.abs(%arg0) : (tensor<100xi64, #SV>) -> tensor<100xi64>
  func.return %0 : tensor<100xi64>
}

// CHECK-LABEL: func @sparse_convert_complex(
// CHECK-SAME:    %[[A:.*]]: tensor<16xcomplex<f64>, #{{.*}}>) -> tensor<16xcomplex<f32>, #{{.*}}> {
// CHECK:         %[[T:.*]] = linalg.generic {{{.*}}} ins(%[[A]] : tensor<16xcomplex<f64>, #{{.*}}>) outs(%{{.*}} : tensor<16xcomplex<f32>, #{{.*}}>)
// CHECK:           %[[U:.*]] = sparse_tensor.unary %{{.*}} : complex<f64> to complex<f32>
// CHECK:           present = {
// CHECK:             complex.re
// CHECK:             arith.truncf
// CHECK:             complex.im
// CHECK:             arith.truncf
// CHECK:             complex.create
// CHECK:             sparse_tensor.yield %{{.*}} : complex<f32>
// CHECK:           }
// CHECK:           absent = {
// CHECK:           }
// CHECK:           linalg.yield %[[U]] : complex<f32>
// CHECK:         } -> tensor<16xcomplex<f32>, #{{.*}}>
// CHECK:         return %[[T]] : tensor<16xcomplex<f32>, #{{.*}}>
// CHECK:       }
func.func @sparse_convert_complex(%arg0: tensor<16xcomplex<f64>, #SV>) -> tensor<16xcomplex<f32>, #SV> {
  %0 = mhlo.convert(%arg0) : (tensor<16xcomplex<f64>, #SV>) -> tensor<16xcomplex<f32>, #SV>
  return %0 : tensor<16xcomplex<f32>, #SV>
}

// CHECK-LABEL: func @sparse_reduce(
// CHECK-SAME:    %[[ARG0:.*]]: tensor<10xi64, #{{.*}}>) -> tensor<i64> {
// CHECK:         %[[T0:.*]] = linalg.generic {{{.*}}} ins(%[[ARG0]] : tensor<10xi64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], {{.*}} }>>)
// CHECK:           arith.addi
// CHECK:         }
// CHECK:         return %[[T0]] : tensor<i64>
// CHECK:       }
func.func @sparse_reduce(%arg0: tensor<10xi64, #SV>) -> tensor<i64> {
  %0 = mhlo.constant dense<0> : tensor<i64>
  %1 = mhlo.reduce(%arg0 init: %0) across dimensions = [0] : (tensor<10xi64, #SV>, tensor<i64>) -> tensor<i64>
   reducer(%arg1: tensor<i64>, %arg2: tensor<i64>)  {
    %2 = mhlo.add %arg1, %arg2 : tensor<i64>
    "mhlo.return"(%2) : (tensor<i64>) -> ()
  }
  func.return %1 : tensor<i64>
}

// CHECK-LABEL: func @sparse_dot(
// CHECK-SAME:    %[[ARG0:.*]]: tensor<?xf32, #{{.*}}>,
// CHECK-SAME:    %[[ARG1:.*]]: tensor<?xf32, #{{.*}}>) -> tensor<f32> {
// CHECK:         %[[T0:.*]] = linalg.generic {{{.*}}} ins(%[[ARG0]], %[[ARG1]] : tensor<?xf32, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], {{.*}} }>>, tensor<?xf32, #sparse_tensor.encoding<{ dimLevelType = [ "compressed" ], {{.*}} }>>)
// CHECK:           arith.mulf
// CHECK:           arith.addf
// CHECK:         }
// CHECK:         return %[[T0]] : tensor<f32>
// CHECK:       }
func.func @sparse_dot(%arg0: tensor<?xf32, #SV>,
                      %arg1: tensor<?xf32, #SV>) -> tensor<f32> {
  %0 = "mhlo.dot_general"(%arg0, %arg1)
       {dot_dimension_numbers = #mhlo.dot<lhs_contracting_dimensions = [0],
                                          rhs_contracting_dimensions = [0]>,
                                          precision_config = [#mhlo<"precision DEFAULT">,
                                          #mhlo<"precision DEFAULT">]}
                  : (tensor<?xf32, #SV>, tensor<?xf32, #SV>) -> tensor<f32>
  func.return %0 : tensor<f32>
}

// CHECK-LABEL: func @sparse_transpose(
// CHECK-SAME:    %[[ARG0:.*]]: tensor<100x200xf64, #{{.*}}>) -> tensor<200x100xf64, #{{.*}}> {
// CHECK-DAG:     %[[C100:.*]] = arith.constant 100 : index
// CHECK-DAG:     %[[C200:.*]] = arith.constant 200 : index
// CHECK:         %[[T0:.*]] = sparse_tensor.init[%[[C200]], %[[C100]]] : tensor<200x100xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed", "compressed" ], {{.*}} }>>
// CHECK:         %[[T1:.*]] = linalg.generic {{.*}} ins(%[[ARG0]] : tensor<100x200xf64, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], {{.*}} }>>) outs(%[[T0]] : tensor<200x100xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed", "compressed" ], {{.*}} }>>) {
// CHECK:           linalg.yield
// CHECK:         }
// CHECK:         return %[[T1]] : tensor<200x100xf64, #sparse_tensor.encoding<{ dimLevelType = [ "compressed", "compressed" ], {{.*}} }>>
// CHECK:       }
func.func @sparse_transpose(%arg0: tensor<100x200xf64, #CSR>)
                                -> tensor<200x100xf64, #DCSR> {
  %0 = "mhlo.transpose"(%arg0) {permutation = dense<[1, 0]> : tensor<2xi64>}
     : (tensor<100x200xf64, #CSR>) -> tensor<200x100xf64, #DCSR>
  func.return %0 : tensor<200x100xf64, #DCSR>
}

// CHECK-LABEL: func @sparse_conv_eltwise(
// CHECK-SAME:    %[[ARG0:.*]]: tensor<2x3xf32, #{{.*}}>) -> tensor<2x3xi32, #{{.*}}> {
// CHECK-DAG:     %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG:     %[[C3:.*]] = arith.constant 3 : index
// CHECK:         %[[OUT:.*]] = sparse_tensor.init{{\[}}%[[C2]], %[[C3]]] : tensor<2x3xi32, #{{.*}}>
// CHECK:         %[[VAL:.*]] = linalg.generic {{{.*}} ins(%[[ARG0]] : tensor<2x3xf32, #sparse_tensor.encoding<{ dimLevelType = [ "dense", "compressed" ], {{.*}} }>>) outs(%[[OUT]] : tensor<2x3xi32, #sparse_tensor.encoding<{ dimLevelType = [ "compressed", "compressed" ], {{.*}} }>>)
//
// CHECK:           arith.fptosi
// CHECK:         }
// CHECK:         return %[[VAL]] : tensor<2x3xi32, #{{.*}}>
// CHECK:       }
func.func @sparse_conv_eltwise(%arg0: tensor<2x3xf32, #CSR>) -> tensor<2x3xi32, #DCSR> {
  %0 = mhlo.convert(%arg0) : (tensor<2x3xf32, #CSR>) -> tensor<2x3xi32, #DCSR>
  return %0 : tensor<2x3xi32, #DCSR>
}

