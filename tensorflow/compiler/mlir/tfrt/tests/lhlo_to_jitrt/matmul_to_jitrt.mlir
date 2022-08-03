// RUN: lhlo-tfrt-opt %s -lmhlo-gpu-to-jitrt -split-input-file | FileCheck %s

// CHECK: @compute(
// CHECK:   %[[A:[a-z0-9]+]]: memref<2x6x2x2xf32>,
// CHECK:   %[[B:[a-z0-9]+]]: memref<2x6x2x2xf32>,
// CHECK:   %[[C:[a-z0-9]+]]: memref<2x6x2x2xf32>,
// CHECK:   %[[D:[a-z0-9]+]]: memref<2x6x2x2xf32>
// CHECK: )
func.func @compute(%a: memref<2x6x2x2xf32>,
                   %b: memref<2x6x2x2xf32>,
                   %c: memref<2x6x2x2xf32>,
                   %d: memref<2x6x2x2xf32>) {

  // CHECK: @xla.gpu.cublas.lt.matmul(%[[A]], %[[B]], %[[C]], %[[D]])
  // CHECK-SAME:   alpha_imag = 0.000000e+00 : f64
  // CHECK-SAME:   alpha_real = 1.000000e+00 : f64
  // CHECK-SAME:   beta = 0.000000e+00 : f64
  // CHECK-SAME:   dot_dims = #mhlo.dot<lhs_batching_dimensions = [0, 1],
  // CHECK-SAME:                        rhs_batching_dimensions = [0, 1],
  // CHECK-SAME:                        lhs_contracting_dimensions = [3],
  // CHECK-SAME:                        rhs_contracting_dimensions = [2]>
  // CHECK-SAME:   epilogue = #lmhlo_gpu<epilogue Default>
  // CHECK-SAME:   precision = dense<0> : tensor<2xi32>
  // CHECK-SAME:   uid = 0 : i64
  "lmhlo_gpu.cublas.lt.matmul"(%a, %b, %c, %d) {
     algorithm = 0 : i64,
     alpha_imag = 0.000000e+00 : f64,
     alpha_real = 1.000000e+00 : f64,
     beta = 0.000000e+00 : f64,
     dot_dimension_numbers = #mhlo.dot<
       lhs_batching_dimensions = [0, 1],
       rhs_batching_dimensions = [0, 1],
       lhs_contracting_dimensions = [3],
       rhs_contracting_dimensions = [2]>,
     epilogue = #lmhlo_gpu<epilogue Default>,
     precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]
   } : (memref<2x6x2x2xf32>, memref<2x6x2x2xf32>,
        memref<2x6x2x2xf32>, memref<2x6x2x2xf32>) -> ()

  return
}

// CHECK: func private @xla.gpu.cublas.lt.matmul(
// CHECK-SAME:   memref<2x6x2x2xf32>, memref<2x6x2x2xf32>,
// CHECK-SAME:   memref<2x6x2x2xf32>, memref<2x6x2x2xf32>
// CHECK-SAME: ) attributes {rt.direct_custom_call = "xla.gpu.cublas.lt.matmul"}

// -----

// CHECK: @compute(
// CHECK:   %[[A:[a-z0-9]+]]: memref<2x6x2x2xf32>,
// CHECK:   %[[B:[a-z0-9]+]]: memref<2x6x2x2xf32>,
// CHECK:   %[[C:[a-z0-9]+]]: memref<2x6x2x2xf32>,
// CHECK:   %[[D:[a-z0-9]+]]: memref<2x6x2x2xf32>,
// CHECK:   %[[BIAS:[a-z0-9]+]]: memref<2x6x2x2xf32>
// CHECK: )
func.func @compute(%a: memref<2x6x2x2xf32>,
                   %b: memref<2x6x2x2xf32>,
                   %c: memref<2x6x2x2xf32>,
                   %d: memref<2x6x2x2xf32>,
                   %bias: memref<2x6x2x2xf32>) {

  // CHECK: @xla.gpu.cublas.lt.matmul(%[[A]], %[[B]], %[[C]], %[[D]], %[[BIAS]])
  // CHECK-SAME:   alpha_imag = 0.000000e+00 : f64
  // CHECK-SAME:   alpha_real = 1.000000e+00 : f64
  // CHECK-SAME:   beta = 0.000000e+00 : f64
  // CHECK-SAME:   dot_dims = #mhlo.dot<lhs_batching_dimensions = [0, 1],
  // CHECK-SAME:                        rhs_batching_dimensions = [0, 1],
  // CHECK-SAME:                        lhs_contracting_dimensions = [3],
  // CHECK-SAME:                        rhs_contracting_dimensions = [2]>
  // CHECK-SAME:   epilogue = #lmhlo_gpu<epilogue Default>
  // CHECK-SAME:   precision = dense<0> : tensor<2xi32>
  // CHECK-SAME:   uid = 0 : i64
  "lmhlo_gpu.cublas.lt.matmul"(%a, %b, %c, %d, %bias) {
     algorithm = 0 : i64,
     alpha_imag = 0.000000e+00 : f64,
     alpha_real = 1.000000e+00 : f64,
     beta = 0.000000e+00 : f64,
     dot_dimension_numbers = #mhlo.dot<
       lhs_batching_dimensions = [0, 1],
       rhs_batching_dimensions = [0, 1],
       lhs_contracting_dimensions = [3],
       rhs_contracting_dimensions = [2]>,
     epilogue = #lmhlo_gpu<epilogue Default>,
     precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>]
   } : (memref<2x6x2x2xf32>, memref<2x6x2x2xf32>, memref<2x6x2x2xf32>,
        memref<2x6x2x2xf32>, memref<2x6x2x2xf32>) -> ()

  return
}

// CHECK: func private @xla.gpu.cublas.lt.matmul(
// CHECK-SAME:   memref<2x6x2x2xf32>, memref<2x6x2x2xf32>, memref<2x6x2x2xf32>,
// CHECK-SAME:   memref<2x6x2x2xf32>, memref<2x6x2x2xf32>
// CHECK-SAME: ) attributes {rt.direct_custom_call = "xla.gpu.cublas.lt.matmul"}
