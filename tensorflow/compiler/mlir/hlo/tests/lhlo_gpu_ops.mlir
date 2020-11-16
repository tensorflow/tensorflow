// RUN: mlir-hlo-opt %s -verify-diagnostics -split-input-file | mlir-hlo-opt | FileCheck %s

// CHECK-LABEL: func @batch_norm_grad_memrefs
func @batch_norm_grad_memrefs(%arg0: memref<8x8x8x8xf32>, %arg1: memref<8xf32>, %arg2: memref<8xf32>,
                              %arg3: memref<8xf32>, %arg4: memref<8x8x8x8xf32>,
                              %grad_operand: memref<8x8x8x8xf32>, %grad_scale: memref<8xf32>,
                              %grad_offset: memref<8xf32>) -> () {
  "lmhlo_gpu.batch_norm_grad"(%arg0, %arg1, %arg2, %arg3, %arg4, %grad_operand, %grad_scale, %grad_offset) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64}
      : (memref<8x8x8x8xf32>, memref<8xf32>, memref<8xf32>, memref<8xf32>, memref<8x8x8x8xf32>,
         memref<8x8x8x8xf32>, memref<8xf32>, memref<8xf32>) -> ()
  return
}

// CHECK-LABEL: func @batch_norm_inference_memrefs
func @batch_norm_inference_memrefs(%arg0: memref<8x8x8x8xf32>, %arg1: memref<8xf32>, %arg2: memref<8xf32>,
                                   %arg3: memref<8xf32>, %arg4: memref<8xf32>, %arg_out: memref<8x8x8x8xf32>) -> () {
  "lmhlo_gpu.batch_norm_inference"(%arg0, %arg1, %arg2, %arg3, %arg4, %arg_out) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64}
      : (memref<8x8x8x8xf32>, memref<8xf32>, memref<8xf32>, memref<8xf32>, memref<8xf32>, memref<8x8x8x8xf32>) -> ()
  return
}

// CHECK-LABEL: func @batch_norm_training_memrefs
func @batch_norm_training_memrefs(%arg0: memref<8x8x8x8xf32>, %arg1: memref<8xf32>, %arg2: memref<8xf32>,
                                  %output: memref<8x8x8x8xf32>, %batch_mean: memref<8xf32>,
                                  %batch_var: memref<8xf32>) -> () {
  "lmhlo_gpu.batch_norm_training"(%arg0, %arg1, %arg2, %output, %batch_mean, %batch_var) {epsilon = 1.000000e-03 : f32, feature_index = 3 : i64}
      : (memref<8x8x8x8xf32>, memref<8xf32>, memref<8xf32>, memref<8x8x8x8xf32>, memref<8xf32>, memref<8xf32>) -> ()
  return
}

// CHECK-LABEL: func @conv_forward
func @conv_forward(%input : memref<1x1x8x8xf16>, %filter: memref<1x1x2x2xf16>, %output: memref<1x1x7x7xf16>) {
  %scratch = alloc() : memref<32xi8>
  // This defined a 2D convolution over a 8x8 single channel input using a 2x2
  // filter and with an output of 7x7xf16. The 1x1x8x8 is (N, C, H, W)
  "lmhlo_gpu.conv_forward"(%input, %filter, %output, %scratch)
    { dimension_numbers = {input_batch_dimension = 0 : i64,
                           input_feature_dimension = 1 : i64,
                           input_spatial_dimensions = dense<[2,3]> : tensor<2xi64>,
                           kernel_input_feature_dimension = 0 : i64,
                           kernel_output_feature_dimension = 1 : i64,
                           kernel_spatial_dimensions = dense<[2,3]> : tensor<2xi64>,
                           output_batch_dimension = 0 : i64,
                           output_feature_dimension = 1 : i64,
                           output_spatial_dimensions = dense<[2,3]> : tensor<2xi64>},
      window_strides = dense<[1, 1]> : tensor<2xi64>,
      padding = dense<[0,0]> : tensor<2xi64>,
      lhs_dilation = dense<[1,1]> : tensor<2xi64>,
      rhs_dilation = dense<[1,1]> : tensor<2xi64>,
      feature_group_count = 1,
      batch_group_count = 1,
      result_scale = 1.0,
      backend_config = {algorithm=0, tensor_ops_enabled = true }
    }
    : (memref<1x1x8x8xf16>, memref<1x1x2x2xf16>, memref<1x1x7x7xf16>, memref<32xi8>) -> ()
  return
}

// -----

// CHECK-LABEL: func @gemm
func @gemm(%lhs: memref<5x4xf32>, %rhs: memref<4x5xf32>, %output:memref<5x5xf32>) {
  "lmhlo_gpu.gemm"(%lhs, %rhs, %output) { dot_dimension_numbers = {
       lhs_batching_dimensions = dense<[1,1]> : tensor<2xi64>,
       rhs_batching_dimensions = dense<[1,1]> : tensor<2xi64>,
       lhs_contracting_dimensions = dense<[1,1]> : tensor<2xi64>,
       rhs_contracting_dimensions = dense<[1,1]> : tensor<2xi64>},
       alpha = 0.5,
       batch_size = 1,
       algorithm = 0}
    : (memref<5x4xf32>, memref<4x5xf32>, memref<5x5xf32>) -> ()
  return
}


// CHECK-LABEL: func @gemm_bias
func @gemm_bias(%lhs: memref<5x4xf32>, %rhs: memref<4x5xf32>,
                %bias: memref<5x5xf32>, %output:memref<5x5xf32>) {
  "lmhlo_gpu.gemm_bias"(%lhs, %rhs, %bias, %output) { dot_dimension_numbers = {
       lhs_batching_dimensions = dense<[1,1]> : tensor<2xi64>,
       rhs_batching_dimensions = dense<[1,1]> : tensor<2xi64>,
       lhs_contracting_dimensions = dense<[1,1]> : tensor<2xi64>,
       rhs_contracting_dimensions = dense<[1,1]> : tensor<2xi64>},
       alpha = 0.5,
       beta = 1.0,
       batch_size = 1,
       algorithm = 0}
    : (memref<5x4xf32>, memref<4x5xf32>, memref<5x5xf32>, memref<5x5xf32>) -> ()
  return
}

// CHECK-LABEL: func @cholesky
func @cholesky(%arg : memref<10x10xf32>, %out: memref<10x10xf32>) {
  %scratch = alloc() : memref<32xi8>
  %info = alloc() : memref<32xi32>
  "lmhlo_gpu.cholesky"(%arg, %out, %scratch, %info) { is_upper = true }
      : (memref<10x10xf32>, memref<10x10xf32>, memref<32xi8>, memref<32xi32>) -> ()
  return
}
