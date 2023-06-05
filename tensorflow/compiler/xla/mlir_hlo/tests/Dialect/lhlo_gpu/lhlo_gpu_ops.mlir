// RUN: mlir-hlo-opt %s -verify-diagnostics -split-input-file \
// RUN:   | mlir-hlo-opt \
// RUN:   | FileCheck %s

// CHECK-LABEL: func @conv_forward_generic
// CHECK: lmhlo_gpu.conv_forward
// CHECK-SAME: dim_numbers = [b, f, 0, 1]x[i, o, 0, 1]->[b, f, 0, 1]
// CHECK-SAME{LITERAL}: window = {stride = [1, 1], pad = [[0, 0], [1, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
func.func @conv_forward_generic(%input : memref<1x1x8x8xf16>, %filter: memref<1x1x2x2xf16>, %output: memref<1x1x7x7xf16>) {
  %scratch = memref.alloc() : memref<32xi8>
  // This defined a 2D convolution over a 8x8 single channel input using a 2x2
  // filter and with an output of 7x7xf16. The 1x1x8x8 is (N, C, H, W)
  "lmhlo_gpu.conv_forward"(%input, %filter, %output, %scratch) {
    dimension_numbers = #mhlo.conv<raw
        input_batch_dimension = 0,
        input_feature_dimension = 1,
        input_spatial_dimensions = [2, 3],
        kernel_input_feature_dimension = 0,
        kernel_output_feature_dimension = 1,
        kernel_spatial_dimensions = [2, 3],
        output_batch_dimension = 0,
        output_feature_dimension = 1,
        output_spatial_dimensions = [2, 3]
    >,
    window_strides = dense<[1, 1]> : tensor<2xi64>,
    padding = dense<[[0, 0], [1, 0]]> : tensor<2x2xi64>,
    lhs_dilation = dense<[1,1]> : tensor<2xi64>,
    rhs_dilation = dense<[1,1]> : tensor<2xi64>,
    feature_group_count = 1,
    batch_group_count = 1,
    result_scale = 1.0,
    backend_config = #lmhlo_gpu.convolution_backend_config<
        algorithm = 0,
        tensor_ops_enabled = true,
        is_cudnn_reordered_int8 = false,
        knob_ids = [],
        knob_values = [],
        is_cudnn_frontend = false,
        workspace_size = -1,
        operand_0_layout = [3,2,1,0],
        operand_1_layout = [3,2,1,0],
        result_layout = [3,2,1,0]
    >
  } : (memref<1x1x8x8xf16>, memref<1x1x2x2xf16>, memref<1x1x7x7xf16>, memref<32xi8>) -> ()
  func.return
}

// CHECK-LABEL: func @conv_forward
// CHECK: lmhlo_gpu.conv_forward
// CHECK-SAME: dim_numbers = [b, f, 0, 1]x[i, o, 0, 1]->[b, f, 0, 1]
// CHECK-SAME{LITERAL}: window = {stride = [1, 1], pad = [[0, 0], [1, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
func.func @conv_forward(%input : memref<1x1x8x8xf16>, %filter: memref<1x1x2x2xf16>, %output: memref<1x1x7x7xf16>) {
  %scratch = memref.alloc() : memref<32xi8>
  // This defined a 2D convolution over a 8x8 single channel input using a 2x2
  // filter and with an output of 7x7xf16. The 1x1x8x8 is (N, C, H, W)
  lmhlo_gpu.conv_forward(%input, %filter, %output, %scratch)
    dim_numbers = [b, f, 0, 1]x[i, o, 0, 1]->[b, f, 0, 1],
    window = {stride = [1, 1], pad = [[0, 0], [1, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
    { feature_group_count = 1, batch_group_count = 1, result_scale = 1.0,
      backend_config = #lmhlo_gpu.convolution_backend_config<
        algorithm = 0,
        tensor_ops_enabled = true,
        knob_ids = [],
        knob_values = [],
        is_cudnn_frontend = false,
        is_cudnn_reordered_int8 = false,
        workspace_size = -1,
        operand_0_layout = [3,2,1,0],
        operand_1_layout = [3,2,1,0],
        result_layout = [3,2,1,0]
      >
  } : (memref<1x1x8x8xf16>, memref<1x1x2x2xf16>, memref<1x1x7x7xf16>, memref<32xi8>) -> ()
  func.return
}

// CHECK-LABEL: func @conv_backfilter
// CHECK: lmhlo_gpu.conv_backwardfilter
// CHECK-SAME: dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f]
// CHECK-SAME{LITERAL}: window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
func.func @conv_backfilter(%input : memref<3x56x56x16xf64>, %filter: memref<3x3x3x64xf64>, %output: memref<54x54x16x64xf64>) {
  %scratch = memref.alloc() : memref<23328xui8>
  lmhlo_gpu.conv_backwardfilter(%input, %filter, %output, %scratch)
    dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
    window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
    { backend_config = #lmhlo_gpu.convolution_backend_config<
        algorithm = 0,
        tensor_ops_enabled = true,
        knob_ids = [],
        knob_values = [],
        is_cudnn_frontend = false,
        is_cudnn_reordered_int8 = false,
        workspace_size = -1,
        operand_0_layout = [3,2,1,0],
        operand_1_layout = [3,2,1,0],
        result_layout = [3,2,1,0]
      >,
      batch_group_count = 1 : i64,
      feature_group_count = 1 : i64,
      precision_config = [],
      result_scale = 1.000000e+00 : f64
  } : (memref<3x56x56x16xf64>, memref<3x3x3x64xf64>, memref<54x54x16x64xf64>, memref<23328xui8>) -> ()
  func.return
}

// CHECK-LABEL: func @conv_backinput
// CHECK: lmhlo_gpu.conv_backwardinput
// CHECK-SAME: dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1]
// CHECK-SAME{LITERAL}: window = {stride = [1, 1], pad = [[3, 0], [1, 5]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [1, 1]}
func.func @conv_backinput(%input : memref<4x5x16x16xf64>, %filter : memref<5x3x7x7xf64>, %output : memref<4x3x16x16xf64>) {
  %scratch = memref.alloc() : memref<32xui8>
  lmhlo_gpu.conv_backwardinput(%input, %filter, %output, %scratch)
    dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
    window = {stride = [1, 1], pad = [[3, 0], [1, 5]], lhs_dilate = [1, 1], rhs_dilate = [1, 1], reverse = [1, 1]}
    { backend_config = #lmhlo_gpu.convolution_backend_config<
        algorithm = 0,
        tensor_ops_enabled = true,
        knob_ids = [],
        knob_values = [],
        is_cudnn_frontend = false,
        is_cudnn_reordered_int8 = false,
        workspace_size = -1,
        operand_0_layout = [3,2,1,0],
        operand_1_layout = [3,2,1,0],
        result_layout = [3,2,1,0]
      >,
      batch_group_count = 1 : i64,
      feature_group_count = 1 : i64,
      precision_config = [],
      result_scale = 1.000000e+00 : f64
  } : (memref<4x5x16x16xf64>, memref<5x3x7x7xf64>, memref<4x3x16x16xf64>, memref<32xui8>) -> ()
  func.return
}

// CHECK-LABEL: func @conv_fused
// CHECK: lmhlo_gpu.conv_forward_fused
// CHECK-SAME: dim_numbers = [b, f, 0, 1]x[0, 1, i, o]->[b, f, 0, 1]
// CHECK-SAME{LITERAL}: window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
func.func @conv_fused(%input : memref<1x17x9x9xf16>, %filter : memref<3x3x17x32xf16>, %bias : memref<32xf16>, %output : memref<1x32x9x9xf16>) {
  %scratch = memref.alloc() : memref<32xui8>
  lmhlo_gpu.conv_forward_fused(%input, %filter, %bias, %output, %scratch)
    dim_numbers = [b, f, 0, 1]x[0, 1, i, o]->[b, f, 0, 1],
    window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
    { activation_mode = #lmhlo_gpu<activation Relu>,
      backend_config = #lmhlo_gpu.convolution_backend_config<
        algorithm = 0,
        tensor_ops_enabled = true,
        knob_ids = [],
        knob_values = [],
        is_cudnn_frontend = false,
        is_cudnn_reordered_int8 = false,
        workspace_size = -1,
        operand_0_layout = [3,2,1,0],
        operand_1_layout = [3,2,1,0],
        result_layout = [3,2,1,0]
      >,
      batch_group_count = 1 : i64,
      feature_group_count = 1 : i64,
      precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>],
      result_scale = 1.000000e+00 : f64
  } : (memref<1x17x9x9xf16>, memref<3x3x17x32xf16>, memref<32xf16>, memref<1x32x9x9xf16>, memref<32xui8>) -> ()
  func.return
}

// CHECK-LABEL: func @conv_fused_side_input
// CHECK: lmhlo_gpu.conv_forward_fused_with_side_input
// CHECK-SAME: dim_numbers = [b, f, 0, 1]x[0, 1, i, o]->[b, f, 0, 1]
// CHECK-SAME{LITERAL}: window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
func.func @conv_fused_side_input(%input : memref<1x17x9x9xf16>, %filter : memref<3x3x17x32xf16>, %bias : memref<32xf16>, %side_input:  memref<32xf16>, %output : memref<1x32x9x9xf16>) {
  %scratch = memref.alloc() : memref<0xui8>
  lmhlo_gpu.conv_forward_fused_with_side_input(%input, %filter, %bias, %side_input, %output, %scratch)
    dim_numbers = [b, f, 0, 1]x[0, 1, i, o]->[b, f, 0, 1],
    window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
    { activation_mode = #lmhlo_gpu<activation Relu>,
      backend_config = #lmhlo_gpu.convolution_backend_config<
        algorithm = 0,
        tensor_ops_enabled = true,
        knob_ids = [],
        knob_values = [],
        is_cudnn_frontend = false,
        is_cudnn_reordered_int8 = false,
        workspace_size = -1,
        operand_0_layout = [3,2,1,0],
        operand_1_layout = [3,2,1,0],
        result_layout = [3,2,1,0]
      >,
      batch_group_count = 1 : i64,
      feature_group_count = 1 : i64,
      precision_config = [#mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>, #mhlo<precision DEFAULT>],
      result_scale = 1.000000e+00 : f64,
      side_input_scale = 1.000000e+00 : f64
  } : (memref<1x17x9x9xf16>, memref<3x3x17x32xf16>, memref<32xf16>, memref<32xf16>, memref<1x32x9x9xf16>, memref<0xui8>) -> ()
  func.return
}

// CHECK-LABEL: func @gemm
func.func @gemm(%lhs: memref<5x4xf32>, %rhs: memref<4x5xf32>, %output:memref<5x5xf32>) {
  "lmhlo_gpu.gemm"(%lhs, %rhs, %output) {
    dot_dimension_numbers = #mhlo.dot<
       lhs_batching_dimensions = [1,1],
       rhs_batching_dimensions = [1,1],
       lhs_contracting_dimensions = [1,1],
       rhs_contracting_dimensions = [1,1]
    >,
    alpha_real = 0.5,
    alpha_imag = 0.0,
    beta = 0.0,
    batch_size = 1,
    lhs_stride = 20,
    rhs_stride = 20,
    algorithm = 0
  } : (memref<5x4xf32>, memref<4x5xf32>, memref<5x5xf32>) -> ()
  func.return
}

// CHECK-LABEL: func @cholesky
func.func @cholesky(%arg : memref<10x10xf32>, %out: memref<10x10xf32>) {
  %scratch = memref.alloc() : memref<32xi8>
  %info = memref.alloc() : memref<32xi32>
  "lmhlo_gpu.cholesky"(%arg, %out, %scratch, %info) { is_lower = true }
      : (memref<10x10xf32>, memref<10x10xf32>, memref<32xi8>, memref<32xi32>) -> ()
  func.return
}

// CHECK-LABEL: func @ag_start
func.func @ag_start(%arg : memref<10x10xf32>, %out: memref<20x10xf32>) {
  %0 = "lmhlo_gpu.all_gather_start"(%arg, %out)
    {
      replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>,
      all_gather_dimension = 0,
      is_sync = false
    }
    : (memref<10x10xf32>, memref<20x10xf32>) -> (!mhlo.token)
  func.return
}

// CHECK-LABEL: func @ag_start_mixed
func.func @ag_start_mixed(%arg0 : memref<10x10xf32>, %arg1 : memref<10x10xf16>,
                    %out0: memref<20x10xf32>, %out1: memref<20x10xf16>) {
  %0 = "lmhlo_gpu.all_gather_start"(%arg0, %arg1, %out0, %out1)
    {
      replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>,
      all_gather_dimension = 0,
      is_sync = true
    }
    : (memref<10x10xf32>, memref<10x10xf16>, memref<20x10xf32>, memref<20x10xf16>) -> (!mhlo.token)
  func.return
}
