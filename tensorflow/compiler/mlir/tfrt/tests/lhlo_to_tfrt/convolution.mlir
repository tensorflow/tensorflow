// RUN: lhlo-tfrt-opt %s     \
// RUN:   -lmhlo-to-tfrt-gpu \
// RUN: | FileCheck %s --dump-input always

// CHECK:      func @conv_forward(
// CHECK-SAME:   %arg0: !tfrt.chain,
// CHECK-SAME:   %arg1: !tfrt_gpu.stream,
// CHECK-SAME:   %arg2: !tfrt_gpu.buffer,
// CHECK-SAME:   %arg3: !tfrt_gpu.buffer,
// CHECK-SAME:   %arg4: !tfrt_gpu.buffer,
// CHECK-SAME:   %arg5: !tfrt_gpu.buffer
// CHECK-SAME: ) -> !tfrt.chain
func @conv_forward(%input : memref<4x256x3x3xf16>, %filter: memref<256x256x2x2xf16>, %output: memref<4x256x2x2xf16>, %scratch: memref<0xui8>) {
  // CHECK-NOT: cast
  // CHECK-NOT: async.execute

  // CHECK: [[FILTER_DESC:%[0-9]+]] = tfrt_gpu.dnn.create_filter_descriptor
  // CHECK-SAME: CUDNN_DATA_HALF, 0, [256 : i32, 256 : i32, 2 : i32, 2 : i32],
  // CHECK-SAME: %arg0
  // CHECK: [[INPUT_DESC:%[0-9]+]] = tfrt_gpu.dnn.create_tensor_descriptor
  // CHECK-SAME: CUDNN_DATA_HALF, [4 : i32, 256 : i32, 3 : i32, 3 : i32],
  // CHECK-SAME: [2304 : i32, 9 : i32, 3 : i32, 1 : i32], %arg0
  // CHECK: [[OUTPUT_DESC:%[0-9]+]] = tfrt_gpu.dnn.create_tensor_descriptor
  // CHECK-SAME: CUDNN_DATA_HALF, [4 : i32, 256 : i32, 2 : i32, 2 : i32],
  // CHECK-SAME: [1024 : i32, 4 : i32, 2 : i32, 1 : i32], %arg0
  // CHECK: [[CONV_DESC:%[0-9]+]] = tfrt_gpu.dnn.create_convolution_descriptor
  // CHECK-SAME: CUDNN_DATA_FLOAT, CUDNN_CROSS_CORRELATION, [0 : i32, 0 : i32],
  // CHECK-SAME: [1 : i32, 1 : i32], [1 : i32, 1 : i32], %arg0
  // CHECK: [[HANDLE:%[0-9]+]] = tfrt_gpu.dnn.create %arg1
  // CHECK: [[ALGO:%[0-9]+]] = tfrt.constant.ui64 0
  // CHECK: [[CHAIN:%[0-9]+]] = tfrt_gpu.dnn.convolution_forward [[HANDLE]],
  // CHECK-SAME: CUDNN_DATA_FLOAT, [[INPUT_DESC]], %arg2, [[FILTER_DESC]],
  // CHECK-SAME: %arg3, [[CONV_DESC]], [[ALGO]], %arg5, [[OUTPUT_DESC]], %arg4

  lmhlo_gpu.conv_forward(%input, %filter, %output, %scratch)
    dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
    window = {stride = [1, 1], lhs_dilate = [1, 1], rhs_dilate = [1, 1],
              reverse = [0, 0]}
    { batch_group_count = 1 : i64, feature_group_count = 1 : i64,
      result_scale = 1.000000e+00 : f64,
      backend_config = {algorithm = 0 : i64,
                        workspace_size = -1,
                        operand_0_layout = [3, 2, 1, 0],
                        operand_1_layout = [3, 2, 1, 0],
                        result_layout = [3, 2, 1, 0],
                        tensor_ops_enabled = false}}
    : (memref<4x256x3x3xf16>, memref<256x256x2x2xf16>, memref<4x256x2x2xf16>, memref<0xui8>) -> ()

  // CHECK-NOT: cast
  // CHECK: tfrt.return [[CHAIN]] : !tfrt.chain
  "lmhlo.terminator"() : () -> ()
}

// CHECK:      func @conv_backwardinput(
// CHECK-SAME:   %arg0: !tfrt.chain,
// CHECK-SAME:   %arg1: !tfrt_gpu.stream,
// CHECK-SAME:   %arg2: !tfrt_gpu.buffer,
// CHECK-SAME:   %arg3: !tfrt_gpu.buffer,
// CHECK-SAME:   %arg4: !tfrt_gpu.buffer,
// CHECK-SAME:   %arg5: !tfrt_gpu.buffer
// CHECK-SAME: ) -> !tfrt.chain
func @conv_backwardinput(%d_output : memref<4x256x3x3xf16>, %filter: memref<256x256x2x2xf16>, %d_input: memref<4x256x2x2xf16>, %scratch: memref<0xui8>) {
  // CHECK-NOT: cast
  // CHECK-NOT: async.execute

  // CHECK: [[FILTER_DESC:%[0-9]+]] = tfrt_gpu.dnn.create_filter_descriptor
  // CHECK-SAME: CUDNN_DATA_HALF, 0, [256 : i32, 256 : i32, 2 : i32, 2 : i32],
  // CHECK-SAME: %arg0
  // CHECK: [[INPUT_DESC:%[0-9]+]] = tfrt_gpu.dnn.create_tensor_descriptor
  // CHECK-SAME: CUDNN_DATA_HALF, [4 : i32, 256 : i32, 2 : i32, 2 : i32],
  // CHECK-SAME: [1024 : i32, 4 : i32, 2 : i32, 1 : i32], %arg0
  // CHECK: [[OUTPUT_DESC:%[0-9]+]] = tfrt_gpu.dnn.create_tensor_descriptor
  // CHECK-SAME: CUDNN_DATA_HALF, [4 : i32, 256 : i32, 3 : i32, 3 : i32],
  // CHECK-SAME: [2304 : i32, 9 : i32, 3 : i32, 1 : i32], %arg0
  // CHECK: [[CONV_DESC:%[0-9]+]] = tfrt_gpu.dnn.create_convolution_descriptor
  // CHECK-SAME: CUDNN_DATA_FLOAT, CUDNN_CROSS_CORRELATION, [0 : i32, 0 : i32],
  // CHECK-SAME: [1 : i32, 1 : i32], [1 : i32, 1 : i32], %arg0
  // CHECK: [[HANDLE:%[0-9]+]] = tfrt_gpu.dnn.create %arg1
  // CHECK: [[ALGO:%[0-9]+]] = tfrt.constant.ui64 0
  // CHECK: [[CHAIN:%[0-9]+]] = tfrt_gpu.dnn.convolution_backward_data
  // CHECK-SAME: [[HANDLE]], CUDNN_DATA_FLOAT, [[FILTER_DESC]], %arg3,
  // CHECK-SAME: [[OUTPUT_DESC]], %arg2, [[CONV_DESC]], [[ALGO]], %arg5,
  // CHECK-SAME: [[INPUT_DESC]], %arg4

  lmhlo_gpu.conv_backwardinput(%d_output, %filter, %d_input, %scratch)
    dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
    window = {stride = [1, 1], lhs_dilate = [1, 1], rhs_dilate = [1, 1],
              reverse = [0, 0]}
    { batch_group_count = 1 : i64, feature_group_count = 1 : i64,
      result_scale = 1.000000e+00 : f64,
      backend_config = {algorithm = 0 : i64,
                        workspace_size = -1,
                        operand_0_layout = [3, 2, 1, 0],
                        operand_1_layout = [3, 2, 1, 0],
                        result_layout = [3, 2, 1, 0],
                        tensor_ops_enabled = false}}
    : (memref<4x256x3x3xf16>, memref<256x256x2x2xf16>, memref<4x256x2x2xf16>, memref<0xui8>) -> ()

  // CHECK-NOT: cast
  // CHECK: tfrt.return [[CHAIN]] : !tfrt.chain
  "lmhlo.terminator"() : () -> ()
}

// CHECK:      func @conv_backwardfilter(
// CHECK-SAME:   %arg0: !tfrt.chain,
// CHECK-SAME:   %arg1: !tfrt_gpu.stream,
// CHECK-SAME:   %arg2: !tfrt_gpu.buffer,
// CHECK-SAME:   %arg3: !tfrt_gpu.buffer,
// CHECK-SAME:   %arg4: !tfrt_gpu.buffer,
// CHECK-SAME:   %arg5: !tfrt_gpu.buffer
// CHECK-SAME: ) -> !tfrt.chain
func @conv_backwardfilter(%input : memref<4x256x3x3xf16>, %d_output: memref<256x256x2x2xf16>, %d_filter: memref<4x256x2x2xf16>, %scratch: memref<0xui8>) {
  // CHECK-NOT: cast
  // CHECK-NOT: async.execute

  // CHECK: [[FILTER_DESC:%[0-9]+]] = tfrt_gpu.dnn.create_filter_descriptor
  // CHECK-SAME: CUDNN_DATA_HALF, 0, [4 : i32, 256 : i32, 2 : i32, 2 : i32],
  // CHECK-SAME: %arg0
  // CHECK: [[INPUT_DESC:%[0-9]+]] = tfrt_gpu.dnn.create_tensor_descriptor
  // CHECK-SAME: CUDNN_DATA_HALF, [4 : i32, 256 : i32, 3 : i32, 3 : i32],
  // CHECK-SAME: [2304 : i32, 9 : i32, 3 : i32, 1 : i32], %arg0
  // CHECK: [[OUTPUT_DESC:%[0-9]+]] = tfrt_gpu.dnn.create_tensor_descriptor
  // CHECK-SAME: CUDNN_DATA_HALF, [256 : i32, 256 : i32, 2 : i32, 2 : i32],
  // CHECK-SAME: [1024 : i32, 4 : i32, 2 : i32, 1 : i32], %arg0
  // CHECK: [[CONV_DESC:%[0-9]+]] = tfrt_gpu.dnn.create_convolution_descriptor
  // CHECK-SAME: CUDNN_DATA_FLOAT, CUDNN_CROSS_CORRELATION, [0 : i32, 0 : i32],
  // CHECK-SAME: [1 : i32, 1 : i32], [1 : i32, 1 : i32], %arg0
  // CHECK: [[HANDLE:%[0-9]+]] = tfrt_gpu.dnn.create %arg1
  // CHECK: [[ALGO:%[0-9]+]] = tfrt.constant.ui64 0
  // CHECK: [[CHAIN:%[0-9]+]] = tfrt_gpu.dnn.convolution_backward_filter
  // CHECK-SAME: [[HANDLE]], CUDNN_DATA_FLOAT, [[INPUT_DESC]], %arg2,
  // CHECK-SAME: [[OUTPUT_DESC]], %arg3, [[CONV_DESC]], [[ALGO]], %arg5,
  // CHECK-SAME: [[FILTER_DESC]], %arg4

  lmhlo_gpu.conv_backwardfilter(%input, %d_output, %d_filter, %scratch)
    dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
    window = {stride = [1, 1], lhs_dilate = [1, 1], rhs_dilate = [1, 1],
              reverse = [0, 0]}
    { batch_group_count = 1 : i64, feature_group_count = 1 : i64,
      result_scale = 1.000000e+00 : f64,
      backend_config = {algorithm = 0 : i64,
                        workspace_size = -1,
                        operand_0_layout = [3, 2, 1, 0],
                        operand_1_layout = [3, 2, 1, 0],
                        result_layout = [3, 2, 1, 0],
                        tensor_ops_enabled = false}}
    : (memref<4x256x3x3xf16>, memref<256x256x2x2xf16>, memref<4x256x2x2xf16>, memref<0xui8>) -> ()

  // CHECK-NOT: cast
  // CHECK: tfrt.return [[CHAIN]] : !tfrt.chain
  "lmhlo.terminator"() : () -> ()
}
