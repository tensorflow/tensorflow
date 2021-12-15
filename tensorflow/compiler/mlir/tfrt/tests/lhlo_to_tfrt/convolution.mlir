// RUN: lhlo-tfrt-opt %s     \
// RUN:   -lmhlo-to-tfrt-gpu \
// RUN: | FileCheck %s

// CHECK: func @get_conv_forward_plan(
// CHECK-SAME:   %arg0: !tfrt_gpu.dnn.handle
// CHECK-SAME: ) -> !tfrt_gpu.dnn.convolution_plan
// CHECK: [[CONV_PLAN:%[0-9]+]] = tfrt_gpu.dnn.build_convolution %arg0,
// CHECK-SAME: CUDNN_DATA_HALF, CUDNN_DATA_HALF, [4, 256, 3, 3],
// CHECK-SAME: [2304, 9, 3, 1], [4, 256, 2, 2], [1024, 4, 2, 1],
// CHECK-SAME: [256, 256, 2, 2], [1024, 4, 2, 1], CUDNN_CROSS_CORRELATION, 2,
// CHECK-SAME: [1, 1], [0, 0], [1, 1], 10, 0
// CHECK: tfrt.return [[CONV_PLAN]] : !tfrt_gpu.dnn.convolution_plan

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

  // CHECK: [[HANDLE:%[0-9]+]] = tfrt_gpu.dnn.create %arg1
  // CHECK: [[CONV_PLAN:%[0-9]+]] = tfrt.once @get_conv_forward_plan([[HANDLE]])
  // CHECK: [[CHAIN:%[0-9]+]] = tfrt_gpu.dnn.run_convolution [[HANDLE]],
  // CHECK-SAME: [[CONV_PLAN]], %arg2, %arg4, %arg3, %arg5

  lmhlo_gpu.conv_forward(%input, %filter, %output, %scratch)
    dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
    window = {stride = [1, 1], lhs_dilate = [1, 1], rhs_dilate = [1, 1],
              reverse = [0, 0]}
    { batch_group_count = 1 : i64, feature_group_count = 1 : i64,
      result_scale = 1.000000e+00 : f64,
      backend_config = {algorithm = 0 : i64,
                        knob_ids = [0, 1, 2, 3],
                        knob_values = [4, 5, 6, 7],
                        workspace_size = 0,
                        is_cudnn_frontend = true,
                        operand_0_layout = [3, 2, 1, 0],
                        operand_1_layout = [3, 2, 1, 0],
                        result_layout = [3, 2, 1, 0],
                        tensor_ops_enabled = false}}
    : (memref<4x256x3x3xf16>, memref<256x256x2x2xf16>, memref<4x256x2x2xf16>, memref<0xui8>) -> ()

  // CHECK-NOT: cast
  // CHECK: tfrt.return [[CHAIN]] : !tfrt.chain
  "lmhlo.terminator"() : () -> ()
}

// CHECK: func @get_conv_backwardinput_plan(
// CHECK-SAME:   %arg0: !tfrt_gpu.dnn.handle
// CHECK-SAME: ) -> !tfrt_gpu.dnn.convolution_plan
// CHECK: [[CONV_PLAN:%[0-9]+]] = tfrt_gpu.dnn.build_convolution %arg0,
// CHECK-SAME: CUDNN_DATA_HALF, CUDNN_DATA_HALF, [4, 256, 2, 2],
// CHECK-SAME: [1024, 4, 2, 1], [4, 256, 3, 3], [2304, 9, 3, 1],
// CHECK-SAME: [256, 256, 2, 2], [1024, 4, 2, 1], CUDNN_CROSS_CORRELATION, 2,
// CHECK-SAME: [1, 1], [0, 0], [1, 1], 12, 0
// CHECK: tfrt.return [[CONV_PLAN]] : !tfrt_gpu.dnn.convolution_plan

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

  // CHECK: [[HANDLE:%[0-9]+]] = tfrt_gpu.dnn.create %arg1
  // CHECK: [[CONV_PLAN:%[0-9]+]] = tfrt.once
  // CHECK-SAME: @get_conv_backwardinput_plan([[HANDLE]])
  // CHECK: [[CHAIN:%[0-9]+]] = tfrt_gpu.dnn.run_convolution [[HANDLE]],
  // CHECK-SAME: [[CONV_PLAN]], %arg4, %arg2, %arg3, %arg5

  lmhlo_gpu.conv_backwardinput(%d_output, %filter, %d_input, %scratch)
    dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
    window = {stride = [1, 1], lhs_dilate = [1, 1], rhs_dilate = [1, 1],
              reverse = [0, 0]}
    { batch_group_count = 1 : i64, feature_group_count = 1 : i64,
      result_scale = 1.000000e+00 : f64,
      backend_config = {algorithm = 0 : i64,
                        knob_ids = [0, 1, 2, 3],
                        knob_values = [4, 5, 6, 7],
                        workspace_size = 0,
                        is_cudnn_frontend = true,
                        operand_0_layout = [3, 2, 1, 0],
                        operand_1_layout = [3, 2, 1, 0],
                        result_layout = [3, 2, 1, 0],
                        tensor_ops_enabled = false}}
    : (memref<4x256x3x3xf16>, memref<256x256x2x2xf16>, memref<4x256x2x2xf16>, memref<0xui8>) -> ()

  // CHECK-NOT: cast
  // CHECK: tfrt.return [[CHAIN]] : !tfrt.chain
  "lmhlo.terminator"() : () -> ()
}

// CHECK: func @get_conv_backwardfilter_plan(
// CHECK-SAME:   %arg0: !tfrt_gpu.dnn.handle
// CHECK-SAME: ) -> !tfrt_gpu.dnn.convolution_plan
// CHECK: [[CONV_PLAN:%[0-9]+]] = tfrt_gpu.dnn.build_convolution %arg0,
// CHECK-SAME: CUDNN_DATA_HALF, CUDNN_DATA_HALF, [4, 256, 3, 3],
// CHECK-SAME: [2304, 9, 3, 1], [256, 256, 2, 2], [1024, 4, 2, 1],
// CHECK-SAME: [4, 256, 2, 2], [1024, 4, 2, 1], CUDNN_CROSS_CORRELATION, 2,
// CHECK-SAME: [1, 1], [0, 0], [1, 1], 11, 0
// CHECK: tfrt.return [[CONV_PLAN]] : !tfrt_gpu.dnn.convolution_plan

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

  // CHECK: [[HANDLE:%[0-9]+]] = tfrt_gpu.dnn.create %arg1
  // CHECK: [[CONV_PLAN:%[0-9]+]] = tfrt.once
  // CHECK-SAME: @get_conv_backwardfilter_plan([[HANDLE]])
  // CHECK: [[CHAIN:%[0-9]+]] = tfrt_gpu.dnn.run_convolution [[HANDLE]],
  // CHECK-SAME: [[CONV_PLAN]], %arg2, %arg3, %arg4, %arg5

  lmhlo_gpu.conv_backwardfilter(%input, %d_output, %d_filter, %scratch)
    dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
    window = {stride = [1, 1], lhs_dilate = [1, 1], rhs_dilate = [1, 1],
              reverse = [0, 0]}
    { batch_group_count = 1 : i64, feature_group_count = 1 : i64,
      result_scale = 1.000000e+00 : f64,
      backend_config = {algorithm = 0 : i64,
                        knob_ids = [0, 1, 2, 3],
                        knob_values = [4, 5, 6, 7],
                        workspace_size = 0,
                        is_cudnn_frontend = true,
                        operand_0_layout = [3, 2, 1, 0],
                        operand_1_layout = [3, 2, 1, 0],
                        result_layout = [3, 2, 1, 0],
                        tensor_ops_enabled = false}}
    : (memref<4x256x3x3xf16>, memref<256x256x2x2xf16>, memref<4x256x2x2xf16>, memref<0xui8>) -> ()

  // CHECK-NOT: cast
  // CHECK: tfrt.return [[CHAIN]] : !tfrt.chain
  "lmhlo.terminator"() : () -> ()
}

// CHECK: func @get_conv_forward_fused_plan(
// CHECK-SAME:   %arg0: !tfrt_gpu.dnn.handle
// CHECK-SAME: ) -> !tfrt_gpu.dnn.convolution_plan
// CHECK: [[ALPHA:%[0-9]+]] = tfrt.constant.f64 1.000000e+00
// CHECK: [[ALPHA2:%[0-9]+]] = tfrt.constant.f64 0.000000e+00
// CHECK: [[CONV_PLAN:%[0-9]+]] = tfrt_gpu.dnn.build_fused_convolution %arg0,
// CHECK-SAME: CUDNN_DATA_HALF, CUDNN_DATA_HALF, CUDNN_DATA_HALF, [1, 17, 9, 9],
// CHECK-SAME: [1377, 81, 9, 1], [1, 32, 9, 9], [2592, 81, 9, 1],
// CHECK-SAME: [3, 3, 17, 32], [1632, 544, 32, 1], [1, 32, 1, 1], [32, 1, 1, 1],
// CHECK-SAME: CUDNN_CROSS_CORRELATION, 2, [1, 1], [0, 0], [1, 1], 10,
// CHECK-SAME: [[ALPHA]], [[ALPHA2]], 1, 0
// CHECK: tfrt.return [[CONV_PLAN]] : !tfrt_gpu.dnn.convolution_plan

// CHECK:      func @conv_forward_fused(
// CHECK-SAME:   %arg0: !tfrt.chain,
// CHECK-SAME:   %arg1: !tfrt_gpu.stream,
// CHECK-SAME:   %arg2: !tfrt_gpu.buffer,
// CHECK-SAME:   %arg3: !tfrt_gpu.buffer,
// CHECK-SAME:   %arg4: !tfrt_gpu.buffer,
// CHECK-SAME:   %arg5: !tfrt_gpu.buffer,
// CHECK-SAME:   %arg6: !tfrt_gpu.buffer
// CHECK-SAME: ) -> !tfrt.chain
func @conv_forward_fused(%input : memref<1x17x9x9xf16>, %filter : memref<3x3x17x32xf16>, %bias : memref<32xf16>, %output : memref<1x32x9x9xf16>, %scratch: memref<32xui8>) {
  // CHECK-NOT: cast
  // CHECK-NOT: async.execute

  // CHECK: [[HANDLE:%[0-9]+]] = tfrt_gpu.dnn.create %arg1
  // CHECK: [[CONV_PLAN:%[0-9]+]] = tfrt.once
  // CHECK-SAME: @get_conv_forward_fused_plan([[HANDLE]])
  // CHECK: [[CHAIN:%[0-9]+]] = tfrt_gpu.dnn.run_fused_convolution [[HANDLE]],
  // CHECK-SAME: [[CONV_PLAN]], %arg2, %arg5, %arg3, %arg5, %arg4, %arg6

  lmhlo_gpu.conv_forward_fused(%input, %filter, %bias, %output, %scratch)
    dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
    window = {stride = [1, 1], lhs_dilate = [1, 1], rhs_dilate = [1, 1],
              reverse = [0, 0]}
    { batch_group_count = 1 : i64, feature_group_count = 1 : i64,
      result_scale = 1.000000e+00 : f64, activation_mode = "Relu",
      backend_config = {algorithm = 0 : i64,
                        knob_ids = [0, 1, 2, 3],
                        knob_values = [4, 5, 6, 7],
                        workspace_size = 0,
                        is_cudnn_frontend = true,
                        operand_0_layout = [3, 2, 1, 0],
                        operand_1_layout = [3, 2, 1, 0],
                        result_layout = [3, 2, 1, 0],
                        tensor_ops_enabled = false}}
    : (memref<1x17x9x9xf16>, memref<3x3x17x32xf16>, memref<32xf16>, memref<1x32x9x9xf16>, memref<32xui8>) -> ()

  // CHECK-NOT: cast
  // CHECK: tfrt.return [[CHAIN]] : !tfrt.chain
  "lmhlo.terminator"() : () -> ()
}

// CHECK: func @get_conv_forward_fused_with_side_input_plan(
// CHECK-SAME:   %arg0: !tfrt_gpu.dnn.handle
// CHECK-SAME: ) -> !tfrt_gpu.dnn.convolution_plan
// CHECK: [[ALPHA:%[0-9]+]] = tfrt.constant.f64 1.000000e+00
// CHECK: [[ALPHA2:%[0-9]+]] = tfrt.constant.f64 1.000000e+00
// CHECK: [[CONV_PLAN:%[0-9]+]] = tfrt_gpu.dnn.build_fused_convolution %arg0,
// CHECK-SAME: CUDNN_DATA_HALF, CUDNN_DATA_HALF, CUDNN_DATA_HALF, [1, 17, 9, 9],
// CHECK-SAME: [1377, 81, 9, 1], [1, 32, 9, 9], [2592, 81, 9, 1],
// CHECK-SAME: [3, 3, 17, 32], [1632, 544, 32, 1], [1, 32, 1, 1], [32, 1, 1, 1],
// CHECK-SAME: CUDNN_CROSS_CORRELATION, 2, [1, 1], [0, 0], [1, 1], 10,
// CHECK-SAME: [[ALPHA]], [[ALPHA2]], 1, 0
// CHECK: tfrt.return [[CONV_PLAN]] : !tfrt_gpu.dnn.convolution_plan

// CHECK:      func @conv_forward_fused_with_side_input(
// CHECK-SAME:   %arg0: !tfrt.chain,
// CHECK-SAME:   %arg1: !tfrt_gpu.stream,
// CHECK-SAME:   %arg2: !tfrt_gpu.buffer,
// CHECK-SAME:   %arg3: !tfrt_gpu.buffer,
// CHECK-SAME:   %arg4: !tfrt_gpu.buffer,
// CHECK-SAME:   %arg5: !tfrt_gpu.buffer,
// CHECK-SAME:   %arg6: !tfrt_gpu.buffer,
// CHECK-SAME:   %arg7: !tfrt_gpu.buffer
// CHECK-SAME: ) -> !tfrt.chain
func @conv_forward_fused_with_side_input(%input : memref<1x17x9x9xf16>, %filter : memref<3x3x17x32xf16>, %bias : memref<32xf16>, %side_input : memref<32xf16>, %output : memref<1x32x9x9xf16>, %scratch: memref<32xui8>) {
  // CHECK-NOT: cast
  // CHECK-NOT: async.execute

  // CHECK: [[HANDLE:%[0-9]+]] = tfrt_gpu.dnn.create %arg1
  // CHECK: [[CONV_PLAN:%[0-9]+]] = tfrt.once
  // CHECK-SAME: @get_conv_forward_fused_with_side_input_plan([[HANDLE]])
  // CHECK: [[CHAIN:%[0-9]+]] = tfrt_gpu.dnn.run_fused_convolution [[HANDLE]],
  // CHECK-SAME: [[CONV_PLAN]], %arg2, %arg6, %arg3, %arg5, %arg4, %arg7

  lmhlo_gpu.conv_forward_fused_with_side_input(%input, %filter, %bias, %side_input, %output, %scratch)
    dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
    window = {stride = [1, 1], lhs_dilate = [1, 1], rhs_dilate = [1, 1],
              reverse = [0, 0]}
    { batch_group_count = 1 : i64, feature_group_count = 1 : i64,
      result_scale = 1.000000e+00 : f64, side_input_scale = 1.000000e+00 : f64,
      activation_mode = "Relu",
      backend_config = {algorithm = 0 : i64,
                        knob_ids = [0, 1, 2, 3],
                        knob_values = [4, 5, 6, 7],
                        workspace_size = 0,
                        is_cudnn_frontend = true,
                        operand_0_layout = [3, 2, 1, 0],
                        operand_1_layout = [3, 2, 1, 0],
                        result_layout = [3, 2, 1, 0],
                        tensor_ops_enabled = false}}
    : (memref<1x17x9x9xf16>, memref<3x3x17x32xf16>, memref<32xf16>, memref<32xf16>, memref<1x32x9x9xf16>, memref<32xui8>) -> ()

  // CHECK-NOT: cast
  // CHECK: tfrt.return [[CHAIN]] : !tfrt.chain
  "lmhlo.terminator"() : () -> ()
}
