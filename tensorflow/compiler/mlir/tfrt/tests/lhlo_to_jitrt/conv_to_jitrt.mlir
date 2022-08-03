// Copyright 2022 The TensorFlow Runtime Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// RUN: lhlo-tfrt-opt %s -lmhlo-gpu-to-jitrt -split-input-file | FileCheck %s

#map0 = affine_map<(d0, d1, d2, d3) -> (d0 * 3 + d1 + d2 * 9 + d3 * 9)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0 * 16384 + d1 * 4 + d2 + d3 * 16)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0 * 4096 + d1 * 2 + d2 + d3 * 4)>

// CHECK: @conv_forward(
// CHECK:   %[[INPUT:[a-z0-9]+]]: memref
// CHECK:   %[[FILTER:[a-z0-9]+]]: memref
// CHECK:   %[[OUTPUT:[a-z0-9]+]]: memref
// CHECK:   %[[SCRATCH:[a-z0-9]+]]: memref
// CHECK: )
func.func @conv_forward(%input: memref<1x4x4x1024xf16, #map1>,
                        %filter: memref<3x3x1x1024xf16, #map0>,
                        %output: memref<1x2x2x1024xf16, #map2>,
                        %scratch: memref<0xui8>) {

  // CHECK: call @xla.gpu.conv.forward(
  // CHECK-SAME: %[[INPUT]], %[[FILTER]], %[[OUTPUT]], %[[SCRATCH]])

  // CHECK-DAG: conv_dims = #mhlo.conv<[b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f]>

  // CHECK-DAG: window_strides = dense<1> : tensor<2xi64>
  // CHECK-DAG: lhs_dilation = dense<1> : tensor<2xi64>
  // CHECK-DAG: rhs_dilation = dense<1> : tensor<2xi64>
  // CHECK-DAG: window_reversal = dense<0> : tensor<2xi64>
  // CHECK-DAG: padding = dense<> : tensor<0xi64>

  // CHECK-DAG: backend_config = #lmhlo_gpu.convolution_backend_config<
  // CHECK-DAG: algorithm = 0
  // CHECK-DAG: is_cudnn_frontend = true
  // CHECK-DAG: knob_ids = []
  // CHECK-DAG: knob_values = []
  // CHECK-DAG: operand_0_layout = [2, 1, 3, 0]
  // CHECK-DAG: operand_1_layout = [1, 0, 2, 3]
  // CHECK-DAG: tensor_ops_enabled = false
  // CHECK-DAG: workspace_size = 0

  // CHECK-DAG: feature_group_count = 1024 : i64
  // CHECK-DAG: result_scale = 1.000000e+00 : f64
  lmhlo_gpu.conv_forward(%input, %filter, %output, %scratch)
    dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
    window = { stride = [1, 1],
               lhs_dilate = [1, 1],
               rhs_dilate = [1, 1],
               reverse = [0, 0]
             }
    { backend_config = #lmhlo_gpu.convolution_backend_config<
        algorithm = 0,
        is_cudnn_frontend = true,
        knob_ids = [],
        knob_values = [],
        operand_0_layout = [2, 1, 3, 0],
        operand_1_layout = [1, 0, 2, 3],
        result_layout = [2, 1, 3, 0],
        tensor_ops_enabled = false,
        workspace_size = 0
      >,
      batch_group_count = 1 : i64,
      feature_group_count = 1024 : i64,
      precision_config = [],
      result_scale = 1.000000e+00 : f64
    } : (memref<1x4x4x1024xf16, #map1>,
         memref<3x3x1x1024xf16, #map0>,
         memref<1x2x2x1024xf16, #map2>,
         memref<0xui8>) -> ()

  return
}

// CHECK: func private @xla.gpu.conv.forward(
// CHECK-SAME:   memref<1x4x4x1024xf16, #map0>, memref<3x3x1x1024xf16, #map1>,
// CHECK-SAME:   memref<1x2x2x1024xf16, #map2>, memref<0xui8>)
// CHECK-SAME: attributes {rt.direct_custom_call = "xla.gpu.conv.forward"}

// -----

#map0 = affine_map<(d0, d1, d2, d3) -> (d0 * 9 + d1 * 3 + d2 + d3 * 9)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0 * 3 + d1 + d2 * 27 + d3 * 9)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0 + d1 + d2 + d3)>

// CHECK: @conv_backwardfilter(
// CHECK:   %[[INPUT:[a-z0-9]+]]: memref
// CHECK:   %[[D_OUTPUT:[a-z0-9]+]]: memref
// CHECK:   %[[D_FILTER:[a-z0-9]+]]: memref
// CHECK:   %[[SCRATCH:[a-z0-9]+]]: memref
// CHECK: )
func.func @conv_backwardfilter(%input: memref<1x3x3x5xf16, #map0>,
                               %d_output: memref<3x3x5x3xf16, #map1>,
                               %d_filter: memref<1x1x1x3xf16, #map2>,
                               %scratch: memref<0xui8>) {
  // CHECK: call @xla.gpu.conv.backward.filter(
  // CHECK-SAME: %[[INPUT]], %[[D_OUTPUT]], %[[D_FILTER]], %[[SCRATCH]])
  lmhlo_gpu.conv_backwardfilter(%input, %d_output, %d_filter, %scratch)
    dim_numbers = [f, 0, 1, b]x[i, 0, 1, o]->[0, 1, b, f],
    window = { stride = [1, 1],
               lhs_dilate = [1, 1],
               rhs_dilate = [1, 1],
               reverse = [0, 0]
             }
    { backend_config = #lmhlo_gpu.convolution_backend_config<
        algorithm = 0,
        is_cudnn_frontend = true,
        knob_ids = [],
        knob_values = [],
        operand_0_layout = [2, 1, 0, 3],
        operand_1_layout = [1, 0, 3, 2],
        result_layout = [2, 1, 0, 3],
        tensor_ops_enabled = false,
        workspace_size = 0
      >,
      batch_group_count = 1 : i64,
      feature_group_count = 1 : i64,
      precision_config = [],
      result_scale = 1.000000e+00 : f64
    } : (memref<1x3x3x5xf16, #map0>,
         memref<3x3x5x3xf16, #map1>,
         memref<1x1x1x3xf16, #map2>,
         memref<0xui8>) -> ()
  return
}

// CHECK: func private @xla.gpu.conv.backward.filter(
// CHECK-SAME:   memref<1x3x3x5xf16, #map0>, memref<3x3x5x3xf16, #map1>,
// CHECK-SAME:   memref<1x1x1x3xf16, #map2>, memref<0xui8>
// CHECK-SAME: ) attributes {rt.direct_custom_call =
// CHECK-SAME:              "xla.gpu.conv.backward.filter"}

// -----

// CHECK: @conv_backwardinput(
// CHECK:   %[[D_OUTPUT:[a-z0-9]+]]: memref
// CHECK:   %[[FILTER:[a-z0-9]+]]: memref
// CHECK:   %[[D_INPUT:[a-z0-9]+]]: memref
// CHECK:   %[[SCRATCH:[a-z0-9]+]]: memref
// CHECK: )
func.func @conv_backwardinput(%d_output: memref<4x5x16x16xf64>,
                              %filter: memref<5x3x7x7xf64>,
                              %d_input: memref<4x3x16x16xf64>,
                              %scratch: memref<0xui8>) {
  // CHECK: call @xla.gpu.conv.backward.input(
  // CHECK-SAME: %[[D_OUTPUT]], %[[FILTER]], %[[D_INPUT]], %[[SCRATCH]])
  lmhlo_gpu.conv_backwardinput(%d_output, %filter, %d_input, %scratch)
    dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
    window = { stride = [1, 1],
               lhs_dilate = [1, 1],
               rhs_dilate = [1, 1],
               reverse = [0, 0]
             }
    { backend_config = #lmhlo_gpu.convolution_backend_config<
        algorithm = 2,
        is_cudnn_frontend = true,
        knob_ids = [3, 2],
        knob_values = [0, 3],
        operand_0_layout = [3, 2, 1, 0],
        operand_1_layout = [3, 2, 1, 0],
        result_layout = [3, 2, 1, 0],
        tensor_ops_enabled = false,
        workspace_size = 0
      >,
      batch_group_count = 1 : i64,
      feature_group_count = 1 : i64,
      precision_config = [],
      result_scale = 1.000000e+00 : f64
    } : (memref<4x5x16x16xf64>,
         memref<5x3x7x7xf64>,
         memref<4x3x16x16xf64>,
         memref<0xui8>) -> ()
  return
}

// CHECK: func private @xla.gpu.conv.backward.input(
// CHECK-SAME:   memref<4x5x16x16xf64>, memref<5x3x7x7xf64>,
// CHECK-SAME:   memref<4x3x16x16xf64>, memref<0xui8>
// CHECK-SAME: ) attributes {rt.direct_custom_call =
// CHECK-SAME:              "xla.gpu.conv.backward.input"}

// -----

#map0 = affine_map<(d0, d1, d2, d3) -> (d0 * 3 + d1 + d2 * 9 + d3 * 9)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0 * 25 + d1 * 5 + d2 + d3 * 25)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0 * 800 + d1 * 5 + d2 + d3 * 25)>

// CHECK: @conv_forward_fused(
// CHECK:   %[[INPUT:[a-z0-9]+]]: memref
// CHECK:   %[[FILTER:[a-z0-9]+]]: memref
// CHECK:   %[[BIAS:[a-z0-9]+]]: memref
// CHECK:   %[[OUTPUT:[a-z0-9]+]]: memref
// CHECK:   %[[SCRATCH:[a-z0-9]+]]: memref
// CHECK: )
func.func @conv_forward_fused(%input: memref<8x5x5x1xf32, #map1>,
                              %filter: memref<3x3x1x32xf32, #map0>,
                              %bias: memref<32xf32>,
                              %output: memref<8x5x5x32xf32, #map2>,
                              %scratch: memref<0xui8>) {
  // CHECK: call @xla.gpu.conv.forward.fused(
  // CHECK-SAME: %[[INPUT]], %[[FILTER]], %[[BIAS]], %[[OUTPUT]], %[[SCRATCH]])

  // CHECK-DAG: activation_mode = #lmhlo_gpu<activation Relu>
  // CHECK-DAG: knob_ids = [2, 3]
  // CHECK-DAG: knob_values = [4, 0]
  lmhlo_gpu.conv_forward_fused(%input, %filter, %bias, %output, %scratch)
    dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
    window = { stride = [1, 1],
               lhs_dilate = [1, 1],
               rhs_dilate = [1, 1],
               reverse = [0, 0]
             }
    { activation_mode = #lmhlo_gpu<activation Relu>,
      backend_config = #lmhlo_gpu.convolution_backend_config<
        algorithm = 11,
        is_cudnn_frontend = true,
        knob_ids = [2, 3],
        knob_values = [4, 0],
        operand_0_layout = [2, 1, 3, 0],
        operand_1_layout = [1, 0, 2, 3],
        result_layout = [2, 1, 3, 0],
        tensor_ops_enabled = false,
        workspace_size = 0
      >,
      batch_group_count = 1 : i64,
      feature_group_count = 1 : i64,
      precision_config = [],
      result_scale = 1.000000e+00 : f64
    } : (memref<8x5x5x1xf32, #map1>,
         memref<3x3x1x32xf32, #map0>,
         memref<32xf32>,
         memref<8x5x5x32xf32, #map2>,
         memref<0xui8>) -> ()

  return
}

// CHECK: func private @xla.gpu.conv.forward.fused(
// CHECK-SAME:   memref<8x5x5x1xf32, #map0>, memref<3x3x1x32xf32, #map1>,
// CHECK-SAME:   memref<32xf32>, memref<8x5x5x32xf32, #map2>, memref<0xui8>
// CHECK-SAME: ) attributes {rt.direct_custom_call =
// CHECK-SAME:               "xla.gpu.conv.forward.fused"}

// -----

#map0 = affine_map<(d0, d1, d2, d3) -> (d0 * 576 + d1 * 3 + d2 + d3 * 9)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0 * 3 + d1 + d2 * 9 + d3 * 576)>

// CHECK: @conv_forward_fused_with_side_input(
// CHECK:   %[[INPUT:[a-z0-9]+]]: memref
// CHECK:   %[[FILTER:[a-z0-9]+]]: memref
// CHECK:   %[[BIAS:[a-z0-9]+]]: memref
// CHECK:   %[[SIDE_INPUT:[a-z0-9]+]]: memref
// CHECK:   %[[OUTPUT:[a-z0-9]+]]: memref
// CHECK:   %[[SCRATCH:[a-z0-9]+]]: memref
// CHECK: )
func.func @conv_forward_fused_with_side_input(
  %input: memref<1x3x3x64xf64, #map0>,
  %filter: memref<3x3x64x64xf64, #map1>,
  %bias: memref<64xf64>,
  %side_input: memref<1x3x3x64xf64, #map0>,
  %output: memref<1x3x3x64xf64, #map0>,
  %scratch: memref<0xui8>) {

  // CHECK: call @xla.gpu.conv.forward.fused.side_input(
  // CHECK-SAME: %[[INPUT]], %[[FILTER]], %[[BIAS]], %[[SIDE_INPUT]],
  // CHECK-SAME: %[[OUTPUT]], %[[SCRATCH]])

  // CHECK-DAG: activation_mode = #lmhlo_gpu<activation Relu>
  // CHECK-DAG: side_input_scale = 1.000000e+00 : f64
  lmhlo_gpu.conv_forward_fused_with_side_input(
      %input, %filter, %bias, %side_input, %output, %scratch)
    dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
    window = { stride = [1, 1],
               lhs_dilate = [1, 1],
               rhs_dilate = [1, 1],
               reverse = [0, 0]
             }
     { activation_mode = #lmhlo_gpu<activation Relu>,
       backend_config = #lmhlo_gpu.convolution_backend_config<
         algorithm = 0,
         is_cudnn_frontend = true,
         knob_ids = [],
         knob_values = [],
         operand_0_layout = [2, 1, 3, 0],
         operand_1_layout = [1, 0, 2, 3],
         result_layout = [2, 1, 3, 0],
         tensor_ops_enabled = false,
         workspace_size = 0
       >,
       batch_group_count = 1 : i64,
       feature_group_count = 1 : i64,
       precision_config = [],
       result_scale = 1.000000e+00 : f64,
       side_input_scale = 1.000000e+00 : f64
     } : (memref<1x3x3x64xf64, #map0>,
          memref<3x3x64x64xf64, #map1>,
          memref<64xf64>,
          memref<1x3x3x64xf64, #map0>,
          memref<1x3x3x64xf64, #map0>,
          memref<0xui8>) -> ()

  return
}

// CHECK: func private @xla.gpu.conv.forward.fused.side_input(
// CHECK-SAME:   memref<1x3x3x64xf64, #map0>, memref<3x3x64x64xf64, #map1>,
// CHECK-SAME:   memref<64xf64>, memref<1x3x3x64xf64, #map0>,
// CHECK-SAME:   memref<1x3x3x64xf64, #map0>, memref<0xui8>
// CHECK-SAME: ) attributes {rt.direct_custom_call =
// CHECK-SAME:               "xla.gpu.conv.forward.fused.side_input"}
