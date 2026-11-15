// Copyright 2026 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ==============================================================================

// Fusing a scalar binary op into the following affine op when the filter (and
// bias) are DenseResourceElementsAttr constants. The filter stays a resource.

// RUN: litert-opt %s -tfl-optimize='enable-canonicalization=false' -split-input-file | FileCheck %s

// Filter [[1, 2], [3, 4]]. Sub folds into a new bias, -0.5 * row sums.
// CHECK-LABEL: func.func @fuseSubIntoFcWithResourceFilter
func.func @fuseSubIntoFcWithResourceFilter(%arg0: tensor<4x2xf32>) -> tensor<4x2xf32> {
  %c = arith.constant dense<5.000000e-01> : tensor<f32>
  %0 = tfl.sub(%arg0, %c) <{fused_activation_function = "NONE"}> : (tensor<4x2xf32>, tensor<f32>) -> tensor<4x2xf32>
  %w = arith.constant dense_resource<w> : tensor<2x2xf32>
  %b = "tfl.no_value"() {value} : () -> none
  %1 = "tfl.fully_connected"(%0, %w, %b) <{fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"}> : (tensor<4x2xf32>, tensor<2x2xf32>, none) -> tensor<4x2xf32>
  func.return %1 : tensor<4x2xf32>

  // CHECK-DAG: %[[W:.*]] = arith.constant dense_resource<w> : tensor<2x2xf32>
  // CHECK-DAG: %[[B:.*]] = "tfl.pseudo_const"() <{value = dense<[-1.500000e+00, -3.500000e+00]> : tensor<2xf32>}>
  // CHECK-NOT: tfl.sub
  // CHECK: %[[FC:.*]] = "tfl.fully_connected"(%arg0, %[[W]], %[[B]])
  // CHECK: return %[[FC]]
}

{-#
  dialect_resources: {
    builtin: {
      w: "0x040000000000803F000000400000404000008040"
    }
  }
#-}

// -----

// Mul scales the filter into a new resource; the bias is unchanged.
// CHECK-LABEL: func.func @fuseMulIntoFcWithResourceFilter
func.func @fuseMulIntoFcWithResourceFilter(%arg0: tensor<4x2xf32>) -> tensor<4x2xf32> {
  %c = arith.constant dense<2.000000e+00> : tensor<f32>
  %0 = tfl.mul(%arg0, %c) <{fused_activation_function = "NONE"}> : (tensor<4x2xf32>, tensor<f32>) -> tensor<4x2xf32>
  %w = arith.constant dense_resource<w> : tensor<2x2xf32>
  %b = arith.constant dense<1.000000e+00> : tensor<2xf32>
  %1 = "tfl.fully_connected"(%0, %w, %b) <{fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"}> : (tensor<4x2xf32>, tensor<2x2xf32>, tensor<2xf32>) -> tensor<4x2xf32>
  func.return %1 : tensor<4x2xf32>

  // CHECK-DAG: %[[B:.*]] = arith.constant dense<1.000000e+00> : tensor<2xf32>
  // CHECK-DAG: %[[W:.*]] = "tfl.pseudo_const"() <{value = dense_resource<w_mul> : tensor<2x2xf32>}>
  // CHECK-NOT: tfl.mul
  // CHECK: %[[FC:.*]] = "tfl.fully_connected"(%arg0, %[[W]], %[[B]])
  // CHECK: return %[[FC]]
}

// [2, 4, 6, 8]
// CHECK: w_mul: "0x400000000000004000008040
// CHECK-SAME: 0000C04000000041"

{-#
  dialect_resources: {
    builtin: {
      w: "0x040000000000803F000000400000404000008040"
    }
  }
#-}

// -----

// An input affine map, (x - 0.5) * 2, like an image normalization: both ops
// fuse. W' = [[2, 4], [6, 8]] and b' = -0.5 * row sums of W'.
// CHECK-LABEL: func.func @fuseSubThenMulIntoFcWithResourceFilter
func.func @fuseSubThenMulIntoFcWithResourceFilter(%arg0: tensor<4x2xf32>) -> tensor<4x2xf32> {
  %c0 = arith.constant dense<5.000000e-01> : tensor<f32>
  %c1 = arith.constant dense<2.000000e+00> : tensor<f32>
  %0 = tfl.sub(%arg0, %c0) <{fused_activation_function = "NONE"}> : (tensor<4x2xf32>, tensor<f32>) -> tensor<4x2xf32>
  %1 = tfl.mul(%0, %c1) <{fused_activation_function = "NONE"}> : (tensor<4x2xf32>, tensor<f32>) -> tensor<4x2xf32>
  %w = arith.constant dense_resource<w> : tensor<2x2xf32>
  %b = "tfl.no_value"() {value} : () -> none
  %2 = "tfl.fully_connected"(%1, %w, %b) <{fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"}> : (tensor<4x2xf32>, tensor<2x2xf32>, none) -> tensor<4x2xf32>
  func.return %2 : tensor<4x2xf32>

  // CHECK-DAG: %[[W:.*]] = "tfl.pseudo_const"() <{value = dense_resource<w_mul> : tensor<2x2xf32>}>
  // CHECK-DAG: %[[B:.*]] = "tfl.pseudo_const"() <{value = dense<[-3.000000e+00, -7.000000e+00]> : tensor<2xf32>}>
  // CHECK-NOT: tfl.sub
  // CHECK-NOT: tfl.mul
  // CHECK: %[[FC:.*]] = "tfl.fully_connected"(%arg0, %[[W]], %[[B]])
  // CHECK: return %[[FC]]
}

{-#
  dialect_resources: {
    builtin: {
      w: "0x040000000000803F000000400000404000008040"
    }
  }
#-}

// -----

// A resource bias is read from its blob: [10, 20] + 1.0 * row sums.
// CHECK-LABEL: func.func @fuseAddIntoFcWithResourceBias
func.func @fuseAddIntoFcWithResourceBias(%arg0: tensor<4x2xf32>) -> tensor<4x2xf32> {
  %c = arith.constant dense<1.000000e+00> : tensor<f32>
  %0 = tfl.add(%arg0, %c) <{fused_activation_function = "NONE"}> : (tensor<4x2xf32>, tensor<f32>) -> tensor<4x2xf32>
  %w = arith.constant dense_resource<w> : tensor<2x2xf32>
  %b = arith.constant dense_resource<b> : tensor<2xf32>
  %1 = "tfl.fully_connected"(%0, %w, %b) <{fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"}> : (tensor<4x2xf32>, tensor<2x2xf32>, tensor<2xf32>) -> tensor<4x2xf32>
  func.return %1 : tensor<4x2xf32>

  // CHECK-DAG: %[[W:.*]] = arith.constant dense_resource<w> : tensor<2x2xf32>
  // CHECK-DAG: %[[B:.*]] = "tfl.pseudo_const"() <{value = dense<[1.300000e+01, 2.700000e+01]> : tensor<2xf32>}>
  // CHECK-NOT: tfl.add
  // CHECK: %[[FC:.*]] = "tfl.fully_connected"(%arg0, %[[W]], %[[B]])
  // CHECK: return %[[FC]]
}

{-#
  dialect_resources: {
    builtin: {
      w: "0x040000000000803F000000400000404000008040",
      b: "0x04000000000020410000A041"
    }
  }
#-}

// -----

// A filter shared with another op is not modified; the fused op gets a copy.
// CHECK-LABEL: func.func @fuseMulIntoFcWithSharedResourceFilter
func.func @fuseMulIntoFcWithSharedResourceFilter(%arg0: tensor<4x2xf32>, %arg1: tensor<4x2xf32>) -> (tensor<4x2xf32>, tensor<4x2xf32>) {
  %c = arith.constant dense<2.000000e+00> : tensor<f32>
  %0 = tfl.mul(%arg0, %c) <{fused_activation_function = "NONE"}> : (tensor<4x2xf32>, tensor<f32>) -> tensor<4x2xf32>
  %w = arith.constant dense_resource<w> : tensor<2x2xf32>
  %b = "tfl.no_value"() {value} : () -> none
  %1 = "tfl.fully_connected"(%0, %w, %b) <{fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"}> : (tensor<4x2xf32>, tensor<2x2xf32>, none) -> tensor<4x2xf32>
  %2 = "tfl.fully_connected"(%arg1, %w, %b) <{fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"}> : (tensor<4x2xf32>, tensor<2x2xf32>, none) -> tensor<4x2xf32>
  func.return %1, %2 : tensor<4x2xf32>, tensor<4x2xf32>

  // CHECK-DAG: %[[W:.*]] = arith.constant dense_resource<w> : tensor<2x2xf32>
  // CHECK-DAG: %[[W2:.*]] = "tfl.pseudo_const"() <{value = dense_resource<w_mul> : tensor<2x2xf32>}>
  // CHECK-DAG: %[[FC1:.*]] = "tfl.fully_connected"(%arg0, %[[W2]],
  // CHECK-DAG: %[[FC2:.*]] = "tfl.fully_connected"(%arg1, %[[W]],
  // CHECK: return %[[FC1]], %[[FC2]]
}

// CHECK-DAG: w: "0x040000000000803F000000400000404000008040"
// CHECK-DAG: w_mul: "0x400000000000004000008040

{-#
  dialect_resources: {
    builtin: {
      w: "0x040000000000803F000000400000404000008040"
    }
  }
#-}

// -----

// A resource without data cannot be read, so nothing is fused.
// CHECK-LABEL: func.func @doNotFuseIntoFcWithBloblessResourceFilter
func.func @doNotFuseIntoFcWithBloblessResourceFilter(%arg0: tensor<4x2xf32>) -> tensor<4x2xf32> {
  %c = arith.constant dense<2.000000e+00> : tensor<f32>
  %0 = tfl.mul(%arg0, %c) <{fused_activation_function = "NONE"}> : (tensor<4x2xf32>, tensor<f32>) -> tensor<4x2xf32>
  %w = arith.constant dense_resource<__elided__> : tensor<2x2xf32>
  %b = "tfl.no_value"() {value} : () -> none
  %1 = "tfl.fully_connected"(%0, %w, %b) <{fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"}> : (tensor<4x2xf32>, tensor<2x2xf32>, none) -> tensor<4x2xf32>
  func.return %1 : tensor<4x2xf32>

  // CHECK: tfl.mul
  // CHECK: "tfl.fully_connected"
}

// -----

// Div by a constant is first rewritten into a mul by its reciprocal, which then
// scales the filter into a new resource: [0.5, 1, 1.5, 2].
// CHECK-LABEL: func.func @fuseDivIntoFcWithResourceFilter
func.func @fuseDivIntoFcWithResourceFilter(%arg0: tensor<4x2xf32>) -> tensor<4x2xf32> {
  %c = arith.constant dense<2.000000e+00> : tensor<f32>
  %0 = tfl.div(%arg0, %c) <{fused_activation_function = "NONE"}> : (tensor<4x2xf32>, tensor<f32>) -> tensor<4x2xf32>
  %w = arith.constant dense_resource<w> : tensor<2x2xf32>
  %b = "tfl.no_value"() {value} : () -> none
  %1 = "tfl.fully_connected"(%0, %w, %b) <{fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"}> : (tensor<4x2xf32>, tensor<2x2xf32>, none) -> tensor<4x2xf32>
  func.return %1 : tensor<4x2xf32>

  // CHECK-DAG: %[[W:.*]] = "tfl.pseudo_const"() <{value = dense_resource<w_mul> : tensor<2x2xf32>}>
  // CHECK-NOT: tfl.div
  // CHECK: %[[FC:.*]] = "tfl.fully_connected"(%arg0, %[[W]],
  // CHECK: return %[[FC]]
}

// CHECK: w_mul: "0x400000000000003F0000803F0000C03F00000040"

{-#
  dialect_resources: {
    builtin: {
      w: "0x040000000000803F000000400000404000008040"
    }
  }
#-}

// -----

// A scalar resource bias is broadcast: 10 + 1.0 * row sums = [13, 17].
// CHECK-LABEL: func.func @fuseAddIntoFcWithScalarResourceBias
func.func @fuseAddIntoFcWithScalarResourceBias(%arg0: tensor<4x2xf32>) -> tensor<4x2xf32> {
  %c = arith.constant dense<1.000000e+00> : tensor<f32>
  %0 = tfl.add(%arg0, %c) <{fused_activation_function = "NONE"}> : (tensor<4x2xf32>, tensor<f32>) -> tensor<4x2xf32>
  %w = arith.constant dense_resource<w> : tensor<2x2xf32>
  %b = arith.constant dense_resource<b> : tensor<1xf32>
  %1 = "tfl.fully_connected"(%0, %w, %b) <{fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"}> : (tensor<4x2xf32>, tensor<2x2xf32>, tensor<1xf32>) -> tensor<4x2xf32>
  func.return %1 : tensor<4x2xf32>

  // CHECK-DAG: %[[B:.*]] = "tfl.pseudo_const"() <{value = dense<[1.300000e+01, 1.700000e+01]> : tensor<2xf32>}>
  // CHECK-NOT: tfl.add
  // CHECK: "tfl.fully_connected"(%arg0, %{{.*}}, %[[B]])
}

{-#
  dialect_resources: {
    builtin: {
      w: "0x040000000000803F000000400000404000008040",
      b: "0x0400000000002041"
    }
  }
#-}

// -----

// conv_2d: the bias channel is the first filter dimension, each channel sums
// H * W * Cin = 2 filter values: [10, 20] + 1.0 * [1 + 2, 3 + 4] = [13, 27].
// CHECK-LABEL: func.func @fuseAddIntoConv2DWithResourceFilter
func.func @fuseAddIntoConv2DWithResourceFilter(%arg0: tensor<1x1x1x2xf32>) -> tensor<1x1x1x2xf32> {
  %c = arith.constant dense<1.000000e+00> : tensor<f32>
  %0 = tfl.add(%arg0, %c) <{fused_activation_function = "NONE"}> : (tensor<1x1x1x2xf32>, tensor<f32>) -> tensor<1x1x1x2xf32>
  %w = arith.constant dense_resource<w> : tensor<2x1x1x2xf32>
  %b = arith.constant dense<[1.000000e+01, 2.000000e+01]> : tensor<2xf32>
  %1 = "tfl.conv_2d"(%0, %w, %b) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x1x1x2xf32>, tensor<2x1x1x2xf32>, tensor<2xf32>) -> tensor<1x1x1x2xf32>
  func.return %1 : tensor<1x1x1x2xf32>

  // CHECK-DAG: %[[W:.*]] = arith.constant dense_resource<w> : tensor<2x1x1x2xf32>
  // CHECK-DAG: %[[B:.*]] = "tfl.pseudo_const"() <{value = dense<[1.300000e+01, 2.700000e+01]> : tensor<2xf32>}>
  // CHECK-NOT: tfl.add
  // CHECK: "tfl.conv_2d"(%arg0, %[[W]], %[[B]])
}

{-#
  dialect_resources: {
    builtin: {
      w: "0x040000000000803F000000400000404000008040"
    }
  }
#-}

// -----

// depthwise_conv_2d: the bias channel is the last filter dimension, so filter
// [1, 2, 3, 4] of shape 1x1x2x2 gives [0, 0] + 1.0 * [1 + 3, 2 + 4] = [4, 6].
// CHECK-LABEL: func.func @fuseAddIntoDepthwiseConv2DWithResourceFilter
func.func @fuseAddIntoDepthwiseConv2DWithResourceFilter(%arg0: tensor<1x1x2x2xf32>) -> tensor<1x1x1x2xf32> {
  %c = arith.constant dense<1.000000e+00> : tensor<f32>
  %0 = tfl.add(%arg0, %c) <{fused_activation_function = "NONE"}> : (tensor<1x1x2x2xf32>, tensor<f32>) -> tensor<1x1x2x2xf32>
  %w = arith.constant dense_resource<w> : tensor<1x1x2x2xf32>
  %b = arith.constant dense<0.000000e+00> : tensor<2xf32>
  %1 = "tfl.depthwise_conv_2d"(%0, %w, %b) {depth_multiplier = 1 : i32, dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x1x2x2xf32>, tensor<1x1x2x2xf32>, tensor<2xf32>) -> tensor<1x1x1x2xf32>
  func.return %1 : tensor<1x1x1x2xf32>

  // CHECK-DAG: %[[B:.*]] = "tfl.pseudo_const"() <{value = dense<[4.000000e+00, 6.000000e+00]> : tensor<2xf32>}>
  // CHECK-NOT: tfl.add
  // CHECK: "tfl.depthwise_conv_2d"(%arg0, %{{.*}}, %[[B]])
}

{-#
  dialect_resources: {
    builtin: {
      w: "0x040000000000803F000000400000404000008040"
    }
  }
#-}

// -----

// Add before a conv_2d with SAME padding is not fused (padding sees the
// constant), as in the dense path.
// CHECK-LABEL: func.func @doNotFuseAddIntoSamePaddingConv2DWithResourceFilter
func.func @doNotFuseAddIntoSamePaddingConv2DWithResourceFilter(%arg0: tensor<1x1x1x2xf32>) -> tensor<1x1x1x2xf32> {
  %c = arith.constant dense<1.000000e+00> : tensor<f32>
  %0 = tfl.add(%arg0, %c) <{fused_activation_function = "NONE"}> : (tensor<1x1x1x2xf32>, tensor<f32>) -> tensor<1x1x1x2xf32>
  %w = arith.constant dense_resource<w> : tensor<2x1x1x2xf32>
  %b = arith.constant dense<0.000000e+00> : tensor<2xf32>
  %1 = "tfl.conv_2d"(%0, %w, %b) {dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "SAME", stride_h = 1 : i32, stride_w = 1 : i32} : (tensor<1x1x1x2xf32>, tensor<2x1x1x2xf32>, tensor<2xf32>) -> tensor<1x1x1x2xf32>
  func.return %1 : tensor<1x1x1x2xf32>

  // CHECK: tfl.add
  // CHECK: "tfl.conv_2d"
}

{-#
  dialect_resources: {
    builtin: {
      w: "0x040000000000803F000000400000404000008040"
    }
  }
#-}

// -----

// A quantized resource filter (through quantize and dequantize ops) is not
// handled by the resource path.
// CHECK-LABEL: func.func @doNotFuseMulIntoFcWithQuantizedResourceFilter
func.func @doNotFuseMulIntoFcWithQuantizedResourceFilter(%arg0: tensor<4x2xf32>) -> tensor<4x2xf32> {
  %c = arith.constant dense<2.000000e+00> : tensor<f32>
  %0 = tfl.mul(%arg0, %c) <{fused_activation_function = "NONE"}> : (tensor<4x2xf32>, tensor<f32>) -> tensor<4x2xf32>
  %w = arith.constant dense_resource<w> : tensor<2x2xf32>
  %q = "tfl.quantize"(%w) {qtype = tensor<2x2x!quant.uniform<i8:f32, 1.000000e+00>>} : (tensor<2x2xf32>) -> tensor<2x2x!quant.uniform<i8:f32, 1.000000e+00>>
  %dq = "tfl.dequantize"(%q) : (tensor<2x2x!quant.uniform<i8:f32, 1.000000e+00>>) -> tensor<2x2xf32>
  %b = "tfl.no_value"() {value} : () -> none
  %1 = "tfl.fully_connected"(%0, %dq, %b) <{fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"}> : (tensor<4x2xf32>, tensor<2x2xf32>, none) -> tensor<4x2xf32>
  func.return %1 : tensor<4x2xf32>

  // CHECK: tfl.mul
  // CHECK: "tfl.fully_connected"
}

{-#
  dialect_resources: {
    builtin: {
      w: "0x040000000000803F000000400000404000008040"
    }
  }
#-}

// -----

// A dense filter with a resource bias is not fused (neither path handles it).
// CHECK-LABEL: func.func @doNotFuseAddIntoFcWithDenseFilterAndResourceBias
func.func @doNotFuseAddIntoFcWithDenseFilterAndResourceBias(%arg0: tensor<4x2xf32>) -> tensor<4x2xf32> {
  %c = arith.constant dense<1.000000e+00> : tensor<f32>
  %0 = tfl.add(%arg0, %c) <{fused_activation_function = "NONE"}> : (tensor<4x2xf32>, tensor<f32>) -> tensor<4x2xf32>
  %w = arith.constant dense<[[1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00]]> : tensor<2x2xf32>
  %b = arith.constant dense_resource<b> : tensor<2xf32>
  %1 = "tfl.fully_connected"(%0, %w, %b) <{fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"}> : (tensor<4x2xf32>, tensor<2x2xf32>, tensor<2xf32>) -> tensor<4x2xf32>
  func.return %1 : tensor<4x2xf32>

  // CHECK: tfl.add
  // CHECK: "tfl.fully_connected"
}

{-#
  dialect_resources: {
    builtin: {
      b: "0x04000000000020410000A041"
    }
  }
#-}
