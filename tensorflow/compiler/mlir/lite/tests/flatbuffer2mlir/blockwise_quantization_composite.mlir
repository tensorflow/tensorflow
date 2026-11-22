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
// RUN: flatbuffer_translate -mlir-to-tflite-flatbuffer --emit-stablehlo-ops=true %s -o - | flatbuffer_translate --tflite-flatbuffer-to-mlir - -o - | FileCheck %s
// RUN: flatbuffer_translate -mlir-to-tflite-flatbuffer --emit-stablehlo-ops=true --use-buffer-offset %s -o - | flatbuffer_translate --tflite-flatbuffer-to-mlir - -o - | FileCheck %s

// Round trips a `STABLEHLO_COMPOSITE` op taking a `!tfl.blockwise_quant`
// constant operand. Such types have no StableHLO equivalent, so the composite
// is imported back as a VHLO composite op, while other VHLO ops are still
// legalized to StableHLO. Blockwise quantized function arguments and results
// keep their quantized type.

// CHECK-LABEL: func.func @main(
// CHECK-SAME:    %[[ARG0:.*]]: tensor<2x4xf32>) -> tensor<2x4xf32>
// CHECK-NOT:     unrealized_conversion_cast
// CHECK-DAG:     %[[W:.*]] = "tfl.pseudo_qconst"() <{qtype = tensor<2x4x!tfl.blockwise_quant<i2 -> f32, block_shape = [1, 4], scales = dense<{{\[}}[5.000000e-01], [2.500000e-01]]> : tensor<2x1xf32>, zero_points = dense<-5.000000e-01> : tensor<1x1xf32>>>
// CHECK-DAG:     %[[LOGISTIC:.*]] = stablehlo.logistic %[[ARG0]] : tensor<2x4xf32>
// CHECK:         %[[RES:.*]] = "vhlo.composite_v{{[0-9]+}}"(%[[LOGISTIC]], %[[W]])
// CHECK-SAME:      decomposition = #vhlo.string_v1<"test.blockwise.impl">
// CHECK-SAME:      name = #vhlo.string_v1<"odml.test_blockwise">
// CHECK-SAME:      : (tensor<2x4xf32>, tensor<2x4x!tfl.blockwise_quant<{{.*}}>>) -> tensor<2x4xf32>
// CHECK:         %[[OUT:.*]] = stablehlo.logistic %[[RES]] : tensor<2x4xf32>
// CHECK:         return %[[OUT]] : tensor<2x4xf32>
// CHECK-LABEL: func.func private @test.blockwise.impl(
// CHECK-SAME:    %{{.*}}: tensor<2x4xf32>, %[[D_ARG1:.*]]: tensor<2x4x!tfl.blockwise_quant<i2 -> f32, block_shape = [1, 4], scales = dense<{{\[}}[5.000000e-01], [2.500000e-01]]> : tensor<2x1xf32>, zero_points = dense<-5.000000e-01> : tensor<1x1xf32>>>) -> tensor<2x4xf32>
// CHECK:         "tfl.dequantize"(%[[D_ARG1]])
func.func @main(%arg0: tensor<2x4xf32>) -> tensor<2x4xf32> {
  %w = "tfl.pseudo_qconst"() {qtype = tensor<2x4x!tfl.blockwise_quant<i2 -> f32, block_shape = [1, 4], scales = dense<[[0.5], [0.25]]> : tensor<2x1xf32>, zero_points = dense<-0.5> : tensor<1x1xf32>>>, value = dense<1> : tensor<2x4xi2>} : () -> tensor<2x4x!tfl.blockwise_quant<i2 -> f32, block_shape = [1, 4], scales = dense<[[0.5], [0.25]]> : tensor<2x1xf32>, zero_points = dense<-0.5> : tensor<1x1xf32>>>
  %0 = "vhlo.logistic_v1"(%arg0) : (tensor<2x4xf32>) -> tensor<2x4xf32>
  %1 = "vhlo.composite_v1"(%0, %w) <{composite_attributes = #vhlo.dict_v1<{}>, decomposition = #vhlo.string_v1<"test.blockwise.impl">, name = #vhlo.string_v1<"odml.test_blockwise">, version = #vhlo.integer_v1<0 : i64>}> : (tensor<2x4xf32>, tensor<2x4x!tfl.blockwise_quant<i2 -> f32, block_shape = [1, 4], scales = dense<[[0.5], [0.25]]> : tensor<2x1xf32>, zero_points = dense<-0.5> : tensor<1x1xf32>>>) -> tensor<2x4xf32>
  %2 = "vhlo.logistic_v1"(%1) : (tensor<2x4xf32>) -> tensor<2x4xf32>
  func.return %2 : tensor<2x4xf32>
}
func.func private @test.blockwise.impl(%arg0: tensor<2x4xf32>, %arg1: tensor<2x4x!tfl.blockwise_quant<i2 -> f32, block_shape = [1, 4], scales = dense<[[0.5], [0.25]]> : tensor<2x1xf32>, zero_points = dense<-0.5> : tensor<1x1xf32>>>) -> tensor<2x4xf32> {
  %0 = "tfl.dequantize"(%arg1) : (tensor<2x4x!tfl.blockwise_quant<i2 -> f32, block_shape = [1, 4], scales = dense<[[0.5], [0.25]]> : tensor<2x1xf32>, zero_points = dense<-0.5> : tensor<1x1xf32>>>) -> tensor<2x4xf32>
  %1 = tfl.add %arg0, %0 {fused_activation_function = "NONE"} : tensor<2x4xf32>
  func.return %1 : tensor<2x4xf32>
}

// Returns a blockwise quantized argument and constant.

// CHECK-LABEL: func.func private @return_blockwise(
// CHECK-SAME:    %[[R_ARG0:.*]]: tensor<2x4x!tfl.blockwise_quant<i4 -> f32, block_shape = [1, 4], scales = dense<{{\[}}[5.000000e-01], [2.500000e-01]]> : tensor<2x1xf32>>>)
// CHECK-SAME:    -> (tensor<2x4x!tfl.blockwise_quant<i4 -> f32, block_shape = [1, 4], scales = dense<{{\[}}[5.000000e-01], [2.500000e-01]]> : tensor<2x1xf32>>>, tensor<2x4x!tfl.blockwise_quant<i4 -> f32, block_shape = [1, 4], scales = dense<{{\[}}[1.000000e+00], [2.000000e+00]]> : tensor<2x1xf32>>>)
// CHECK:         %[[R_W:.*]] = "tfl.pseudo_qconst"()
// CHECK:         return %[[R_ARG0]], %[[R_W]]
func.func private @return_blockwise(%arg0: tensor<2x4x!tfl.blockwise_quant<i4 -> f32, block_shape = [1, 4], scales = dense<[[0.5], [0.25]]> : tensor<2x1xf32>>>) -> (tensor<2x4x!tfl.blockwise_quant<i4 -> f32, block_shape = [1, 4], scales = dense<[[0.5], [0.25]]> : tensor<2x1xf32>>>, tensor<2x4x!tfl.blockwise_quant<i4 -> f32, block_shape = [1, 4], scales = dense<[[1.0], [2.0]]> : tensor<2x1xf32>>>) {
  %w = "tfl.pseudo_qconst"() {qtype = tensor<2x4x!tfl.blockwise_quant<i4 -> f32, block_shape = [1, 4], scales = dense<[[1.0], [2.0]]> : tensor<2x1xf32>>>, value = dense<3> : tensor<2x4xi4>} : () -> tensor<2x4x!tfl.blockwise_quant<i4 -> f32, block_shape = [1, 4], scales = dense<[[1.0], [2.0]]> : tensor<2x1xf32>>>
  func.return %arg0, %w : tensor<2x4x!tfl.blockwise_quant<i4 -> f32, block_shape = [1, 4], scales = dense<[[0.5], [0.25]]> : tensor<2x1xf32>>>, tensor<2x4x!tfl.blockwise_quant<i4 -> f32, block_shape = [1, 4], scales = dense<[[1.0], [2.0]]> : tensor<2x1xf32>>>
}
