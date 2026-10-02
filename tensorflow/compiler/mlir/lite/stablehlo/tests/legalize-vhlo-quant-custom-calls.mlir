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
// RUN: odml-to-stablehlo-opt %s -legalize-vhlo-quant-custom-calls -split-input-file | FileCheck %s

// CHECK-LABEL: func.func @legalize_vhlo_quant_dequantize
func.func @legalize_vhlo_quant_dequantize(%arg0: tensor<4xi8>, %arg1: tensor<f32>, %arg2: tensor<i32>) -> tensor<4xf32> {
  // CHECK-NOT: vhlo.custom_call_v1
  // CHECK: %[[RES:.*]] = stablehlo.custom_call @quant.dequantize(%arg0, %arg1, %arg2) {axis = 0 : i32} : (tensor<4xi8>, tensor<f32>, tensor<i32>) -> tensor<4xf32>
  // CHECK: return %[[RES]] : tensor<4xf32>
  %0 = "vhlo.custom_call_v1"(%arg0, %arg1, %arg2) <{
    api_version = #vhlo<api_version_v1 API_VERSION_ORIGINAL>,
    backend_config = #vhlo.string_v1<"">,
    call_target_name = #vhlo.string_v1<"quant.dequantize">,
    called_computations = #vhlo.array_v1<[]>,
    has_side_effect = #vhlo.bool_v1<false>,
    operand_layouts = #vhlo.array_v1<[]>,
    output_operand_aliases = #vhlo.array_v1<[]>,
    result_layouts = #vhlo.array_v1<[]>
  }> {axis = #vhlo.integer_v1<0 : i32>} : (tensor<4xi8>, tensor<f32>, tensor<i32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

// CHECK-LABEL: func.func @legalize_vhlo_quant_quantize
func.func @legalize_vhlo_quant_quantize(%arg0: tensor<4xf32>, %arg1: tensor<f32>, %arg2: tensor<i32>) -> tensor<4xi8> {
  // CHECK-NOT: vhlo.custom_call_v1
  // CHECK: %[[RES:.*]] = stablehlo.custom_call @quant.quantize(%arg0, %arg1, %arg2) : (tensor<4xf32>, tensor<f32>, tensor<i32>) -> tensor<4xi8>
  // CHECK: return %[[RES]] : tensor<4xi8>
  %0 = "vhlo.custom_call_v1"(%arg0, %arg1, %arg2) <{
    api_version = #vhlo<api_version_v1 API_VERSION_ORIGINAL>,
    backend_config = #vhlo.string_v1<"">,
    call_target_name = #vhlo.string_v1<"quant.quantize">,
    called_computations = #vhlo.array_v1<[]>,
    has_side_effect = #vhlo.bool_v1<false>,
    operand_layouts = #vhlo.array_v1<[]>,
    output_operand_aliases = #vhlo.array_v1<[]>,
    result_layouts = #vhlo.array_v1<[]>
  }> : (tensor<4xf32>, tensor<f32>, tensor<i32>) -> tensor<4xi8>
  return %0 : tensor<4xi8>
}

// -----

// CHECK-LABEL: func.func @legalize_vhlo_quant_fake_quant
func.func @legalize_vhlo_quant_fake_quant(%arg0: tensor<4xf32>, %arg1: tensor<f32>, %arg2: tensor<i32>) -> tensor<4xf32> {
  // CHECK-NOT: vhlo.custom_call_v1
  // CHECK: %[[RES:.*]] = stablehlo.custom_call @quant.fake_quant(%arg0, %arg1, %arg2) {narrow_range = false} : (tensor<4xf32>, tensor<f32>, tensor<i32>) -> tensor<4xf32>
  // CHECK: return %[[RES]] : tensor<4xf32>
  %0 = "vhlo.custom_call_v1"(%arg0, %arg1, %arg2) <{
    api_version = #vhlo<api_version_v1 API_VERSION_ORIGINAL>,
    backend_config = #vhlo.string_v1<"">,
    call_target_name = #vhlo.string_v1<"quant.fake_quant">,
    called_computations = #vhlo.array_v1<[]>,
    has_side_effect = #vhlo.bool_v1<false>,
    operand_layouts = #vhlo.array_v1<[]>,
    output_operand_aliases = #vhlo.array_v1<[]>,
    result_layouts = #vhlo.array_v1<[]>
  }> {narrow_range = #vhlo.bool_v1<false>} : (tensor<4xf32>, tensor<f32>, tensor<i32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

// CHECK-LABEL: func.func @keep_other_vhlo_custom_call
func.func @keep_other_vhlo_custom_call(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: "vhlo.custom_call_v1"(%arg0)
  // CHECK-SAME: call_target_name = #vhlo.string_v1<"other.custom_call">
  %0 = "vhlo.custom_call_v1"(%arg0) <{
    api_version = #vhlo<api_version_v1 API_VERSION_ORIGINAL>,
    backend_config = #vhlo.string_v1<"">,
    call_target_name = #vhlo.string_v1<"other.custom_call">,
    called_computations = #vhlo.array_v1<[]>,
    has_side_effect = #vhlo.bool_v1<false>,
    operand_layouts = #vhlo.array_v1<[]>,
    output_operand_aliases = #vhlo.array_v1<[]>,
    result_layouts = #vhlo.array_v1<[]>
  }> : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}
