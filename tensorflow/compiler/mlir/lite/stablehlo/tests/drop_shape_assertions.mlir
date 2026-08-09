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
// RUN: odml-to-stablehlo-opt %s -drop-shape-assertions -split-input-file | FileCheck %s

// CHECK-LABEL: func.func @drop_vhlo_shape_assertion
func.func @drop_vhlo_shape_assertion(%arg0: tensor<i1>, %arg1: tensor<i32>) {
  // CHECK-NOT: vhlo.custom_call_v1
  // CHECK-NOT: shape_assertion
  "vhlo.custom_call_v1"(%arg0, %arg1) <{
    api_version = #vhlo<api_version_v1 API_VERSION_STATUS_RETURNING>,
    backend_config = #vhlo.string_v1<"">,
    call_target_name = #vhlo.string_v1<"shape_assertion">,
    called_computations = #vhlo.array_v1<[]>,
    has_side_effect = #vhlo.bool_v1<true>,
    operand_layouts = #vhlo.array_v1<[]>,
    output_operand_aliases = #vhlo.array_v1<[]>,
    result_layouts = #vhlo.array_v1<[]>
  }> {error_message = #vhlo.string_v1<"Input shapes do not match">} : (tensor<i1>, tensor<i32>) -> ()
  return
}

// -----

// CHECK-LABEL: func.func @drop_stablehlo_shape_assertion
func.func @drop_stablehlo_shape_assertion(%arg0: tensor<i1>, %arg1: tensor<i32>) {
  // CHECK-NOT: stablehlo.custom_call
  // CHECK-NOT: shape_assertion
  "stablehlo.custom_call"(%arg0, %arg1) <{
    call_target_name = "shape_assertion",
    has_side_effect = true
  }> : (tensor<i1>, tensor<i32>) -> ()
  return
}

// -----

// CHECK-LABEL: func.func @keep_other_custom_call
func.func @keep_other_custom_call(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: stablehlo.custom_call @some_other_call(%arg0)
  %0 = "stablehlo.custom_call"(%arg0) <{
    call_target_name = "some_other_call",
    has_side_effect = false
  }> : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}
