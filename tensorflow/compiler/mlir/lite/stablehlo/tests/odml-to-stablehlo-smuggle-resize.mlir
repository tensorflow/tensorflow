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
// RUN: odml_to_stablehlo %s -skip-resize -smuggle-disallowed-ops -o - | FileCheck %s
// RUN: odml-to-stablehlo-opt %s --smuggle-disallowed-ops-pass | FileCheck %s --check-prefix=CHECK-OPT

// CHECK-LABEL: @main
module attributes {tf.versions = {bad_consumers = [], min_consumer = 12 : i32, producer = 975 : i32}, tf_saved_model.semantics}  {
  func.func @serving_default(%arg0: tensor<1x32x32x128xf32> {tf_saved_model.index_path = ["a"]}) -> (tensor<1x64x64x128xf32> {tf_saved_model.index_path = ["b"]}) attributes {tf.entry_function = {control_outputs = "", inputs = "c:0", outputs = "d:0"}, tf_saved_model.exported_names = ["serving_default"]} {
      %0  = "tf.Const"() {value = dense<[56, 904]> : tensor<2xi32>} : () -> tensor<2xi32>
      // CHECK: %{{.*}} = stablehlo.custom_call @tf.ResizeBilinear(%arg0, %{{.*}}) {align_corners = false, device = "", half_pixel_centers = true} : (tensor<1x32x32x128xf32>, tensor<2xi32>) -> tensor<1x64x64x128xf32>
      // CHECK-OPT: %{{.*}} = stablehlo.custom_call @tf.ResizeBilinear(%arg0, %cst) {align_corners = false, device = "", half_pixel_centers = true} : (tensor<1x32x32x128xf32>, tensor<2xi32>) -> tensor<1x64x64x128xf32>
      %1 = "tf.ResizeBilinear"(%arg0, %0) {
        align_corners = false, device = "", half_pixel_centers = true
      } : (tensor<1x32x32x128xf32>, tensor<2xi32>) -> tensor<1x64x64x128xf32>
      func.return %1 : tensor<1x64x64x128xf32>
  }
}
