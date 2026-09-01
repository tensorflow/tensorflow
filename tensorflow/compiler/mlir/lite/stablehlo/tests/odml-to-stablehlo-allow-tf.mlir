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
// RUN: odml_to_stablehlo %s --allow-tf=false -o /tmp/temp.mlir; [ -f /tmp/temp.mlir ]; [ -f /tmp/debug_stablehlo.mlir ]
// RUN: odml_to_stablehlo %s --allow-tf=true -o /tmp/temp2.mlir; [ -f /tmp/temp2.mlir ]

module attributes {tf.versions = {bad_consumers = [], min_consumer = 12 : i32, producer = 975 : i32}, tf_saved_model.semantics}  {
  func.func @serving_default(%arg0: tensor<1x20x20x28xf32> {tf_saved_model.index_path = ["a"]}) -> (tensor<1x40x40x28xf32> {tf_saved_model.index_path = ["b"]}) attributes {tf.entry_function = {control_outputs = "", inputs = "c:0", outputs = "d:0"}, tf_saved_model.exported_names = ["serving_default"]} {
      %c = stablehlo.constant dense<40> : tensor<2xi32>
      %0 = "tf.UnconvertedOp"(%arg0, %c) {align_corners = false, half_pixel_centers = false} : (tensor<1x20x20x28xf32>, tensor<2xi32>) -> tensor<1x40x40x28xf32>
      func.return %0 : tensor<1x40x40x28xf32>
  }
}
