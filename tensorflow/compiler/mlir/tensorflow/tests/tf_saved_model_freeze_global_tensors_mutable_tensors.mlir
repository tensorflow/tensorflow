// Copyright 2026 Google Inc. All Rights Reserved.
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
// RUN: tf-opt "-tf-saved-model-freeze-global-tensors=allow-mutable-tensors=true" -split-input-file %s | FileCheck %s

module attributes {tf_saved_model.semantics} {

  // Test case: Do not fail if the tensor is mutable but allow_mutable_tensors is true

  "tf_saved_model.global_tensor"() { is_mutable, sym_name = "v", type = tensor<f32>, value = dense<1.0> : tensor<f32> } : () -> ()

  func.func @f(%arg0: tensor<!tf_type.resource<tensor<f32>>> {tf_saved_model.bound_input = @v})
  attributes {tf_saved_model.exported_names = ["f"]} {
    // CHECK: "tf.ReadVariableOp"
    %0 = "tf.ReadVariableOp"(%arg0) {device = ""} : (tensor<!tf_type.resource<tensor<f32>>>) -> tensor<f32>
    func.return
  }

}
