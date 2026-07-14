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
// RUN: tf-opt -tf-saved-model-initialize-variables-in-session-init -verify-diagnostics -fail-to-fetch-local-device-manager %s

// expected-error@below{{No Local Device Manager}}
module attributes {tf_saved_model.semantics, tf_saved_model.under_construction} {

  func.func @serving_default(%arg0: tensor<!tf_type.resource<tensor<100x50xf32>>> {tf.resource_name = "dense/kernel"}) -> (tensor<100x50xf32> {tf_saved_model.index_path = ["dense_2"]})
  attributes {tf.entry_function = {control_outputs = "", inputs = "", outputs = "dense_2/Add:0"}, tf_saved_model.exported_names = ["serving_default"]} {
    %0 = "tf.VarHandleOp"() {_class = ["loc:@dense/kernel"], allowed_devices = [], container = "", device = "/job:worker/replica:0/task:1/device:CPU:0", shared_name = "var1"} : () -> tensor<!tf_type.resource<tensor<100x50xf32>>>
    %1 = "tf.ReadVariableOp"(%0) {device = ""} : (tensor<!tf_type.resource<tensor<100x50xf32>>>) -> tensor<100x50xf32>
    func.return %1 : tensor<100x50xf32>
  }
}
