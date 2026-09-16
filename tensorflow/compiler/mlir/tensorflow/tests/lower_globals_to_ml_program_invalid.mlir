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
// RUN: tf-opt %s --allow-unregistered-dialect --tf-saved-model-lower-globals-to-mlprogram --split-input-file --verify-diagnostics

module attributes {tf_saved_model.semantics} {
  "tf_saved_model.global_tensor"() { is_mutable, sym_name = "v", type = tensor<?xf32>, value = dense<1.> : tensor<1xf32> } : () -> ()
  "tf_saved_model.global_tensor"() { is_mutable, sym_name = "v1", type = tensor<?xf32>, value = dense<1.> : tensor<1xf32> } : () -> ()
  // expected-error@+1 {{Incompatible code paths}}
  func.func @f(%arg0: tensor<?xf32> {tf_saved_model.index_path = [0]}, %v: tensor<!tf_type.resource<tensor<?xf32>>> {tf_saved_model.bound_input = @v}, %v1: tensor<!tf_type.resource<tensor<?xf32>>> {tf_saved_model.bound_input = @v1}) -> (tensor<?xf32> {tf_saved_model.index_path = [0]}) attributes {tf_saved_model.exported_names = ["f"]} {
    %pred = arith.constant false
    cf.cond_br %pred, ^bb1(%v : tensor<!tf_type.resource<tensor<?xf32>>>), ^bb1(%v1 : tensor<!tf_type.resource<tensor<?xf32>>>)
  ^bb1(%either: tensor<!tf_type.resource<tensor<?xf32>>>):
    %ret = "tf.ReadVariableOp"(%either) : (tensor<!tf_type.resource<tensor<?xf32>>>) -> tensor<?xf32>
    return %ret : tensor<?xf32>
  }
}

// -----

module attributes {tf_saved_model.semantics} {
  // expected-error@+1 {{no predecessor}}
  func.func @f() -> () attributes {tf_saved_model.exported_names = ["f"]} {
  ^entry():
    return
  ^deadcode(%0: tensor<!tf_type.resource<tensor<?xf32>>>):
    %ret = "tf.ReadVariableOp"(%0) : (tensor<!tf_type.resource<tensor<?xf32>>>) -> tensor<?xf32>
    return
  }
}

// -----

module attributes {tf_saved_model.semantics} {
  // expected-error@+1 {{no predecessor}}
  func.func @f() -> () attributes {tf_saved_model.exported_names = ["f"]} {
  ^entry():
    return
  ^infinite(%0: tensor<!tf_type.resource<tensor<?xf32>>>):
    %ret = "tf.ReadVariableOp"(%0) : (tensor<!tf_type.resource<tensor<?xf32>>>) -> tensor<?xf32>
    cf.br ^infinite(%0: tensor<!tf_type.resource<tensor<?xf32>>>)
  }
}

// -----

module attributes {tf_saved_model.semantics} {
  func.func @f() -> () attributes {tf_saved_model.exported_names = ["f"]} {
    // expected-error@+1 {{Non constant predecessor}}
    %0 = "nonsense.op"() : () -> tensor<!tf_type.resource<tensor<?xf32>>>
    %1 = "tf.ReadVariableOp"(%0) : (tensor<!tf_type.resource<tensor<?xf32>>>) -> tensor<?xf32>
    return
  }
}
