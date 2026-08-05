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
// RUN: tf-opt -split-input-file -verify-diagnostics %s

// Test warning on using deprecated attribute or type in old debug dump.

func.func @main() {
  // expected-error@+1 {{#tf_type.shape}}
  "tf.foo"() { shape = #tf.shape<?>} : () -> ()
  func.return
}

// -----

func.func @main() {
  // expected-error@+1 {{!tf_type.string}}
  "tf.foo"() : () -> (tensor<*x!tf.string>)
  func.return
}