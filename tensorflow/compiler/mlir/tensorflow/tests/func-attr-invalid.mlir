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
// RUN: tf-opt %s -split-input-file -verify-diagnostics

// Tests invalid #tf_type.func attributes.

// expected-error@+1 {{expected '<'}}
func.func @main() attributes {tf._implements = #tf_type.func} {
  func.return
}

// -----

// expected-error@+2 {{expected attribute value}}
// expected-error@+1 {{expected symbol while parsing tf.func attribute}}
func.func @main() attributes {tf._implements = #tf_type.func<>} {
  func.return
}

// -----

// expected-error@+1 {{expected ','}}
func.func @main() attributes {tf._implements = #tf_type.func<@symbol>} {
  func.return
}

// -----

// expected-error@+1 {{expected symbol while parsing tf.func attribute}}
func.func @main() attributes {tf._implements = #tf_type.func<{}>} {
  func.return
}

// -----

// expected-error@+1 {{expected empty string or symbol while parsing tf.func attribute}}
func.func @main() attributes {tf._implements = #tf_type.func<"test", {}>} {
  func.return
}

// -----

// expected-error@+1 {{expected Dictionary attribute while parsing tf.func attribute}}
func.func @main() attributes {tf._implements = #tf_type.func<@symbol, "">} {
  func.return
}

// -----

// expected-error@+1 {{expected '>'}}
func.func @main() attributes {tf._implements = #tf_type.func<@symbol, {}, "">} {
  func.return
}
