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
// RUN: tf-opt %s --tf-name-anonymous-iterators --split-input-file | FileCheck %s

// CHECK-LABEL: gives_a_name_to_anonymous_iterators
func.func private @gives_a_name_to_anonymous_iterators() {
  // CHECK: "tf.Iterator"
  // CHECK-SAME: output_shapes{{.*}}200x28x28x1{{.*}}200x10
  // CHECK-SAME: output_types = [f32, f32]
  // CHECK-SAME: shared_name = "_iterator1"
  %0 = "tf.AnonymousIteratorV3"() {output_shapes = [
    #tf_type.shape<200x28x28x1>,
    #tf_type.shape<200x10>], output_types = [f32, f32]} : () -> tensor<!tf_type.resource>
  // CHECK: "tf.Iterator"
  // CHECK-SAME: shared_name = "_iterator2"
  %1 = "tf.AnonymousIteratorV3"() {output_shapes = [
    #tf_type.shape<200x10>], output_types = [f32]} : () -> tensor<!tf_type.resource>
  return
}

// -----

// CHECK-LABEL: handles_all_versions
func.func private @handles_all_versions() {
  // CHECK: "tf.Iterator"
  // CHECK-SAME: 1x42
  %0 = "tf.AnonymousIterator"() {output_shapes = [
    #tf_type.shape<1x42>], output_types = [f32]} : () -> tensor<!tf_type.resource>
  // CHECK: "tf.Iterator"
  // CHECK-SAME: 2x42
  %1, %2 = "tf.AnonymousIteratorV2"() {output_shapes = [
    #tf_type.shape<2x42>], output_types = [f32]} : () -> (tensor<!tf_type.resource>, tensor<!tf_type.variant>)
  // CHECK: "tf.Iterator"
  // CHECK-SAME: 3x42
  %3 = "tf.AnonymousIteratorV3"() {output_shapes = [
    #tf_type.shape<3x42>], output_types = [f32]} : () -> tensor<!tf_type.resource>
  return
}
