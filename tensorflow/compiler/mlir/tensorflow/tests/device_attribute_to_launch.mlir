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
// RUN: tf-opt %s -split-input-file -tf-device-attribute-to-launch | FileCheck %s

// Tests that single TensorFlow op with device attribute is wrapped in `tf_device.launch` with the correct device assigned.
// CHECK-LABEL: func @single_op_launch
func.func @single_op_launch() {
  // CHECK: "tf_device.launch"
  // CHECK: device = "CPU:0"
  // CHECK: "tf.opA"
  // CHECK-NOT device
  // CHECK: tf_device.return
  "tf.opA"() {device = "CPU:0"} : () -> tensor<i1>
  func.return
}

// Tests that usage of wrapped op is replaced by launch return
// CHECK-LABEL: func @launch_return
func.func @launch_return() -> tensor<i1> {
  // CHECK: %[[LAUNCH_OUT:.*]] = "tf_device.launch"
  // CHECK: device = "CPU:0"
  // CHECK: %[[A_OUT:.*]] = "tf.opA"
  // CHECK-NOT device
  // CHECK: tf_device.return %[[A_OUT]]
  // CHECK: return %[[LAUNCH_OUT]]
  %a = "tf.opA"() {device = "CPU:0"} : () -> tensor<i1>
  func.return %a : tensor<i1>
}

// Tests that single TensorFlow op with no device attribute is not wrapped in `tf_device.launch`.
// CHECK-LABEL: func @no_device_attribute
func.func @no_device_attribute() {
  // CHECK-NOT: "tf_device.launch"
  // CHECK: "tf.opA"
  "tf.opA"() : () -> tensor<i1>
  func.return
}

// Tests that single TensorFlow op with empty device attribute is not wrapped in `tf_device.launch`.
// CHECK-LABEL: func @empty_device_attribute
func.func @empty_device_attribute() {
  // CHECK-NOT: "tf_device.launch"
  // CHECK: "tf.opA"
  "tf.opA"() {device = ""} : () -> tensor<i1>
  func.return
}

// Tests that an op not in tf dialect (tf_device.launch) with device attribute is not wrapped in `tf_device.launch`.
// Also tests that a `tf_device.launch` is not rewrapped.
// CHECK-LABEL: func @non_tf_op
func.func @non_tf_op() {
  // CHECK: "tf_device.launch"
  // CHECK-NOT "tf_device.launch"
  // CHECK: "tf.opA"
  "tf_device.launch"() ({
    "tf.opA"()  : () -> tensor<i1>
    tf_device.return
  }) {device = "CPU:0"} : () -> ()
  func.return
}
