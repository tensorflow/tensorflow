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
// RUN: tf-opt %s -allow-unregistered-dialect --tf-move-tpu-compile-to-front --split-input-file | FileCheck %s

module {

// CHECK-LABEL: does_basic_reordering
func.func @does_basic_reordering() -> () {
   // CHECK: _TPUCompileMlir
   // CHECK-SAME: X
   // CHECK: _TPUCompileMlir
   // CHECK-SAME: Y
   // CHECK: OpA
   // CHECK: OpB
   // CHECK: OpC
   "tf.OpA"() : () -> ()
   %status_x, %program_x = "tf._TPUCompileMlir"() { metadata = "X", mlir_module = "..." } : () -> (tensor<!tf_type.string>, tensor<!tf_type.string>)
   "tf.OpB"() : () -> ()
   %status_y, %program_y = "tf._TPUCompileMlir"() { metadata = "Y", mlir_module = "..." } : () -> (tensor<!tf_type.string>, tensor<!tf_type.string>)
   "tf.OpC"() : () -> ()
}

// CHECK-LABEL: does_reordering_for_nested_compiles
func.func @does_reordering_for_nested_compiles() -> () {
   // CHECK: _TPUCompileMlir
   // CHECK-SAME: Z
   // CHECK: tf_device.launch
   // CHECK-NEXT: _TPUCompileMlir
   // CHECK-SAME: X
   // CHECK: tf_device.launch
   // CHECK-NEXT: _TPUCompileMlir
   // CHECK-SAME: Y
   // CHECK: OpA
   // CHECK: OpB
   // CHECK: OpC
   "tf.OpA"() : () -> ()
   "tf_device.launch"() ({
     %status_x, %program_x = "tf._TPUCompileMlir"() { metadata = "X", mlir_module = "..." } : () -> (tensor<!tf_type.string>, tensor<!tf_type.string>)
     tf_device.return
   }) {device = ""} : () -> ()
   "tf.OpB"() : () -> ()
   "tf_device.launch"() ({
     %status_y, %program_y = "tf._TPUCompileMlir"() { metadata = "Y", mlir_module = "..." } : () -> (tensor<!tf_type.string>, tensor<!tf_type.string>)
     tf_device.return
   }) {device = ""} : () -> ()
   %status_z, %program_z = "tf._TPUCompileMlir"() { metadata = "Z", mlir_module = "..." } : () -> (tensor<!tf_type.string>, tensor<!tf_type.string>)
   "tf.OpC"() : () -> ()
}
}
