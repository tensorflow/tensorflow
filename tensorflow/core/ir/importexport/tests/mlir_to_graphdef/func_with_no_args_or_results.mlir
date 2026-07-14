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
// RUN: tfg-translate -mlir-to-graphdef --split-input-file %s | FileCheck %s

// CHECK: signature {
// CHECK-NEXT: name: "func_no_args_no_results"
// CHECK-NOT: input_arg
// CHECK-NOT: output_arg
tfg.func @func_no_args_no_results() -> () {
  return
}

// -----

// CHECK: signature {
// CHECK-NEXT: name: "func_no_args"
// CHECK-NEXT: output_arg
// CHECK-NOT: input_arg
tfg.func @func_no_args() -> (tensor<1xi32> {tfg.name = "ret1"}) {
  %Const, %ctl_2 = Const name("c") {dtype = i32, value = dense<0> : tensor<1xi32>} : () -> (tensor<1xi32>)
  return (%Const) : tensor<1xi32>
}

// -----

// CHECK: signature {
// CHECK-NEXT: name: "func_no_results"
// CHECK-NEXT: input_arg
// CHECK-NOT: output_arg
tfg.func @func_no_results(%arg : tensor<i32> {tfg.name = "arg"}) -> () {
  return
}
