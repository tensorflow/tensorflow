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
// RUN: tfg-transforms-opt --tfg-functional-to-region %s | FileCheck %s

tfg.func @then(%arg: tensor<*xi32>) -> (tensor<*xi32>) {
  return(%arg) : tensor<*xi32>
}

// CHECK-LABEL: tfg.func @test_missing_function
tfg.func @test_missing_function(%cond: tensor<*xi1>, %arg: tensor<*xi32>) -> (tensor<*xi32>) {
  // CHECK: If(
  %If, %ctlIf = If(%cond, %arg) {
    then_branch = #tf_type.func<@then, {}>,
    else_branch = #tf_type.func<@else, {}>
  } : (tensor<*xi1>, tensor<*xi32>) -> (tensor<*xi32>)
  return(%If) : tensor<*xi32>
}
