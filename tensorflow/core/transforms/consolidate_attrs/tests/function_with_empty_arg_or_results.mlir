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
// RUN: tfg-transforms-opt %s --tfg-consolidate-attrs | FileCheck %s
// RUN: tfg-transforms-opt %s --tfg-prepare-attrs-export | FileCheck %s

// This is used to ensure that the pass is handling empty arg_attrs/res_attrs,
// e.g., no crash happens.

// CHECK-LABEL: @test_no_arg
tfg.func @test_no_arg() -> (tensor<*xi32>) {
  %A, %ctl = A() : () -> (tensor<*xi32>)
  return(%A) : tensor<*xi32>
}

// CHECK-LABEL: @test_without_result
tfg.func @test_without_result(%arg0: tensor<*xi32>) -> () {
  return
}

// CHECK-LABEL: @test_without_arg_nor_result
tfg.func @test_without_arg_nor_result() -> () {
  return
}
