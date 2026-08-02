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
// RUN: litert-opt -loop-invariant-code-motion %s -o - | FileCheck %s

// CHECK: while_1([[ARG0:%[^ :]*]]: tensor<i32>, [[ARG1:%[^ :]*]]: tensor<1xf32>)
func.func @while_1(%arg0: tensor<i32>, %arg1: tensor<1xf32>) -> tensor<1xf32> {
  // CHECK: [[CST:%[^ ]*]] = arith.constant dense<1> : tensor<i32>
  // CHECK: "tfl.while"([[ARG0]], [[ARG1]])
  // CHECK: (tensor<i32>, tensor<1xf32>) -> (tensor<i32>, tensor<1xf32>)
  %0:2 = "tfl.while"(%arg0, %arg1) (
    // cond
    {
    ^bb0(%condArg0: tensor<*xi32>, %condArg1: tensor<*xf32>):
      %0 = "arith.constant" () {value = dense<0> : tensor<i32>} : () -> tensor<i32> loc("Const")
      %1 = "tfl.greater"(%condArg0, %0) : (tensor<*xi32>, tensor<i32>) -> tensor<i1>
      "tfl.yield"(%1) : (tensor<i1>) -> ()
    },
    // body
    {
    ^bb0(%bodyArg0: tensor<*xi32>, %bodyArg1: tensor<*xf32>):
      %0 = "arith.constant" () {value = dense<1> : tensor<i32>} : () -> tensor<i32> loc("Const")
      %1 = "tfl.sub"(%bodyArg0, %0) {fused_activation_function = "NONE"} : (tensor<*xi32>, tensor<i32>) -> tensor<*xi32>
      %2 = tfl.add %bodyArg1, %bodyArg1 {fused_activation_function = "NONE"} : tensor<*xf32>
      "tfl.yield"(%1, %2) : (tensor<*xi32>, tensor<*xf32>) -> ()
    }
  ) : (tensor<i32>, tensor<1xf32>) -> (tensor<i32>, tensor<1xf32>) loc("WhileOp")
  func.return %0#1 : tensor<1xf32>
}
