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
// RUN: tfg-opt-no-passes -split-input-file -canonicalize %s | FileCheck %s

// Test CastOp (or any other op) with no uses is not removed in a tfg.graph by
// either the folder or by being marked as trivially dead.
//
// Any op in the graph might be a feed, a fetch, or an op that was explicitly
// marked as "keep".
// TODO(jeffniu): This information should be imported from GraphDef/Grappler
// by adding a side-effectful `tfg.sink` op that depends on these ops.
// CHECK-LABEL: tfg.graph
tfg.graph #tf_type.version<producer = 42, min_consumer = 24> {
  // CHECK: %[[CONST:.*]], %{{.*}} = Const
  %Const, %ctl = Const name("const") {dtype = i32, value = dense<0> : tensor<i32>} : () -> (tensor<i32>)
  // CHECK: Cast(%[[CONST]])
  %Cast, %ctl_0 = Cast(%Const) name("cast") {SrcT = i32, DstT = f32} : (tensor<i32>) -> (tensor<f32>)
}

// -----

// Test CastOp with no uses is removed in a tfg.func by the pruner.
// CHECK-LABEL: tfg.func @test_cast_no_uses
tfg.func @test_cast_no_uses(%a: tensor<i32> {tfg.name = "a"})
   -> (tensor<i32> {tfg.name = "b"}) {
  // CHECK-NOT: Cast
  %Cast, %ctl = Cast(%a) name("cast") {SrcT = i32, DstT = f32} : (tensor<i32>) -> (tensor<f32>)
  // CHECK: return
  return(%a) : tensor<i32>
}

// -----

// Test CastOp between the same type is not folded.
//
// Some passes (e.g. quantization) will, in the process of lowering one
// high-level op to many low-level ops, propagate the metadata not to all the
// generated ops but to a "dummy" CastOp between the same type. Folding these
// away will cause errors due to the erased metadata.
// CHECK-LABEL: @test_cast_same_type(%a: tensor<i32> {tfg.name = "a"}
tfg.func @test_cast_same_type(%a: tensor<i32> {tfg.name = "a"})
   -> (tensor<i32> {tfg.name = "b"}) {
  // CHECK: Cast
  %Cast, %ctl = Cast(%a) name("cast") {SrcT = i32, DstT = i32, _some_metadata = "foo"} : (tensor<i32>) -> (tensor<i32>)
  return(%Cast) : tensor<i32>
}
