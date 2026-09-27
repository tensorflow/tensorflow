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
// RUN: tfg-transforms-opt --tfg-region-to-functional %s | FileCheck %s
// RUN: tfg-transforms-opt --tfg-region-to-functional='force-control-capture=true' %s | FileCheck %s --check-prefix=CAPTURE

// Check that chain constants are inserted to force control capture (when
// enabled).

// CAPTURE: @test(%[[INDEX:.*]]: tensor<i32>
tfg.func @test(%index: tensor<i32> {tfg.name = "index"}) -> (tensor<i32>) {
  %ctlA = NoOp [%index.ctl]
  %ctlB = NoOp
  // CHECK: CaseRegion
  // CAPTURE: %[[UNUSED_0:.*]], %{{.*}} = Const [%{{.*}}] device("/foo") assigned_device("/bar") name("Case_mlir_const_capture_0")
  // CAPTURE-SAME: _tpu_replicate = "cluster"
  // CAPTURE: %[[UNUSED_1:.*]], %{{.*}} = Const [%{{.*}}] device("/foo") assigned_device("/bar") name("Case_mlir_const_capture_1")
  // CAPTURE-SAME: _tpu_replicate = "cluster"
  // CAPTURE: Case(%[[INDEX]], %[[UNUSED_0]], %[[UNUSED_1]]) device("/foo") assigned_device("/bar") name("Case")
  // CAPTURE-SAME: _tpu_replicate = "cluster"
  %Case, %ctl_0 = CaseRegion %index {
    %A, %ctl_1 = A [%ctlA, %ctlB] : () -> (tensor<i32>)
    yield(%A) : tensor<i32>
  } {_tpu_replicate = "cluster", _mlir_name = "Case",
     _mlir_device = "/foo", _mlir_assigned_device = "/bar"}
  : (tensor<i32>) -> (tensor<i32>)
  return(%Case) : tensor<i32>
}
