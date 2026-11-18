// Copyright 2026 The OpenXLA Authors. All Rights Reserved.
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

// Each test file is targeted to a specific version that is latest as of date.
//
// Smoke test:
// RUN: sdy_opt %s.bc | FileCheck %s
// RUN: sdy_translate --deserialize %s.bc | sdy_translate --target-version=0.0.0 --serialize | sdy_opt | FileCheck %s
//
// Backward compatibility test:
// RUN: sdy_translate --deserialize %s.bc | sdy_opt > %t.0
// RUN: sdy_opt --strip-debuginfo %s > %t.1
// RUN: cmp -s %t.0 %t.1
//
// Forward compatibility test:
// RUN: sdy_translate --target-version=0.0.0 --serialize --strip-debuginfo %s > %t.2
// RUN: cmp -s %s.bc %t.2
//
// Verify all attributes/types are implemented during bytecode Write.
// RUN: sdy_opt -emit-bytecode %if asserts %{-debug-only=sdy-bytecode%} %s 2>&1 | FileCheck --check-prefix=CHECK-WARN %s
// RUN: sdy_opt -emit-bytecode %s | sdy_opt %if asserts %{-debug-only=sdy-bytecode%} 2>&1 | FileCheck --check-prefix=CHECK-WARN %s
// CHECK-WARN-NOT: Not Implemented

// CHECK: module @module attributes
// CHECK-SAME: {mhlo.frontend_attributes = {xla.sdy.meshes = "{mesh_xyz = #sdy.mesh<[\\\22x\\\22=4, \\\22y\\\22=2, \\\22z\\\22=8]>}"}}
module @module attributes {mhlo.frontend_attributes = {xla.sdy.meshes =
    "{mesh_xyz = #sdy.mesh<[\\\22x\\\22=4, \\\22y\\\22=2, \\\22z\\\22=8]>}"}} {

  sdy.mesh @mesh_xy = #sdy.mesh<["x"=2, "y"=2]>

  // CHECK-LABEL: func @sharding
  // CHECK-SAME: %arg0: tensor<32xi32> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh_xyz, [{\\\22y\\\22}p2]>"}}
  // CHECK-SAME: %arg1: tensor<32xi32> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh_xyz, [{\\\22x\\\22}p1]>"}}
  // CHECK-SAME: %arg2: tensor<32xi32> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh_xyz, [{\\\22z\\\22}p0]>"}}
  func.func @sharding(
    %arg0: tensor<32xi32> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh_xyz, [{\\\22y\\\22}p2]>"}},
    %arg1: tensor<32xi32> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh_xyz, [{\\\22x\\\22}p1]>"}},
    %arg2: tensor<32xi32> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh_xyz, [{\\\22z\\\22}p0]>"}}
  ) -> (tensor<32xi32>, tensor<32xi32>, tensor<32xi32>) {
    return %arg0, %arg1, %arg2 : tensor<32xi32>, tensor<32xi32>, tensor<32xi32>
  }

  // CHECK-LABEL: func @sharding_rule
  func.func @sharding_rule(%arg0: tensor<8x2xf32>, %arg1: tensor<16xf32>) -> tensor<8x2xf64> {
    // CHECK: %{{.*}} = stablehlo.custom_call @foo(%arg0, %arg1)
    // CHECK-SAME: {mhlo.frontend_attributes = {xla.sdy.sharding_rule = "#sdy.op_sharding_rule<([i, j], [ij])->([i, j]) {i=8, j=2}>"}}
    %0 = stablehlo.custom_call @foo(%arg0, %arg1)
      {mhlo.frontend_attributes = {xla.sdy.sharding_rule = "#sdy.op_sharding_rule<([i, j], [ij])->([i, j]) {i=8, j=2}>"}}
      : (tensor<8x2xf32>, tensor<16xf32>) -> tensor<8x2xf64>
    return %0 : tensor<8x2xf64>
  }

  // CHECK-LABEL: func @func_results_with_sharding
  func.func @func_results_with_sharding(%arg0: tensor<8x8xi32>, %arg1: tensor<64xi32>) -> (tensor<8x8xi32>, tensor<64xi32>) {
    // CHECK: %{{.*}} = stablehlo.custom_call @xla.sdy.FuncResultSharding(%arg0)
    // CHECK-SAME: {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding_per_value<[<@mesh_xyz, [{\\\22x\\\22}p0, {\\\22y\\\22}p1]>]>"}}
    %0 = stablehlo.custom_call @xla.sdy.FuncResultSharding(%arg0)
      {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding_per_value<[<@mesh_xyz, [{\\\22x\\\22}p0, {\\\22y\\\22}p1]>]>"}}
       : (tensor<8x8xi32>) -> tensor<8x8xi32>
    // CHECK: %{{.*}} = stablehlo.custom_call @xla.sdy.FuncResultSharding(%arg1)
    // CHECK-SAME: {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding_per_value<[<@mesh_xyz, [{\\\22z\\\22, ?}p2]>]>"}}
    %1 = stablehlo.custom_call @xla.sdy.FuncResultSharding(%arg1)
      {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding_per_value<[<@mesh_xyz, [{\\\22z\\\22, ?}p2]>]>"}}
      : (tensor<64xi32>) -> tensor<64xi32>
    return %0, %1 : tensor<8x8xi32>, tensor<64xi32>
  }

  // CHECK-LABEL: func @sharding_group
  func.func @sharding_group(%arg0: tensor<16x8xf32>) -> tensor<16x8xf32> {
  // CHECK: stablehlo.custom_call @xla.sdy.ShardingGroup(%arg0)
  stablehlo.custom_call @xla.sdy.ShardingGroup(%arg0)
    {has_side_effect = true, mhlo.frontend_attributes = {xla.sdy.sharding_group_id = "102 : i64"}} : (tensor<16x8xf32>) -> ()
  return %arg0 : tensor<16x8xf32>
  }

  // CHECK-LABEL: func @sharding_barrier
  func.func @sharding_barrier(%arg0: tensor<16x8xf32>) -> tensor<16x8xf32> {
  // CHECK: stablehlo.custom_call @xla.sdy.PropagationBarrier(%arg0)
  stablehlo.custom_call @xla.sdy.PropagationBarrier(%arg0)
    {mhlo.frontend_attributes = {xla.sdy.allowed_direction = "2 : i32"}} : (tensor<16x8xf32>) -> ()
  return %arg0 : tensor<16x8xf32>
  }

  // CHECK-LABEL: func.func @manual_computation_op
  func.func @manual_computation_op(%arg0: tensor<16x32xf32>) -> (tensor<128x32xf32>, tensor<128x32xf32>) {
    // CHECK:      %[[LOCAL:.*]] = stablehlo.custom_call @xla.sdy.GlobalToLocalShape(%arg0) {
    // CHECK-SAME:   mhlo.frontend_attributes = {
    // CHECK-SAME:     xla.sdy.in_shardings = "#sdy.sharding_per_value<[<@mesh_xyz, [{}, {}], replicated={\\\22x\\\22, \\\22y\\\22}>]>",
    // CHECK-SAME:     xla.sdy.manual_axes = "#sdy<manual_axes{\\\22x\\\22, \\\22y\\\22}>"}
    // CHECK-SAME: } : (tensor<16x32xf32>) -> tensor<16x32xf32>
    %0 = stablehlo.custom_call @xla.sdy.GlobalToLocalShape(%arg0) {
      mhlo.frontend_attributes = {
        xla.sdy.in_shardings = "#sdy.sharding_per_value<[<@mesh_xyz, [{}, {}], replicated={\\\22x\\\22, \\\22y\\\22}>]>",
        xla.sdy.manual_axes = "#sdy<manual_axes{\\\22x\\\22, \\\22y\\\22}>"}
    } : (tensor<16x32xf32>) -> tensor<16x32xf32>
    // CHECK: %[[OUT:.*]]:2 = call @xla.sdy.manual_computation_body(%[[LOCAL]]) : (tensor<16x32xf32>) -> (tensor<16x32xf32>, tensor<16x32xf32>)
    %1:2 = call @xla.sdy.manual_computation_body(%0) : (tensor<16x32xf32>) -> (tensor<16x32xf32>, tensor<16x32xf32>)
    // CHECK:      %[[GLOBAL:.*]]:2 = stablehlo.custom_call @xla.sdy.LocalToGlobalShape(%[[OUT]]#0, %[[OUT]]#1) {
    // CHECK-SAME:   mhlo.frontend_attributes = {
    // CHECK-SAME:     xla.sdy.in_shardings = "#sdy.sharding_per_value<[<@mesh_xyz, [{}, {}], replicated={\\\22x\\\22, \\\22y\\\22}>]>",
    // CHECK-SAME:     xla.sdy.manual_axes = "#sdy<manual_axes{\\\22x\\\22, \\\22y\\\22}>"}
    // CHECK-SAME: } : (tensor<16x32xf32>, tensor<16x32xf32>) -> (tensor<128x32xf32>, tensor<128x32xf32>)
    %2:2 = stablehlo.custom_call @xla.sdy.LocalToGlobalShape(%1#0, %1#1) {
      mhlo.frontend_attributes = {
        xla.sdy.in_shardings = "#sdy.sharding_per_value<[<@mesh_xyz, [{}, {}], replicated={\\\22x\\\22, \\\22y\\\22}>]>",
        xla.sdy.manual_axes = "#sdy<manual_axes{\\\22x\\\22, \\\22y\\\22}>"}
    } : (tensor<16x32xf32>, tensor<16x32xf32>) -> (tensor<128x32xf32>, tensor<128x32xf32>)
    return %2#0, %2#1 : tensor<128x32xf32>, tensor<128x32xf32>
  }

  // CHECK-LABEL: func.func private @xla.sdy.manual_computation_body
  func.func private @xla.sdy.manual_computation_body(%arg0: tensor<16x32xf32>) -> (tensor<16x32xf32>, tensor<16x32xf32>) {
    return %arg0, %arg0 : tensor<16x32xf32>, tensor<16x32xf32>
  }

  // CHECK-LABEL: func @call_no_out_shardings
  func.func @call_no_out_shardings(%arg0: tensor<8x2xi32>) -> (tensor<8x2xi32>) {
    // CHECK: %{{.*}} = call @foo(%arg0)
    // CHECK-SAME: mhlo.frontend_attributes = {backend_config = "{\22flag_configs\22:[],\22scoped_memory_configs\22:[],\22device_type\22:\22DEVICE_TYPE_HOST\22,\22used_scoped_memory_configs\22:[]}"}
    // CHECK-SAME: random_attr = "random_value"
    %0 = call @foo(%arg0) {random_attr = "random_value",
      mhlo.frontend_attributes = {backend_config = "{\22flag_configs\22:[],\22scoped_memory_configs\22:[],\22device_type\22:\22DEVICE_TYPE_HOST\22,\22used_scoped_memory_configs\22:[]}"}}
      : (tensor<8x2xi32>) -> tensor<8x2xi32>
    %1 = stablehlo.custom_call @MoveToHost(%0) {backend_config = ""} : (tensor<8x2xi32>) -> tensor<8x2xi32>
    return %1 : tensor<8x2xi32>
  }

  // CHECK-LABEL: func.func private @foo
  func.func private @foo(%arg0: tensor<8x2xi32>) -> tensor<8x2xi32> {
    %0 = stablehlo.multiply %arg0, %arg0 {mhlo.frontend_attributes = {_xla_compute_type = "host"}} : tensor<8x2xi32>
    return %0 : tensor<8x2xi32>
  }

  // CHECK-LABEL: func @call_out_shardings
  func.func @call_out_shardings(%arg0: tensor<8x2xi32>) -> (tensor<8x2xi32>) {
    // CHECK: %{{.*}} = call @bar(%arg0)
    // CHECK-SAME: mhlo.frontend_attributes = {backend_config = "{\22flag_configs\22:[],\22scoped_memory_configs\22:[],\22device_type\22:\22DEVICE_TYPE_HOST\22,\22used_scoped_memory_configs\22:[]}"}
    // CHECK-SAME: random_attr = "random_value", sdy.sharding = #sdy.sharding_per_value<[<@mesh_xy, [{"x"}, {"y"}]>]>
    %0 = call @bar(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh_xy, [{"x"}, {"y"}]>]>, random_attr = "random_value",
      mhlo.frontend_attributes = {backend_config = "{\22flag_configs\22:[],\22scoped_memory_configs\22:[],\22device_type\22:\22DEVICE_TYPE_HOST\22,\22used_scoped_memory_configs\22:[]}"}}
      : (tensor<8x2xi32>) -> tensor<8x2xi32>
    %1 = stablehlo.custom_call @MoveToHost(%0) {backend_config = ""} : (tensor<8x2xi32>) -> tensor<8x2xi32>
    return %1 : tensor<8x2xi32>
  }

  // CHECK-LABEL: func.func private @bar
  func.func private @bar(%arg0: tensor<8x2xi32>) -> (tensor<8x2xi32> {sdy.sharding = #sdy.sharding<@mesh_xy, [{}, {"y"}]>}) {
    %0 = stablehlo.multiply %arg0, %arg0 {mhlo.frontend_attributes = {_xla_compute_type = "host"}} : tensor<8x2xi32>
    return %0 : tensor<8x2xi32>
  }

}
