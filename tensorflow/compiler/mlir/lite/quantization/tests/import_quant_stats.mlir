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
// RUN: litert-opt %s -quant-import-stats --quant-test-stats='entries { name: "op" params { min_max { min: -1 max: 1 } } } entries { name: "op_0:0" params { min_max { min: -2 max: 2 } } }  entries { name_regex: "op_*" params { min_max { min: -3 max: 3 } } }' | FileCheck %s


// CHECK-LABEL: import_stats_skip
func.func @import_stats_skip(%arg0: tensor<4xf32>, %cst: tensor<i32>) -> (tensor<2xf32>,tensor<2xf32>) {
  %0:2 = "tfl.split"(%cst, %arg0) {num_splits = 2 : i32} : (tensor<i32>, tensor<4xf32>) -> (tensor<2xf32>, tensor<2xf32>)
    loc(fused["skip1", "skip2.cc":10:8, callsite("op" at "skip3.cc":10:8)])
  func.return %0#0, %0#1 : tensor<2xf32>, tensor<2xf32>

// CHECK-NEXT: "tfl.split"
// CHECK-NEXT: return
}

// CHECK-LABEL: import_stats_name
func.func @import_stats_name(%arg0: tensor<4xf32>, %cst: tensor<i32>) -> (tensor<2xf32>,tensor<2xf32>) {
  %0:2 = "tfl.split"(%cst, %arg0) {num_splits = 2 : i32} : (tensor<i32>, tensor<4xf32>) -> (tensor<2xf32>, tensor<2xf32>)
    loc(fused["skip1.cc":10:8, "op", callsite("skip2" at "skip3.cc":10:8)])
  func.return %0#0, %0#1 : tensor<2xf32>, tensor<2xf32>

// CHECK-NEXT: %[[split:.*]]:2 = "tfl.split"
// CHECK-NEXT: %[[stats1:.*]] = "quantfork.stats"(%[[split]]#0) <{layerStats = dense<[-1.000000e+00, 1.000000e+00]>
// CHECK-NEXT: %[[stats2:.*]] = "quantfork.stats"(%[[split]]#1) <{layerStats = dense<[-1.000000e+00, 1.000000e+00]>
// CHECK-NEXT: return %[[stats1]], %[[stats2]] : tensor<2xf32>, tensor<2xf32>
}

// CHECK-LABEL: import_stats_name_port
func.func @import_stats_name_port(%arg0: tensor<4xf32>, %cst: tensor<i32>) -> (tensor<2xf32>,tensor<2xf32>) {
  %0:2 = "tfl.split"(%cst, %arg0) {num_splits = 2 : i32} : (tensor<i32>, tensor<4xf32>) -> (tensor<2xf32>, tensor<2xf32>)
    loc(fused["skip1.cc":10:8, "op_0", callsite("skip2" at "skip3.cc":10:8)])
  func.return %0#0, %0#1 : tensor<2xf32>, tensor<2xf32>

// CHECK-NEXT: %[[split:.*]]:2 = "tfl.split"
// CHECK-NEXT: %[[stats1:.*]] = "quantfork.stats"(%[[split]]#0) <{layerStats = dense<[-2.000000e+00, 2.000000e+00]>
// CHECK-NEXT: return %[[stats1]],  %[[split]]#1 : tensor<2xf32>, tensor<2xf32>
}

// CHECK-LABEL: import_stats_name_regex
func.func @import_stats_name_regex(%arg0: tensor<4xf32>, %cst: tensor<i32>) -> (tensor<2xf32>,tensor<2xf32>) {
  %0:2 = "tfl.split"(%cst, %arg0) {num_splits = 2 : i32, name = "op_regex"} : (tensor<i32>, tensor<4xf32>) -> (tensor<2xf32>, tensor<2xf32>)
    loc(fused["skip1.cc":10:8, "op_regex", callsite("skip2" at "skip3.cc":10:8)])
  func.return %0#0, %0#1 : tensor<2xf32>, tensor<2xf32>

// CHECK-NEXT: %[[split:.*]]:2 = "tfl.split"
// CHECK-NEXT: %[[stats1:.*]] = "quantfork.stats"(%[[split]]#0) <{layerStats = dense<[-3.000000e+00, 3.000000e+00]>
// CHECK-NEXT: %[[stats2:.*]] = "quantfork.stats"(%[[split]]#1) <{layerStats = dense<[-3.000000e+00, 3.000000e+00]>
// CHECK-NEXT: return %[[stats1]], %[[stats2]] : tensor<2xf32>, tensor<2xf32>
}
