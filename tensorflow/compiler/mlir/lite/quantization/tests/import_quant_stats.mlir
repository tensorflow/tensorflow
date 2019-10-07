// RUN: tf-opt %s -quant-import-stats --quant-test-stats='entries { name: "op" params { min_max { min: -1 max: 1 } } } entries { name: "op_0:0" params { min_max { min: -2 max: 2 } } }  entries { name_regex: "op_*" params { min_max { min: -3 max: 3 } } }' | FileCheck %s --dump-input-on-failure


// CHECK-LABEL: import_stats_skip
func @import_stats_skip(%arg0: tensor<4xf32>, %cst: tensor<i32>) -> (tensor<2xf32>,tensor<2xf32>) {
  %0:2 = "tfl.split"(%cst, %arg0) {num_splits = 2 : i32, name = "skip"} : (tensor<i32>, tensor<4xf32>) -> (tensor<2xf32>, tensor<2xf32>)
  return %0#0, %0#1 : tensor<2xf32>, tensor<2xf32>

// CHECK-NEXT: "tfl.split"
// CHECK-NEXT: return
}

// CHECK-LABEL: import_stats_name
func @import_stats_name(%arg0: tensor<4xf32>, %cst: tensor<i32>) -> (tensor<2xf32>,tensor<2xf32>) {
  %0:2 = "tfl.split"(%cst, %arg0) {num_splits = 2 : i32, name = "op"} : (tensor<i32>, tensor<4xf32>) -> (tensor<2xf32>, tensor<2xf32>)
  return %0#0, %0#1 : tensor<2xf32>, tensor<2xf32>

// CHECK-NEXT: %[[split:.*]]:2 = "tfl.split"
// CHECK-NEXT: %[[stats1:.*]] = "quant.stats"(%[[split]]#0) {layerStats = dense<[-1.000000e+00, 1.000000e+00]>
// CHECK-NEXT: %[[stats2:.*]] = "quant.stats"(%[[split]]#1) {layerStats = dense<[-1.000000e+00, 1.000000e+00]>
// CHECK-NEXT: return %[[stats1]], %[[stats2]] : tensor<2xf32>, tensor<2xf32>
}

// CHECK-LABEL: import_stats_name_port
func @import_stats_name_port(%arg0: tensor<4xf32>, %cst: tensor<i32>) -> (tensor<2xf32>,tensor<2xf32>) {
  %0:2 = "tfl.split"(%cst, %arg0) {num_splits = 2 : i32, name = "op_0"} : (tensor<i32>, tensor<4xf32>) -> (tensor<2xf32>, tensor<2xf32>)
  return %0#0, %0#1 : tensor<2xf32>, tensor<2xf32>

// CHECK-NEXT: %[[split:.*]]:2 = "tfl.split"
// CHECK-NEXT: %[[stats1:.*]] = "quant.stats"(%[[split]]#0) {layerStats = dense<[-2.000000e+00, 2.000000e+00]>
// CHECK-NEXT: return %[[stats1]],  %[[split]]#1 : tensor<2xf32>, tensor<2xf32>
}

// CHECK-LABEL: import_stats_name_regex
func @import_stats_name_regex(%arg0: tensor<4xf32>, %cst: tensor<i32>) -> (tensor<2xf32>,tensor<2xf32>) {
  %0:2 = "tfl.split"(%cst, %arg0) {num_splits = 2 : i32, name = "op_regex"} : (tensor<i32>, tensor<4xf32>) -> (tensor<2xf32>, tensor<2xf32>)
  return %0#0, %0#1 : tensor<2xf32>, tensor<2xf32>

// CHECK-NEXT: %[[split:.*]]:2 = "tfl.split"
// CHECK-NEXT: %[[stats1:.*]] = "quant.stats"(%[[split]]#0) {layerStats = dense<[-3.000000e+00, 3.000000e+00]>
// CHECK-NEXT: %[[stats2:.*]] = "quant.stats"(%[[split]]#1) {layerStats = dense<[-3.000000e+00, 3.000000e+00]>
// CHECK-NEXT: return %[[stats1]], %[[stats2]] : tensor<2xf32>, tensor<2xf32>
}
