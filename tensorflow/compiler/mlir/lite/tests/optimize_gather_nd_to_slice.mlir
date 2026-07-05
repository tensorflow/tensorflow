// RUN: litert-opt %s --tfl-optimize | FileCheck %s

// CHECK-LABEL: optimize_gather_nd_to_slice
func.func @optimize_gather_nd_to_slice(%arg0: tensor<10x4xf32>) -> tensor<3x4xf32> {
  // Indices [[2], [3], [4]]
  %0 = "tfl.pseudo_const"() {value = dense<[[2], [3], [4]]> : tensor<3x1xi32>} : () -> tensor<3x1xi32>
  %1 = "tfl.gather_nd"(%arg0, %0) : (tensor<10x4xf32>, tensor<3x1xi32>) -> tensor<3x4xf32>
  func.return %1 : tensor<3x4xf32>
}
// CHECK-DAG: %[[BEGIN:.*]] = arith.constant dense<[2, 0]> : tensor<2xi32>
// CHECK-DAG: %[[SIZE:.*]] = arith.constant dense<[3, 4]> : tensor<2xi32>
// CHECK: %[[RESULT:.*]] = "tfl.slice"(%arg0, %[[BEGIN]], %[[SIZE]])
// CHECK: return %[[RESULT]]

// CHECK-LABEL: optimize_gather_nd_sliding_window_f32
func.func @optimize_gather_nd_sliding_window_f32(%arg0: tensor<5x4xf32>) -> tensor<3x2x4xf32> {
  // Indices [[[0], [1]], [[1], [2]], [[2], [3]]]
  // Window size 2, stride 1, 3 windows
  %0 = "tfl.pseudo_const"() {value = dense<[[[0], [1]], [[1], [2]], [[2], [3]]]> : tensor<3x2x1xi32>} : () -> tensor<3x2x1xi32>
  %1 = "tfl.gather_nd"(%arg0, %0) : (tensor<5x4xf32>, tensor<3x2x1xi32>) -> tensor<3x2x4xf32>
  func.return %1 : tensor<3x2x4xf32>
}
// CHECK-DAG: %[[SLICE_SIZE:.*]] = arith.constant dense<[3, 4]> : tensor<2xi32>
// CHECK-DAG: %[[RESHAPE_SIZE:.*]] = arith.constant dense<[3, 1, 4]> : tensor<3xi32>
// CHECK-DAG: %[[BEGIN0:.*]] = arith.constant dense<0> : tensor<2xi32>
// CHECK-DAG: %[[BEGIN1:.*]] = arith.constant dense<[1, 0]> : tensor<2xi32>
// CHECK: %[[SLICE0:.*]] = "tfl.slice"(%arg0, %[[BEGIN0]], %[[SLICE_SIZE]])
// CHECK: %[[RESHAPE0:.*]] = "tfl.reshape"(%[[SLICE0]], %[[RESHAPE_SIZE]])
// CHECK: %[[SLICE1:.*]] = "tfl.slice"(%arg0, %[[BEGIN1]], %[[SLICE_SIZE]])
// CHECK: %[[RESHAPE1:.*]] = "tfl.reshape"(%[[SLICE1]], %[[RESHAPE_SIZE]])
// CHECK: %[[RESULT:.*]] = "tfl.concatenation"(%[[RESHAPE0]], %[[RESHAPE1]]) <{axis = 1 : i32, fused_activation_function = "NONE"}>
// CHECK: return %[[RESULT]]

// CHECK-LABEL: optimize_gather_nd_i1
func.func @optimize_gather_nd_i1(%arg0: tensor<10x4xi1>) -> tensor<3x4xi1> {
  %0 = "tfl.pseudo_const"() {value = dense<[[2], [3], [4]]> : tensor<3x1xi32>} : () -> tensor<3x1xi32>
  %1 = "tfl.gather_nd"(%arg0, %0) : (tensor<10x4xi1>, tensor<3x1xi32>) -> tensor<3x4xi1>
  func.return %1 : tensor<3x4xi1>
}
// CHECK: "tfl.slice"(%arg0, {{.*}}, {{.*}}) : (tensor<10x4xi1>, tensor<2xi32>, tensor<2xi32>) -> tensor<3x4xi1>

// CHECK-LABEL: optimize_gather_nd_sliding_window_i1
func.func @optimize_gather_nd_sliding_window_i1(%arg0: tensor<5x4xi1>) -> tensor<3x2x4xi1> {
  %0 = "tfl.pseudo_const"() {value = dense<[[[0], [1]], [[1], [2]], [[2], [3]]]> : tensor<3x2x1xi32>} : () -> tensor<3x2x1xi32>
  %1 = "tfl.gather_nd"(%arg0, %0) : (tensor<5x4xi1>, tensor<3x2x1xi32>) -> tensor<3x2x4xi1>
  func.return %1 : tensor<3x2x4xi1>
}
// CHECK: "tfl.reshape"
// CHECK: "tfl.reshape"
// CHECK: "tfl.concatenation"
