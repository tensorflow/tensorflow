// RUN: tf-mlir-translate -mlir-hlo-to-hlo-text %s | FileCheck %s

func @main(%arg0: tensor<2x17x31x7xi32>) -> tensor<2x3x5x7xi32> {
  %0 = xla_hlo.constant dense<-2147483648> : tensor<i32>
  %1 = "xla_hlo.reduce_window"(%arg0, %0) ( {
  ^bb0(%arg1: tensor<i32>, %arg2: tensor<i32>):	// no predecessors
    %2 = xla_hlo.max %arg1, %arg2 : tensor<i32>
    "xla_hlo.return"(%2) : (tensor<i32>) -> ()
  }) {
    window_dimensions = dense<[1, 2, 2, 1]> : tensor<4xi64>,
    window_strides = dense<[1, 4, 4, 1]> : tensor<4xi64>,
    padding = dense<[[0, 0], [2, 0], [0, 2], [0, 0]]> : tensor<4x2xi64>,
    base_dilations = dense<[1, 1, 1, 1]> : tensor<4xi64>,
    window_dilations = dense<[1, 2, 2, 1]> : tensor<4xi64>
  } : (tensor<2x17x31x7xi32>, tensor<i32>) -> tensor<2x3x5x7xi32>
  return %1 : tensor<2x3x5x7xi32>
}

// CHECK: %[[MAX_COMPUTATION:.*]] ([[ARG0:.*]]: s32[], [[ARG1:.*]]: s32[]) -> s32[]
// ROOT %[[RESULT:.*]] = s32[] maximum(s32[] %[[ARG0]], s32[] %[[ARG1]])

// CHECK: ENTRY %main
// CHECK-DAG: %[[ARG0:.*]] = s32[2,17,31,7] parameter(0)
// CHECK-DAG: %[[INIT:.*]] = s32[] constant(-2147483648)
// CHECK: ROOT %[[RESULT:.*]] = s32[2,5,8,7] reduce-window(s32[2,17,31,7] %[[ARG0]], s32[] %constant.2),
// CHECK-SAME: window={size=1x2x2x1 stride=1x4x4x1 pad=0_0x2_0x0_2x0_0 rhs_dilate=1x2x2x1},
// CHECK-SAME: to_apply=%[[MAX_COMPUTATION]]
