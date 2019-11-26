// RUN: tf-mlir-translate -mlir-hlo-to-hlo-text %s | FileCheck %s

func @main(%arg0: tensor<10x24x24x64xf32>, %arg1: tensor<10x12x12x64xf32>) -> tensor<10x24x24x64xf32> {
  %0 = xla_hlo.constant dense<0.000000e+00> : tensor<f32>
  %1 = "xla_hlo.select_and_scatter"(%arg0, %arg1, %0) ( {
  ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):	// no predecessors
    %2 = "xla_hlo.compare"(%arg3, %arg4) {comparison_direction = "GE"} : (tensor<f32>, tensor<f32>) -> tensor<i1>
    "xla_hlo.return"(%2) : (tensor<i1>) -> ()
  },  {
  ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):	// no predecessors
    %2 = xla_hlo.add %arg3, %arg4 : tensor<f32>
    "xla_hlo.return"(%2) : (tensor<f32>) -> ()
  }) {
    window_dimensions = dense<[1, 2, 2, 1]> : tensor<4xi64>,
    window_strides = dense<[1, 2, 2, 1]> : tensor<4xi64>
  } : (tensor<10x24x24x64xf32>, tensor<10x12x12x64xf32>, tensor<f32>) -> tensor<10x24x24x64xf32>
  return %1 : tensor<10x24x24x64xf32>
}

// CHECK: %[[SELECT_COMPUTATION:.*]] ([[ARG0:.*]]: f32[], [[ARG1:.*]]: f32[]) -> pred[] {
// CHECK:   ROOT %[[RESULT:.*]] = pred[] compare(f32[] %[[ARG0]], f32[] %[[ARG1]]), direction=GE

// CHECK: %[[SCATTER_COMPUTATION:.*]] ([[ARG0:.*]]: f32[], [[ARG1:.*]]: f32[]) -> f32[] {
// CHECK:   ROOT %[[RESULT:.*]] = f32[] add(f32[] %[[ARG0]], f32[] %[[ARG1]])

// CHECK: ENTRY %main
// CHECK:   %[[ARG0:.*]] = f32[10,24,24,64] parameter(0)
// CHECK:   %[[ARG1:.*]] = f32[10,12,12,64] parameter(1)
// CHECK:   %[[INIT:.*]] = f32[] constant(0)

// CHECK:   ROOT %[[RESULT:.*]] = f32[10,24,24,64]
// CHECK-SAME: select-and-scatter(f32[10,24,24,64] %[[ARG0]], f32[10,12,12,64] %[[ARG1]], f32[] %[[INIT]]),
// CHECK-SAME: window={size=1x2x2x1 stride=1x2x2x1},
// CHECK-SAME: select=%[[SELECT_COMPUTATION]], scatter=%[[SCATTER_COMPUTATION]]
