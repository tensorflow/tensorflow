// RUN: tf-mlir-translate -mlir-hlo-to-hlo-text %s | FileCheck %s

func @main(%arg0 : tensor<1x10xf32>, %arg1 : tensor<1x10xi32>, %arg2 : tensor<f32>, %arg3 : tensor<i32>) -> (tensor<1xf32>, tensor<1xi32>) {
  %result0, %result1 = "xla_hlo.reduce"(%arg0, %arg1, %arg2, %arg3) ( {
    ^bb0(%fa: tensor<f32>, %ia : tensor<i32>, %fb: tensor<f32>, %ib: tensor<i32>):   // no predecessors
      %fmax = "xla_hlo.max"(%fa, %fb) {} : (tensor<f32>, tensor<f32>) -> tensor<f32>
      %imax = "xla_hlo.max"(%ia, %ib) {} : (tensor<i32>, tensor<i32>) -> tensor<i32>
      "xla_hlo.return"(%fmax, %imax) : (tensor<f32>, tensor<i32>) -> ()
    }) {dimensions = dense<1> : tensor<1xi64>} : (tensor<1x10xf32>, tensor<1x10xi32>, tensor<f32>, tensor<i32>) -> (tensor<1xf32>, tensor<1xi32>)
  return %result0, %result1 : tensor<1xf32>, tensor<1xi32>
}

// CHECK: %[[REGION:region_[0-9]+]]
// CHECK-SAME: ([[ARG_FA:.*]]: f32[], [[ARG_IA:.*]]: s32[], [[ARG_FB:.*]]: f32[], [[ARG_IB:.*]]: s32[]) -> (f32[], s32[])
// CHECK:  %[[FMAX:.*]] = f32[] maximum(f32[] %[[ARG_FA]], f32[] %[[ARG_FB]])
// CHECK:  %[[IMAX:.*]] = s32[] maximum(s32[] %[[ARG_IA]], s32[] %[[ARG_IB]])
// CHECK:  ROOT %[[RESULT_REGION:.*]] = (f32[], s32[]) tuple(f32[] %[[FMAX]], s32[] %[[IMAX]])

// CHECK: ENTRY %main
// CHECK-SAME: ([[ARG0:.*]]: f32[1,10], [[ARG0:.*]]: s32[1,10], [[ARG0:.*]]: f32[], [[ARG0:.*]]: s32[]) -> (f32[1], s32[1])
// CHECK: %[[RESULT:.*]] = (f32[1], s32[1]) reduce(f32[1,10] %Arg_0.1, s32[1,10] %Arg_1.2, f32[] %Arg_2.3, s32[] %Arg_3.4), dimensions={1}, to_apply=%[[REGION]]
// CHECK: %[[RESULT0:.*]] = f32[1] get-tuple-element((f32[1], s32[1]) %[[RESULT]]), index=0
// CHECK: %[[RESULT1:.*]] = s32[1] get-tuple-element((f32[1], s32[1]) %[[RESULT]]), index=1
// CHECK: ROOT %[[RESULT:.*]] = (f32[1], s32[1]) tuple(f32[1] %[[RESULT0]], s32[1] %[[RESULT1]])
