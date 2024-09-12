// RUN: xla-translate -split-input-file -mlir-hlo-to-hlo-text %s | FileCheck %s

// CHECK: HloModule main, entry_computation_layout={(s64[<=4,1]{1,0})->s64[1,<=4]{1,0}}
func.func @main(%arg0: tensor<?x1xi64, #mhlo.type_extensions<bounds = [4, ?]>>) -> tensor<1x?xi64, #mhlo.type_extensions<bounds = [?, 4]>> {
  %0 = mhlo.constant dense<1> : tensor<1xi32>
  %1 = "mhlo.get_dimension_size"(%arg0) <{dimension = 0 : i64}> : (tensor<?x1xi64, #mhlo.type_extensions<bounds = [4, ?]>>) -> tensor<i32>
  %2 = mhlo.reshape %1 : (tensor<i32>) -> tensor<1xi32>
  %3 = "mhlo.concatenate"(%0, %2) <{dimension = 0 : i64}> : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
  %4 = mhlo.dynamic_reshape %arg0, %3 : (tensor<?x1xi64, #mhlo.type_extensions<bounds = [4, ?]>>, tensor<2xi32>) -> tensor<1x?xi64, #mhlo.type_extensions<bounds = [?, 4]>>
  func.return %4 : tensor<1x?xi64, #mhlo.type_extensions<bounds = [?, 4]>>
  //      CHECK: %[[ARG0:.*]] = s64[<=4,1] parameter(0)
  // CHECK-NEXT: %[[SIZE0x1:.*]] = s32[1] constant({1})
  // CHECK-NEXT: %[[SIZE1:.*]] = s32[] get-dimension-size(s64[<=4,1] %[[ARG0]]), dimensions={0}
  // CHECK-NEXT: %[[SIZE1x1:.*]] = s32[1] reshape(s32[] %[[SIZE1]])
  // CHECK-NEXT: %[[SHAPE:.*]] = s32[2] concatenate(s32[1] %[[SIZE0x1]], s32[1] %[[SIZE1x1]]), dimensions={0}
  // CHECK-NEXT: %[[SHAPE0x1:.*]] = s32[1] slice(s32[2] %[[SHAPE]]), slice={[0:1]}
  // CHECK-NEXT: %[[SHAPE0:.*]] = s32[] reshape(s32[1] %[[SHAPE0x1]])
  // CHECK-NEXT: %[[SHAPE1x1:.*]] = s32[1] slice(s32[2] %[[SHAPE]]), slice={[1:2]}
  // CHECK-NEXT: %[[SHAPE1:.*]] = s32[] reshape(s32[1] %[[SHAPE1x1]])
  // CHECK-NEXT: ROOT %dynamic-reshape.10 = s64[1,<=4] dynamic-reshape(s64[<=4,1] %[[ARG0]], s32[] %[[SHAPE0]], s32[] %[[SHAPE1]])
}
