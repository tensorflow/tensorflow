// RUN: tf-mlir-translate -mlir-hlo-to-hlo-text %s | FileCheck %s

func @main(%arg0 : tensor<5x2xf32>,
           %arg1 : tensor<5x5xf32>,
           %arg2 : tensor<5x7xf32>) -> tensor<5x14xf32> {
  %result = "xla_hlo.concatenate"(%arg0, %arg1, %arg2) {
    dimension = 1 : i64
  } : (tensor<5x2xf32>, tensor<5x5xf32>, tensor<5x7xf32>) -> tensor<5x14xf32>
  return %result : tensor<5x14xf32>
}


// CHECK-LABEL: main
// CHECK: %[[ARG0:.*]] = f32[5,2] parameter(0)
// CHECK: %[[ARG1:.*]] = f32[5,5] parameter(1)
// CHECK: %[[ARG2:.*]] = f32[5,7] parameter(2)
// CHECK: ROOT %[[RESULT:.*]] = f32[5,14] concatenate(f32[5,2] %[[ARG0]], f32[5,5] %[[ARG1]], f32[5,7] %[[ARG2]]), dimensions={1}
