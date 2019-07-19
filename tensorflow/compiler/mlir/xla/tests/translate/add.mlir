// RUN: tf-mlir-translate -mlir-hlo-to-hlo-text %s | FileCheck %s

// CHECK-LABEL: ENTRY %main.5 (Arg_0.1: f32[4], Arg_1.2: f32[4]) -> f32[4] {
func @main(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK-NEXT: %Arg_0.1 = f32[4] parameter(0)
  // CHECK-NEXT: %Arg_1.2 = f32[4] parameter(1)

  // CHECK-NEXT: %add.3 = f32[4] add(f32[4] %Arg_0.1, f32[4] %Arg_1.2)
  %0 = "xla.add"(%arg0, %arg1) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>

  // CHECK-NEXT: ROOT %add.4 = f32[4] add(f32[4] %add.3, f32[4] %Arg_1.2)
  %1 = "xla.add"(%0, %arg1) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %1 : tensor<4xf32>
}
