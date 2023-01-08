// RUN: xla-translate -mlir-hlo-to-hlo-text %s | FileCheck %s
// RUN: xla-translate -mlir-hlo-to-hlo-text -emit-use-tuple-args %s | FileCheck %s --check-prefix=TUPLE-ARG
// RUN: xla-translate -mlir-hlo-to-hlo-text -emit-return-tuple %s | FileCheck %s --check-prefix=TUPLE-RET
// RUN: xla-translate -mlir-hlo-to-hlo-text -emit-use-tuple-args -emit-return-tuple %s | FileCheck %s --check-prefix=TUPLES

// CHECK-LABEL: ENTRY %main.{{.*}} (Arg_0.1: f32[4], Arg_1.2: f32[4]) -> f32[4]
// TUPLE-ARG-LABEL: ENTRY %main.{{.*}} (arg_tuple.1: (f32[4], f32[4])) -> f32[4]
// TUPLE-RET-LABEL: ENTRY %main.{{.*}} (Arg_0.1: f32[4], Arg_1.2: f32[4]) -> (f32[4])
// TUPLES-LABEL: ENTRY %main.{{.*}} (arg_tuple.1: (f32[4], f32[4])) -> (f32[4])
func.func @main(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK-NEXT: %Arg_0.1 = f32[4] parameter(0)
  // CHECK-NEXT: %Arg_1.2 = f32[4] parameter(1)

  // CHECK-NEXT: %add.3 = f32[4] add(f32[4] %Arg_0.1, f32[4] %Arg_1.2)
  %0 = "mhlo.add"(%arg0, %arg1) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>

  // CHECK-NEXT: ROOT %add.4 = f32[4] add(f32[4] %add.3, f32[4] %Arg_1.2)
  %1 = "mhlo.add"(%0, %arg1) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  func.return %1 : tensor<4xf32>
}
