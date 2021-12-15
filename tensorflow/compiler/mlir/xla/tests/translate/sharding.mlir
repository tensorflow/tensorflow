// RUN: tf-mlir-translate -mlir-hlo-to-hlo-text %s | FileCheck %s

// CHECK-LABEL: ENTRY %main.{{.*}} (Arg_0.1: f32[], Arg_1.2: f32[4]) -> f32[4,4]
func public @main(%arg0: tensor<f32> {mhlo.sharding = ""}, %arg1: tensor<4xf32> {mhlo.sharding = "\08\03\1A\01\02\22\02\00\01"}) -> (tensor<4x4xf32> {mhlo.sharding = "\08\03\1A\02\02\01\22\02\00\01"}) {
  // CHECK-NEXT: %Arg_1.2 = f32[4] parameter(1), sharding={devices=[2]0,1}
  // CHECK-NEXT: %Arg_0.1 = f32[] parameter(0), sharding={replicated}
  %0 = "mhlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<4xf32>
  %1 = mhlo.multiply %arg1, %0 : tensor<4xf32>
  %2 = "mhlo.broadcast_in_dim"(%1) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<4xf32>) -> tensor<4x4xf32>
  // CHECK: ROOT {{.*}}, sharding={devices=[2,1]0,1}
  return %2 : tensor<4x4xf32>
}

