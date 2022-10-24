// RUN: xla-translate -split-input-file -mlir-hlo-to-hlo-text %s | FileCheck %s

// CHECK-LABEL: ENTRY %main.{{.*}} (Arg_0.1: f32[], Arg_1.2: f32[4]) -> f32[4,4]
func.func public @main(%arg0: tensor<f32> {mhlo.sharding = ""}, %arg1: tensor<4xf32> {mhlo.sharding = "\08\03\1A\01\02\22\02\00\01"}) -> (tensor<4x4xf32> {mhlo.sharding = "\08\03\1A\02\02\01\22\02\00\01"}) {
  // CHECK-NEXT: %Arg_1.2 = f32[4] parameter(1), sharding={devices=[2]0,1}
  // CHECK-NEXT: %Arg_0.1 = f32[] parameter(0), sharding={replicated}
  %0 = "mhlo.broadcast_in_dim"(%arg0) {broadcast_dimensions = dense<> : tensor<0xi64>} : (tensor<f32>) -> tensor<4xf32>
  %1 = mhlo.multiply %arg1, %0 : tensor<4xf32>
  %2 = "mhlo.broadcast_in_dim"(%1) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<4xf32>) -> tensor<4x4xf32>
  // CHECK: ROOT {{.*}}, sharding={devices=[2,1]0,1}
  func.return %2 : tensor<4x4xf32>
}

// -----

// CHECK-LABEL: ENTRY %main.{{.*}} ({{[^,]*}}: f32[5,8,128]) -> f32[5,8,128]
func.func @main(%arg0: tensor<5x8x128xf32> {mhlo.sharding = "\08\03\1A\03\01\02\01\22\02\00\01"}) -> (tensor<5x8x128xf32> {mhlo.sharding = "\08\03\1A\03\01\02\01\22\02\00\01"}) {
  // CHECK-NEXT: %Arg_0.1 = f32[5,8,128] parameter(0), sharding={devices=[1,2,1]0,1}
  // CHECK-NEXT: %custom-call.2 = f32[5,8,128] custom-call(f32[5,8,128] %Arg_0.1), custom_call_target="Sharding", sharding={devices=[1,2,1]0,1}
  // CHECK-NEXT: %tuple.3 = (f32[5,8,128]) tuple(f32[5,8,128] %custom-call.2)
  // CHECK-NEXT: ROOT %get-tuple-element.4 = f32[5,8,128] get-tuple-element((f32[5,8,128]) %tuple.3), index=0
  // CHECK-SAME: sharding={devices=[1,2,1]0,1}
  %0 = "mhlo.custom_call"(%arg0) {call_target_name = "Sharding",
				  mhlo.sharding = "\08\03\1A\03\01\02\01\22\02\00\01"
				 } : (tensor<5x8x128xf32>) -> tensor<5x8x128xf32>
  func.return %0 : tensor<5x8x128xf32>
}
