// RUN: xla-translate -split-input-file -mlir-hlo-to-hlo-text %s | FileCheck %s

module @call_with_backend_config {
  func.func @main(%arg0: tensor<8x2xi32>) -> tensor<8x2xi32> {
    // CHECK: ENTRY %main.7 (Arg_0.1: s32[8,2]) -> s32[8,2] {
    // CHECK-NEXT: %[[ARG0:.*]] = s32[8,2] parameter(0)
    // CHECK-NEXT: s32[8,2] call(s32[8,2] %[[ARG0]]), to_apply=%g.2.2, backend_config={"flag_configs":[],"scoped_memory_configs":[],"device_type":"DEVICE_TYPE_HOST","used_scoped_memory_configs":[]}
    %0 = call @g.2(%arg0) {mhlo.frontend_attributes = {backend_config = "{\22flag_configs\22:[],\22scoped_memory_configs\22:[],\22device_type\22:\22DEVICE_TYPE_HOST\22,\22used_scoped_memory_configs\22:[]}"}} : (tensor<8x2xi32>) -> tensor<8x2xi32>
    %1 = mhlo.custom_call @MoveToHost(%0) {backend_config = ""} : (tensor<8x2xi32>) -> tensor<8x2xi32>
    return %1 : tensor<8x2xi32>
  }

  func.func private @g.2(%arg0: tensor<8x2xi32>) -> tensor<8x2xi32> {
    %0 = mhlo.multiply %arg0, %arg0 : tensor<8x2xi32>
    return %0 : tensor<8x2xi32>
  }
}

// -----

module @call_with_sharding {
  func.func @main(%arg0: tensor<8x2xi32>) -> tensor<8x2xi32> {
    // CHECK: ENTRY %main.7 (Arg_0.1: s32[8,2]) -> s32[8,2] {
    // CHECK-NEXT: %[[ARG0:.*]] = s32[8,2] parameter(0)
    // CHECK-NEXT: s32[8,2] call(s32[8,2] %[[ARG0]]), to_apply=%g.2.2, sharding={devices=[2,2]<=[4]}
    %0 = call @g.2(%arg0) {mhlo.sharding = "{devices=[2,2]<=[4]}"} : (tensor<8x2xi32>) -> tensor<8x2xi32>
    %1 = mhlo.custom_call @MoveToHost(%0) {backend_config = ""} : (tensor<8x2xi32>) -> tensor<8x2xi32>
    return %1 : tensor<8x2xi32>
  }

  func.func private @g.2(%arg0: tensor<8x2xi32>) -> tensor<8x2xi32> {
    %0 = mhlo.multiply %arg0, %arg0 : tensor<8x2xi32>
    return %0 : tensor<8x2xi32>
  }
}
