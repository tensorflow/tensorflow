// RUN: xla-translate -split-input-file -mlir-hlo-to-hlo-text %s | FileCheck %s

module @call_with_backend_config {
  func.func @main(%arg0: tensor<8x2xi32>) -> tensor<8x2xi32> {
    // CHECK:      ENTRY %main.{{[0-9]+}} ([[ARG0:Arg_0.[0-9]+]]: s32[8,2]) -> s32[8,2] {
    // CHECK-NEXT:   %[[ARG0]] = s32[8,2] parameter(0)
    // CHECK-NEXT:   s32[8,2] call(%[[ARG0]]), to_apply=%g.{{[0-9.]+}}, backend_config={"flag_configs":[],"scoped_memory_configs":[],"device_type":"DEVICE_TYPE_HOST","used_scoped_memory_configs":[]}
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
    // CHECK:      ENTRY %main.{{[0-9]+}} ([[ARG0:Arg_0.[0-9]+]]: s32[8,2]) -> s32[8,2] {
    // CHECK-NEXT:   %[[ARG0]] = s32[8,2] parameter(0)
    // CHECK-NEXT:   s32[8,2] call(%[[ARG0]]), to_apply=%g.{{[0-9.]+}}, sharding={devices=[2,2]<=[4]}
    %0 = call @g.2(%arg0) {mhlo.sharding = "{devices=[2,2]<=[4]}"} : (tensor<8x2xi32>) -> tensor<8x2xi32>
    %1 = mhlo.custom_call @MoveToHost(%0) {backend_config = ""} : (tensor<8x2xi32>) -> tensor<8x2xi32>
    return %1 : tensor<8x2xi32>
  }

  func.func private @g.2(%arg0: tensor<8x2xi32>) -> tensor<8x2xi32> {
    %0 = mhlo.multiply %arg0, %arg0 : tensor<8x2xi32>
    return %0 : tensor<8x2xi32>
  }
}

// -----

module @call_twice_no_duplication {
  func.func @main(%arg0: tensor<8x2xi32>) -> tensor<8x2xi32> {
    // CHECK: %inner{{.[0-9+]}} (Arg_0.{{[0-9+]}}: s32[8,2]) -> s32[8,2] {
    // CHECK-NOT: %inner{{.[0-9+]}} (Arg_0.{{[0-9+]}}: s32[8,2]) -> s32[8,2] {
    // CHECK: ENTRY %main{{.[0-9+]}} (Arg_0.{{[0-9+]}}: s32[8,2]) -> s32[8,2] {
    %0 = call @inner(%arg0) : (tensor<8x2xi32>) -> tensor<8x2xi32>
    %1 = call @inner(%0) : (tensor<8x2xi32>) -> tensor<8x2xi32>
    return %1 : tensor<8x2xi32>
  }

  func.func private @inner(%arg0: tensor<8x2xi32>) -> tensor<8x2xi32> {
    %0 = mhlo.multiply %arg0, %arg0 : tensor<8x2xi32>
    return %0 : tensor<8x2xi32>
  }
}


// -----

module @call_with_sharding_multiple_results {
  func.func @main(%arg0: tensor<8x2xi32>) -> tensor<8x2xi32> {
    // CHECK:               ENTRY %main.{{[0-9]+}} ([[ARG0:Arg_0.[0-9]+]]: s32[8,2]) -> s32[8,2] {
    // CHECK-NEXT:            %[[ARG0]] = s32[8,2] parameter(0)
    // CHECK-NEXT:            %[[CALL:.*]] = (s32[8,2], s32[8,2]) call(%[[ARG0]]), to_apply=%g.2.{{[0-9]}},
    // CHECK-SAME{LITERAL}:     sharding={{maximal device=0}, {replicated}}, backend_config={"flag_configs":[],"scoped_memory_configs":[],"device_type":"DEVICE_TYPE_HOST","used_scoped_memory_configs":[]}
    // CHECK-NEXT:            %[[IGNORE:.*]] = s32[8,2] get-tuple-element(%[[CALL]]), index=1, sharding={replicated}
    // CHECK-NEXT:            %[[GET_ELEMENT:.*]] = s32[8,2] get-tuple-element(%[[CALL]]), index=0, sharding={maximal device=0}
    // CHECK-NEXT:            ROOT %custom-call.{{[0-9]+}} = s32[8,2] custom-call(%[[GET_ELEMENT]]), custom_call_target="MoveToHost"
    %0:2 = call @g.2(%arg0) {mhlo.frontend_attributes = {backend_config = "{\22flag_configs\22:[],\22scoped_memory_configs\22:[],\22device_type\22:\22DEVICE_TYPE_HOST\22,\22used_scoped_memory_configs\22:[]}"}, mhlo.sharding = "{{maximal device=0}, {replicated}}"} : (tensor<8x2xi32>) -> (tensor<8x2xi32>, tensor<8x2xi32>)
    %1 = mhlo.custom_call @MoveToHost(%0#0) {backend_config = ""} : (tensor<8x2xi32>) -> tensor<8x2xi32>
    return %1 : tensor<8x2xi32>
  }

  func.func private @g.2(%arg0: tensor<8x2xi32>) -> (tensor<8x2xi32>, tensor<8x2xi32>) {
    %0 = mhlo.multiply %arg0, %arg0 : tensor<8x2xi32>
    return %0, %0 : tensor<8x2xi32>, tensor<8x2xi32>
  }
}
