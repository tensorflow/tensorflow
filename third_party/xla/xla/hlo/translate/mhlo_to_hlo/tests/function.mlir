// RUN: xla-translate -split-input-file -mlir-hlo-to-hlo-text %s | FileCheck %s

module @non_entry_function_shardings {
  func.func @main(%arg0: tensor<8x2xi32>) -> tensor<8x2xi32> {
    %0 = call @called_computation(%arg0) : (tensor<8x2xi32>) -> tensor<8x2xi32>
    return %0 : tensor<8x2xi32>
  }

  // CHECK:      %called_computation.{{[0-9]+}} (Arg_0.{{[0-9]+}}: s32[8,2]) -> s32[8,2] {
  // CHECK-NEXT:   %[[ARG:.*]] = s32[8,2] parameter(0), sharding={devices=[2,2]<=[4]}
  // CHECK-NEXT:   %[[MULT:.*]] = s32[8,2] multiply(s32[8,2] %[[ARG]], s32[8,2] %[[ARG]])
  // CHECK-NEXT:   %[[TUPLE:.*]] = (s32[8,2]) tuple(s32[8,2] %[[MULT]])
  // CHECK-NEXT:   ROOT %get-tuple-element.{{[0-9]+}} = s32[8,2] get-tuple-element((s32[8,2]) %[[TUPLE]]), index=0, sharding={devices=[2,2]<=[4]}
  func.func private @called_computation(%arg0: tensor<8x2xi32> {mhlo.sharding = "{devices=[2,2]<=[4]}"}) -> (tensor<8x2xi32> {mhlo.sharding = "{devices=[2,2]<=[4]}"}) {
    %0 = mhlo.multiply %arg0, %arg0 : tensor<8x2xi32>
    return %0 : tensor<8x2xi32>
  }
}
