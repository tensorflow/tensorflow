// RUN: xla-translate -split-input-file -mlir-hlo-to-hlo-text %s | FileCheck %s

// Shouldn't fail with an `attribute created with unregistered dialect` error.

// CHECK-LABEL: HloModule
module {
  sdy.mesh @empty_mesh = <[]>
  // TODO(b/435663161) - Allow Shardy attributes to be lowered via hlo-translate
  // CHECK: f32[1] parameter(0)
  func.func @main(%arg0 : tensor<1xf32> {sdy.sharding = #sdy.sharding<@empty_mesh, [{}]>}) -> tensor<1xf32> {
    %0 = mhlo.add %arg0, %arg0 : tensor<1xf32>
    return %0 : tensor<1xf32>
  }
}
