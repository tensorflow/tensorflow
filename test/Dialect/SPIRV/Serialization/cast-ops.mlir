// RUN: mlir-translate -test-spirv-roundtrip %s | FileCheck %s

spv.module "Logical" "GLSL450" {
  func @fmul(%arg0 : f32) {
    // CHECK: {{%.*}} = spv.Bitcast {{%.*}} from f32 to i32
    %0 = spv.Bitcast %arg0 from f32 to i32
    spv.Return
  }
}
