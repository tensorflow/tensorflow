// RUN: mlir-translate -test-spirv-roundtrip %s | FileCheck %s

spv.module "Logical" "GLSL450" {
  func @fmul(%arg0 : f32) {
    // CHECK: {{%.*}} = spv.GLSL.Exp {{%.*}} : f32
    %0 = spv.GLSL.Exp %arg0 : f32
    spv.Return
  }
}