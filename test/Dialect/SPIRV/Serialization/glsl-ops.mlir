// RUN: mlir-translate -test-spirv-roundtrip %s | FileCheck %s

spv.module "Logical" "GLSL450" {
  func @fmul(%arg0 : f32, %arg1 : f32) {
    // CHECK: {{%.*}} = spv.GLSL.Exp {{%.*}} : f32
    %0 = spv.GLSL.Exp %arg0 : f32
    // CHECK: {{%.*}} = spv.GLSL.FMax {{%.*}}, {{%.*}} : f32
    %1 = spv.GLSL.FMax %arg0, %arg1 : f32
    spv.Return
  }
}