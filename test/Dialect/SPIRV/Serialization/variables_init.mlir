// RUN: mlir-translate -test-spirv-roundtrip %s | FileCheck %s

spv.module "Logical" "GLSL450" {
  // CHECK:         spv.globalVariable @var1 : !spv.ptr<f32, Input>
  // CHECK-NEXT:    spv.globalVariable @var2 initializer(@var1) bind(1, 0) : !spv.ptr<f32, Input>
  spv.globalVariable @var1 : !spv.ptr<f32, Input>
  spv.globalVariable @var2 initializer(@var1) bind(1, 0) : !spv.ptr<f32, Input>
}
