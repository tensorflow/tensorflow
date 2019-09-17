// RUN: mlir-translate -test-spirv-roundtrip %s | FileCheck %s

spv.module "Logical" "GLSL450" {
  func @noop() -> () {
    spv.Return
  }
  // CHECK:      spv.EntryPoint "GLCompute" @noop
  // CHECK-NEXT: spv.ExecutionMode @noop "ContractionOff"
  spv.EntryPoint "GLCompute" @noop
  spv.ExecutionMode @noop "ContractionOff"
}
