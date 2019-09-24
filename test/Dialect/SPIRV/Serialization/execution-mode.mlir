// RUN: mlir-translate -test-spirv-roundtrip %s | FileCheck %s

spv.module "Logical" "GLSL450" {
  func @foo() -> () {
    spv.Return
  }
  spv.EntryPoint "GLCompute" @foo
  // CHECK: spv.ExecutionMode @foo "LocalSizeHint", 3, 4, 5
  spv.ExecutionMode @foo "LocalSizeHint", 3, 4, 5
}
