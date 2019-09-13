// RUN: mlir-translate -serialize-spirv %s | mlir-translate -deserialize-spirv | FileCheck %s

spv.module "Logical" "GLSL450" {
  func @foo() -> () {
    spv.Return
  }
  spv.EntryPoint "GLCompute" @foo
  // CHECK: spv.ExecutionMode @foo "LocalSizeHint", 3, 4, 5
  spv.ExecutionMode @foo "LocalSizeHint", 3, 4, 5
}
