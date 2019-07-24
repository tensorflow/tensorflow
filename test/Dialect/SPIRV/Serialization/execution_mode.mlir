// RUN: mlir-translate -serialize-spirv %s | mlir-translate -deserialize-spirv | FileCheck %s

func @spirv_loadstore() -> () {
  spv.module "Logical" "VulkanKHR" {
    func @foo() -> () {
      spv.Return
    }
    spv.EntryPoint "GLCompute" @foo
    // CHECK: spv.ExecutionMode @foo "LocalSizeHint", 3, 4, 5
    spv.ExecutionMode @foo "LocalSizeHint", 3, 4, 5
  }
  return
}