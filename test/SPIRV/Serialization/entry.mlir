// RUN: mlir-translate -serialize-spirv %s | mlir-translate -deserialize-spirv | FileCheck %s

func @spirv_loadstore() -> () {
  spv.module "Logical" "VulkanKHR" {
    func @noop() -> () {
      spv.Return
    }
    // CHECK:      spv.EntryPoint "GLCompute" @noop
    // CHECK-NEXT: spv.ExecutionMode @noop "ContractionOff"
    spv.EntryPoint "GLCompute" @noop
    spv.ExecutionMode @noop "ContractionOff"
  }
  return
}