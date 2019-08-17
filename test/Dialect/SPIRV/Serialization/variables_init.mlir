// RUN: mlir-translate -serialize-spirv %s | mlir-translate -deserialize-spirv | FileCheck %s

func @spirv_variables() -> () {
  spv.module "Logical" "VulkanKHR" {
    // CHECK:         spv.globalVariable !spv.ptr<f32, Input> @var1
    // CHECK-NEXT:    spv.globalVariable !spv.ptr<f32, Input> @var2 initializer(@var1) bind(1, 0)
    spv.globalVariable !spv.ptr<f32, Input> @var1
    spv.globalVariable !spv.ptr<f32, Input> @var2 initializer(@var1) bind(1, 0)
  }
  return
}