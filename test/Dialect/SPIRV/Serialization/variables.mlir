// RUN: mlir-translate -serialize-spirv %s | mlir-translate -deserialize-spirv | FileCheck %s

// CHECK:           {{%.*}} = spv.Variable bind(1, 0) : !spv.ptr<f32, Input>
// CHECK-NEXT:      {{%.*}} = spv.Variable bind(0, 1) : !spv.ptr<f32, Output>
func @spirv_variables() -> () {
  spv.module "Logical" "VulkanKHR" {
    %2 = spv.Variable bind(1, 0) : !spv.ptr<f32, Input>
    %3 = spv.Variable bind(0, 1): !spv.ptr<f32, Output>
  }
  return
}