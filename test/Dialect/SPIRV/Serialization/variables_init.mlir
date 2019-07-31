// RUN: mlir-translate -serialize-spirv %s | mlir-translate -deserialize-spirv | FileCheck %s

func @spirv_variables() -> () {
  spv.module "Logical" "VulkanKHR" {
    // CHECK: [[INIT:%.*]] = spv.constant 4.000000e+00 : f32
    // CHECK: {{%.*}} = spv.Variable init([[INIT]]) bind(1, 0) : !spv.ptr<f32, Input>
    %0 = spv.constant 4.0 : f32
    %2 = spv.Variable init(%0) bind(1, 0) : !spv.ptr<f32, Input>
  }
  return
}