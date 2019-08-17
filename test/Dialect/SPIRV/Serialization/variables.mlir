// RUN: mlir-translate -serialize-spirv %s | mlir-translate -deserialize-spirv | FileCheck %s

// CHECK:         spv.globalVariable !spv.ptr<f32, Input> @var0 bind(1, 0)
// CHECK-NEXT:    spv.globalVariable !spv.ptr<f32, Output> @var1 bind(0, 1)
// CHECK-NEXT:    spv.globalVariable !spv.ptr<vector<3xi32>, Input> @var2 built_in("GlobalInvocationId")
// CHECK-NEXT:    spv.globalVariable !spv.ptr<vector<3xi32>, Input> @var3 built_in("GlobalInvocationId")
func @spirv_variables() -> () {
  spv.module "Logical" "VulkanKHR" {
    spv.globalVariable !spv.ptr<f32, Input> @var0 bind(1, 0)
    spv.globalVariable !spv.ptr<f32, Output> @var1 bind(0, 1)
    spv.globalVariable !spv.ptr<vector<3xi32>, Input> @var2 {built_in = "GlobalInvocationId"}
    spv.globalVariable !spv.ptr<vector<3xi32>, Input> @var3 built_in("GlobalInvocationId")
  }
  return
}