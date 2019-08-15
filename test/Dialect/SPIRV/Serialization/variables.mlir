// RUN: mlir-translate -serialize-spirv %s | mlir-translate -deserialize-spirv | FileCheck %s

// CHECK:           {{%.*}} = spv.Variable bind(1, 0) : !spv.ptr<f32, Input>
// CHECK-NEXT:      {{%.*}} = spv.Variable bind(0, 1) : !spv.ptr<f32, Output>
// CHECK-NEXT:      {{%.*}} = spv.Variable built_in("GlobalInvocationId") : !spv.ptr<vector<3xi32>, Input>
// CHECK-NEXT:      {{%.*}} = spv.Variable built_in("GlobalInvocationId") : !spv.ptr<vector<3xi32>, Input>
func @spirv_variables() -> () {
  spv.module "Logical" "VulkanKHR" {
    %2 = spv.Variable bind(1, 0) : !spv.ptr<f32, Input>
    %3 = spv.Variable bind(0, 1): !spv.ptr<f32, Output>
    %4 = spv.Variable {built_in = "GlobalInvocationId"} : !spv.ptr<vector<3xi32>, Input>
    %5 = spv.Variable built_in("GlobalInvocationId") : !spv.ptr<vector<3xi32>, Input>
  }
  return
}