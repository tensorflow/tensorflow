// RUN: mlir-translate -serialize-spirv %s | mlir-translate -deserialize-spirv | FileCheck %s

// CHECK:      spv.globalVariable @var0 bind(1, 0) : !spv.ptr<f32, Input>
// CHECK-NEXT: spv.globalVariable @var1 bind(0, 1) : !spv.ptr<f32, Output>
// CHECK-NEXT: spv.globalVariable @var2 built_in("GlobalInvocationId") : !spv.ptr<vector<3xi32>, Input>
// CHECK-NEXT: spv.globalVariable @var3 built_in("GlobalInvocationId") : !spv.ptr<vector<3xi32>, Input>

spv.module "Logical" "GLSL450" {
  spv.globalVariable @var0 bind(1, 0) : !spv.ptr<f32, Input>
  spv.globalVariable @var1 bind(0, 1) : !spv.ptr<f32, Output>
  spv.globalVariable @var2 {built_in = "GlobalInvocationId"} : !spv.ptr<vector<3xi32>, Input>
  spv.globalVariable @var3 built_in("GlobalInvocationId") : !spv.ptr<vector<3xi32>, Input>
}
