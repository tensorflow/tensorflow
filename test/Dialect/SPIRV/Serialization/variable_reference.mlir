// RUN: mlir-translate -test-spirv-roundtrip %s | FileCheck %s

spv.module "Logical" "GLSL450" {
  spv.globalVariable @globalInvocationID built_in("GlobalInvocationId") : !spv.ptr<vector<3xi32>, Input>
  func @foo() {
    // CHECK: %[[ADDR:.*]] = spv._address_of @globalInvocationID : !spv.ptr<vector<3xi32>, Input>
    %0 = spv._address_of @globalInvocationID : !spv.ptr<vector<3xi32>, Input>
    %1 = spv.constant 0: i32
    // CHECK: spv.AccessChain %[[ADDR]]
    %2 = spv.AccessChain %0[%1] : !spv.ptr<vector<3xi32>, Input>
    spv.Return
  }
}
