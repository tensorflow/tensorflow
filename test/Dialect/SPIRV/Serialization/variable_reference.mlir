// RUN: mlir-translate -serialize-spirv %s

// TODO: This example doesn't work on deserialization since constants
// are always added to module scope and need to be materialized into
// function scope. So for now just run the serialization.

func @spirv_global_vars() -> () {
  spv.module "Logical" "VulkanKHR" {
    spv.globalVariable @globalInvocationID built_in("GlobalInvocationId") : !spv.ptr<vector<3xi32>, Input>
    func @foo() {
      %0 = spv._address_of @globalInvocationID : !spv.ptr<vector<3xi32>, Input>
      %1 = spv.constant 0: i32
      %2 = spv.AccessChain %0[%1] : !spv.ptr<vector<3xi32>, Input>
      spv.Return
    }
  }
  return
}