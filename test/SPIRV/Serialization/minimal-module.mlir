// RUN: mlir-translate -serialize-spirv %s | mlir-translate -deserialize-spirv | FileCheck %s

// CHECK-LABEL: func @spirv_module
// CHECK:      spv.module "Logical" "VulkanKHR" {
// CHECK-NEXT: } attributes {major_version = 1 : i32, minor_version = 0 : i32}

// TODO(ravishankarm) : The output produced is not correct, since it
// doesnt get the function body. The serialization doesnt handle
// functions yet. Change the CHECK once it does, to make sure the
// function is reproduced

func @spirv_module() -> () {
  spv.module "Logical" "VulkanKHR" {
    func @foo() -> () {
       spv.Return
    }
  }
  return
}
