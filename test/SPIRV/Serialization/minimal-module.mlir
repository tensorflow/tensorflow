// RUN: mlir-translate -serialize-spirv %s | mlir-translate -deserialize-spirv | FileCheck %s

// CHECK-LABEL: func @spirv_module
// CHECK:      spv.module "Logical" "VulkanKHR" {
// CHECK-NEXT:   func @foo() {
// CHECK-NEXT:     spv.Return
// CHECK-NEXT:   }
// CHECK-NEXT: } attributes {major_version = 1 : i32, minor_version = 0 : i32}

func @spirv_module() -> () {
  spv.module "Logical" "VulkanKHR" {
    func @foo() -> () {
       spv.Return
    }
  }
  return
}
