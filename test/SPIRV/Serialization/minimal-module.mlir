// RUN: mlir-translate -serialize-spirv %s | mlir-translate -deserialize-spirv | FileCheck %s

// CHECK:      spv.module "Logical" "VulkanKHR" {
// CHECK-NEXT: } attributes {major_version = 1 : i32, minor_version = 0 : i32}

func @spirv_module() -> () {
  spv.module "Logical" "VulkanKHR" { }
  return
}
