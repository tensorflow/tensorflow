// RUN: mlir-translate -serialize-spirv %s | mlir-translate -deserialize-spirv | FileCheck %s

// CHECK:      spv.module {
// CHECK-NEXT: } attributes {addressing_model: "Logical", major_version: 1 : i32, memory_model: "VulkanKHR", minor_version: 0 : i32}

func @spirv_module() -> () {
  spv.module {
  } attributes {
    addressing_model: "Logical",
    memory_model: "VulkanKHR"
  }
  return
}
