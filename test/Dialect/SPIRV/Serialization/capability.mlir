// RUN: mlir-translate -test-spirv-roundtrip %s | FileCheck %s

spv.module "Logical" "GLSL450" {
} attributes {
  // CHECK: capabilities = ["Shader", "Float16"]
  capabilities = ["Shader", "Float16"]
}
