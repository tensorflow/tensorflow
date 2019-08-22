// RUN: mlir-translate -serialize-spirv %s | mlir-translate -deserialize-spirv | FileCheck %s

spv.module "Logical" "GLSL450" {
} attributes {
  // CHECK: capabilities = ["Shader", "Float16"]
  capabilities = ["Shader", "Float16"]
}
