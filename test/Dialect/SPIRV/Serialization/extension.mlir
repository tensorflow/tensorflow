// RUN: mlir-translate -test-spirv-roundtrip %s | FileCheck %s

spv.module "Logical" "GLSL450" {
} attributes {
  // CHECK: extensions = ["SPV_KHR_float_controls", "SPV_KHR_subgroup_vote"]
  extensions = ["SPV_KHR_float_controls", "SPV_KHR_subgroup_vote"]
}

