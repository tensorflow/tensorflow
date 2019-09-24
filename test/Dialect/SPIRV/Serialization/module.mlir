// RUN: mlir-translate -test-spirv-roundtrip -split-input-file %s | FileCheck %s

// CHECK:      spv.module "Logical" "GLSL450" {
// CHECK-NEXT:   func @foo() {
// CHECK-NEXT:     spv.Return
// CHECK-NEXT:   }
// CHECK-NEXT: } attributes {major_version = 1 : i32, minor_version = 0 : i32}

spv.module "Logical" "GLSL450" {
  func @foo() -> () {
     spv.Return
  }
}

// -----

spv.module "Logical" "GLSL450" {
} attributes {
  // CHECK: capabilities = ["Shader", "Float16"]
  capabilities = ["Shader", "Float16"]
}

// -----

spv.module "Logical" "GLSL450" {
} attributes {
  // CHECK: extensions = ["SPV_KHR_float_controls", "SPV_KHR_subgroup_vote"]
  extensions = ["SPV_KHR_float_controls", "SPV_KHR_subgroup_vote"]
}

