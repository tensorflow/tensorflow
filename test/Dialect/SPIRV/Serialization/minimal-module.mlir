// RUN: mlir-translate -test-spirv-roundtrip %s | FileCheck %s

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
