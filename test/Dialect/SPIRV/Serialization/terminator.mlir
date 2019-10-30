// RUN: mlir-translate -test-spirv-roundtrip %s | FileCheck %s

spv.module "Logical" "GLSL450" {
  // CHECK-LABEL: @ret
  func @ret() -> () {
    // CHECK: spv.Return
    spv.Return
  }

  // CHECK-LABEL: @ret_val
  func @ret_val() -> (i32) {
    %0 = spv.Variable : !spv.ptr<i32, Function>
    %1 = spv.Load "Function" %0 : i32
    // CHECK: spv.ReturnValue {{.*}} : i32
    spv.ReturnValue %1 : i32
  }

  // CHECK-LABEL: @unreachable
  func @unreachable() {
    spv.Return
  // CHECK-NOT: ^bb
  ^bb1:
    // Unreachable blocks will be dropped during serialization.
    // CHECK-NOT: spv.Unreachable
    spv.Unreachable
  }
}
