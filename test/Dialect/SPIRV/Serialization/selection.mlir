// RUN: mlir-translate -test-spirv-roundtrip %s | FileCheck %s

spv.module "Logical" "GLSL450" {
  func @selection(%cond: i1) -> () {
    %zero = spv.constant 0: i32
    %one = spv.constant 1: i32
    %two = spv.constant 2: i32
    %var = spv.Variable init(%zero) : !spv.ptr<i32, Function>

// CHECK:        spv.Branch ^bb1
// CHECK-NEXT: ^bb1:
// CHECK-NEXT:   spv.selection
    spv.selection {
// CHECK-NEXT: spv.BranchConditional %{{.*}}, ^bb1, ^bb2
      spv.BranchConditional %cond, ^then, ^else

// CHECK-NEXT:   ^bb1:
    ^then:
// CHECK-NEXT:     spv.constant 1
// CHECK-NEXT:     spv.Store
      spv.Store "Function" %var, %one : i32
// CHECK-NEXT:     spv.Branch ^bb3
      spv.Branch ^merge

// CHECK-NEXT:   ^bb2:
    ^else:
// CHECK-NEXT:     spv.constant 2
// CHECK-NEXT:     spv.Store
      spv.Store "Function" %var, %two : i32
// CHECK-NEXT:     spv.Branch ^bb3
      spv.Branch ^merge

// CHECK-NEXT:   ^bb3:
    ^merge:
// CHECK-NEXT:     spv._merge
      spv._merge
    }

    spv.Return
  }

  func @main() -> () {
    spv.Return
  }
  spv.EntryPoint "GLCompute" @main
  spv.ExecutionMode @main "LocalSize", 1, 1, 1
} attributes {
  capabilities = ["Shader"]
}
