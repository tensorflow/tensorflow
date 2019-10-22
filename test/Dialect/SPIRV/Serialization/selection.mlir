// RUN: mlir-translate -test-spirv-roundtrip -split-input-file %s | FileCheck %s

// Selection with both then and else branches

spv.module "Logical" "GLSL450" {
  func @selection(%cond: i1) -> () {
// CHECK:        spv.Branch ^bb1
// CHECK-NEXT: ^bb1:
    %zero = spv.constant 0: i32
    %one = spv.constant 1: i32
    %two = spv.constant 2: i32
    %var = spv.Variable init(%zero) : !spv.ptr<i32, Function>

// CHECK-NEXT:   spv.selection {
// CHECK-NEXT:     spv.constant 0
// CHECK-NEXT:     spv.Variable
    spv.selection {
// CHECK-NEXT: spv.BranchConditional %{{.*}} [5, 10], ^bb1, ^bb2
      spv.BranchConditional %cond [5, 10], ^then, ^else

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

// -----

// Selection with only then branch
// Selection in function entry block

spv.module "Logical" "GLSL450" {
// CHECK:      func @selection(%[[ARG:.*]]: i1
  func @selection(%cond: i1) -> (i32) {
// CHECK:        spv.Branch ^bb1
// CHECK-NEXT: ^bb1:
// CHECK-NEXT:   spv.selection
    spv.selection {
// CHECK-NEXT: spv.BranchConditional %[[ARG]], ^bb1, ^bb2
      spv.BranchConditional %cond, ^then, ^merge

// CHECK:        ^bb1:
    ^then:
      %zero = spv.constant 0 : i32
      spv.ReturnValue  %zero : i32

// CHECK:        ^bb2:
    ^merge:
// CHECK-NEXT:     spv._merge
      spv._merge
    }

    %one = spv.constant 1 : i32
    spv.ReturnValue  %one : i32
  }

  func @main() -> () {
    spv.Return
  }
  spv.EntryPoint "GLCompute" @main
  spv.ExecutionMode @main "LocalSize", 1, 1, 1
} attributes {
  capabilities = ["Shader"]
}

