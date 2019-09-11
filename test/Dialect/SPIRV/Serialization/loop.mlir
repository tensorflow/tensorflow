// RUN: mlir-translate -serialize-spirv %s | mlir-translate -deserialize-spirv | FileCheck %s

spv.module "Logical" "GLSL450" {
  // for (int i = 0; i < count; ++i) {}
  func @loop(%count : i32) -> () {
    %zero = spv.constant 0: i32
    %one = spv.constant 1: i32
    %var = spv.Variable init(%zero) : !spv.ptr<i32, Function>

// CHECK:        spv.Branch ^bb1
// CHECK-NEXT: ^bb1:
// CHECK-NEXT:   spv.loop
    spv.loop {
// CHECK-NEXT:     spv.Branch ^bb1
      spv.Branch ^header

// CHECK-NEXT:   ^bb1:
    ^header:
// CHECK-NEXT:     spv.Load
      %val0 = spv.Load "Function" %var : i32
// CHECK-NEXT:     spv.SLessThan
      %cmp = spv.SLessThan %val0, %count : i32
// CHECK-NEXT:     spv.BranchConditional %{{.*}}, ^bb2, ^bb4
      spv.BranchConditional %cmp, ^body, ^merge

// CHECK-NEXT:   ^bb2:
    ^body:
      // Do nothing
// CHECK-NEXT:     spv.Branch ^bb3
      spv.Branch ^continue

// CHECK-NEXT:   ^bb3:
    ^continue:
// CHECK-NEXT:     spv.Load
      %val1 = spv.Load "Function" %var : i32
// CHECK-NEXT:     spv.constant 1
// CHECK-NEXT:     spv.IAdd
      %add = spv.IAdd %val1, %one : i32
// CHECK-NEXT:     spv.Store
      spv.Store "Function" %var, %add : i32
// CHECK-NEXT:     spv.Branch ^bb1
      spv.Branch ^header

// CHECK-NEXT:   ^bb4:
// CHECK-NEXT:     spv._merge
    ^merge:
      spv._merge
    }
    spv.Return
  }

  func @main() -> () {
    spv.Return
  }
  spv.EntryPoint "GLCompute" @main
} attributes {
  capabilities = ["Shader"]
}

