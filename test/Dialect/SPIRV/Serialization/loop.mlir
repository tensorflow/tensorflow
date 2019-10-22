// RUN: mlir-translate -test-spirv-roundtrip %s | FileCheck %s

// Single loop

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
// CHECK-NEXT:     spv.BranchConditional %{{.*}} [1, 1], ^bb2, ^bb4
      spv.BranchConditional %cmp [1, 1], ^body, ^merge

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

// -----

// TODO(antiagainst): re-enable this after fixing the assertion failure.
// Nested loop

//spv.module "Logical" "GLSL450" {
  // for (int i = 0; i < count; ++i) {
  //   for (int j = 0; j < count; ++j) { }
  // }
  //func @loop(%count : i32) -> () {
    //%zero = spv.constant 0: i32
    //%one = spv.constant 1: i32
    //%ivar = spv.Variable init(%zero) : !spv.ptr<i32, Function>
    //%jvar = spv.Variable init(%zero) : !spv.ptr<i32, Function>

// TOCHECK:        spv.Branch ^bb1
// TOCHECK-NEXT: ^bb1:
// TOCHECK-NEXT:   spv.loop
    //spv.loop {
// TOCHECK-NEXT:     spv.Branch ^bb1
      //spv.Branch ^header

// TOCHECK-NEXT:   ^bb1:
    //^header:
// TOCHECK-NEXT:     spv.Load
      //%ival0 = spv.Load "Function" %ivar : i32
// TOCHECK-NEXT:     spv.SLessThan
      //%icmp = spv.SLessThan %ival0, %count : i32
// TOCHECK-NEXT:     spv.BranchConditional %{{.*}}, ^bb2, ^bb5
      //spv.BranchConditional %icmp, ^body, ^merge

// TOCHECK-NEXT:   ^bb2:
    //^body:
// TOCHECK-NEXT:     spv.constant 0
// TOCHECK-NEXT: 		 spv.Store
      //spv.Store "Function" %jvar, %zero : i32
// TOCHECK-NEXT:     spv.Branch ^bb3
// TOCHECK-NEXT:   ^bb3:
// TOCHECK-NEXT:     spv.loop
      //spv.loop {
// TOCHECK-NEXT:       spv.Branch ^bb1
        //spv.Branch ^header

// TOCHECK-NEXT:     ^bb1:
      //^header:
// TOCHECK-NEXT:       spv.Load
        //%jval0 = spv.Load "Function" %jvar : i32
// TOCHECK-NEXT:       spv.SLessThan
        //%jcmp = spv.SLessThan %jval0, %count : i32
// TOCHECK-NEXT:       spv.BranchConditional %{{.*}}, ^bb2, ^bb4
        //spv.BranchConditional %jcmp, ^body, ^merge

// TOCHECK-NEXT:     ^bb2:
      //^body:
        // Do nothing
// TOCHECK-NEXT:       spv.Branch ^bb3
        //spv.Branch ^continue

// TOCHECK-NEXT:     ^bb3:
      //^continue:
// TOCHECK-NEXT:       spv.Load
        //%jval1 = spv.Load "Function" %jvar : i32
// TOCHECK-NEXT:       spv.constant 1
// TOCHECK-NEXT:       spv.IAdd
        //%add = spv.IAdd %jval1, %one : i32
// TOCHECK-NEXT:       spv.Store
        //spv.Store "Function" %jvar, %add : i32
// TOCHECK-NEXT:       spv.Branch ^bb1
        //spv.Branch ^header

// TOCHECK-NEXT:     ^bb4:
      //^merge:
// TOCHECK-NEXT:       spv._merge
        //spv._merge
      //} // end inner loop

// TOCHECK:          spv.Branch ^bb4
      //spv.Branch ^continue

// TOCHECK-NEXT:   ^bb4:
    //^continue:
// TOCHECK-NEXT:     spv.Load
      //%ival1 = spv.Load "Function" %ivar : i32
// TOCHECK-NEXT:     spv.constant 1
// TOCHECK-NEXT:     spv.IAdd
      //%add = spv.IAdd %ival1, %one : i32
// TOCHECK-NEXT:     spv.Store
      //spv.Store "Function" %ivar, %add : i32
// TOCHECK-NEXT:     spv.Branch ^bb1
      //spv.Branch ^header

// TOCHECK-NEXT:   ^bb5:
// TOCHECK-NEXT:     spv._merge
    //^merge:
      //spv._merge
    //} // end outer loop
    //spv.Return
  //}

  //func @main() -> () {
    //spv.Return
  //}
  //spv.EntryPoint "GLCompute" @main
//} attributes {
  //capabilities = ["Shader"]
//}

