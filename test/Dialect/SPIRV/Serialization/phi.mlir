// RUN: mlir-translate -split-input-file -test-spirv-roundtrip %s | FileCheck %s

// Test branch with one block argument

spv.module "Logical" "GLSL450" {
  func @foo() -> () {
// CHECK:        %[[CST:.*]] = spv.constant 0
    %zero = spv.constant 0 : i32
// CHECK-NEXT:   spv.Branch ^bb1(%[[CST]] : i32)
    spv.Branch ^bb1(%zero : i32)
// CHECK-NEXT: ^bb1(%{{.*}}: i32):
  ^bb1(%arg0: i32):
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

// Test branch with multiple block arguments

spv.module "Logical" "GLSL450" {
  func @foo() -> () {
// CHECK:        %[[ZERO:.*]] = spv.constant 0
    %zero = spv.constant 0 : i32
// CHECK-NEXT:   %[[ONE:.*]] = spv.constant 1
    %one = spv.constant 1.0 : f32
// CHECK-NEXT:   spv.Branch ^bb1(%[[ZERO]], %[[ONE]] : i32, f32)
    spv.Branch ^bb1(%zero, %one : i32, f32)

// CHECK-NEXT: ^bb1(%{{.*}}: i32, %{{.*}}: f32):     // pred: ^bb0
  ^bb1(%arg0: i32, %arg1: f32):
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

// Test using block arguments within branch

spv.module "Logical" "GLSL450" {
  func @foo() -> () {
// CHECK:        %[[CST0:.*]] = spv.constant 0
    %zero = spv.constant 0 : i32
// CHECK-NEXT:   spv.Branch ^bb1(%[[CST0]] : i32)
    spv.Branch ^bb1(%zero : i32)

// CHECK-NEXT: ^bb1(%[[ARG:.*]]: i32):
  ^bb1(%arg0: i32):
// CHECK-NEXT:   %[[ADD:.*]] = spv.IAdd %[[ARG]], %[[ARG]] : i32
    %0 = spv.IAdd %arg0, %arg0 : i32
// CHECK-NEXT:   %[[CST1:.*]] = spv.constant 0
// CHECK-NEXT:   spv.Branch ^bb2(%[[CST1]], %[[ADD]] : i32, i32)
    spv.Branch ^bb2(%zero, %0 : i32, i32)

// CHECK-NEXT: ^bb2(%{{.*}}: i32, %{{.*}}: i32):
  ^bb2(%arg1: i32, %arg2: i32):
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

// Test block not following domination order

spv.module "Logical" "GLSL450" {
  func @foo() -> () {
// CHECK:        spv.Branch ^bb1
    spv.Branch ^bb1

// CHECK-NEXT: ^bb1:
// CHECK-NEXT:   %[[ZERO:.*]] = spv.constant 0
// CHECK-NEXT:   %[[ONE:.*]] = spv.constant 1
// CHECK-NEXT:   spv.Branch ^bb2(%[[ZERO]], %[[ONE]] : i32, f32)

// CHECK-NEXT: ^bb2(%{{.*}}: i32, %{{.*}}: f32):
  ^bb2(%arg0: i32, %arg1: f32):
// CHECK-NEXT:   spv.Return
   spv.Return

  // This block is reordered to follow domination order.
  ^bb1:
    %zero = spv.constant 0 : i32
    %one = spv.constant 1.0 : f32
    spv.Branch ^bb2(%zero, %one : i32, f32)
  }

  func @main() -> () {
    spv.Return
  }
  spv.EntryPoint "GLCompute" @main
} attributes {
  capabilities = ["Shader"]
}

// -----

// Test multiple predecessors

spv.module "Logical" "GLSL450" {
  func @foo() -> () {
    %var = spv.Variable : !spv.ptr<i32, Function>

// CHECK:      spv.selection
    spv.selection {
      %true = spv.constant true
// CHECK:        spv.BranchConditional %{{.*}}, ^bb1, ^bb2
      spv.BranchConditional %true, ^true, ^false

// CHECK-NEXT: ^bb1:
    ^true:
// CHECK-NEXT:   %[[ZERO:.*]] = spv.constant 0
      %zero = spv.constant 0 : i32
// CHECK-NEXT:   spv.Branch ^bb3(%[[ZERO]] : i32)
      spv.Branch ^phi(%zero: i32)

// CHECK-NEXT: ^bb2:
    ^false:
// CHECK-NEXT:   %[[ONE:.*]] = spv.constant 1
      %one = spv.constant 1 : i32
// CHECK-NEXT:   spv.Branch ^bb3(%[[ONE]] : i32)
      spv.Branch ^phi(%one: i32)

// CHECK-NEXT: ^bb3(%[[ARG:.*]]: i32):
    ^phi(%arg: i32):
// CHECK-NEXT:   spv.Store "Function" %{{.*}}, %[[ARG]] : i32
      spv.Store "Function" %var, %arg : i32
// CHECK-NEXT:   spv.Return
      spv.Return

// CHECK-NEXT: ^bb4:
    ^merge:
// CHECK-NEXT:   spv._merge
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
