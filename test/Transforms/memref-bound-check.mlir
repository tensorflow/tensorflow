// RUN: mlir-opt %s -memref-bound-check -split-input-file -verify | FileCheck %s

// -----

// CHECK-LABEL: mlfunc @test() {
mlfunc @test() {
  %zero = constant 0 : index
  %minusone = constant -1 : index
  %sym = constant 111 : index

  %A = alloc() : memref<9 x 9 x i32>
  %B = alloc() : memref<111 x i32>

  for %i = -1 to 9 {
    for %j = -1 to 9 {
      %idx = affine_apply (d0, d1) -> (d0, d1)(%i, %j)
      // Out of bound access.
      %x  = load %A[%idx#0, %idx#1] : memref<9 x 9 x i32>  
      // expected-error@-1 {{'load' op memref out of upper bound access along dimension #1}}
      // expected-error@-2 {{'load' op memref out of lower bound access along dimension #1}}
      // expected-error@-3 {{'load' op memref out of upper bound access along dimension #2}}
      // expected-error@-4 {{'load' op memref out of lower bound access along dimension #2}}
      // This will access 0 to 110 - hence an overflow.
      %idy = affine_apply (d0, d1) -> (10*d0 - d1 + 19)(%i, %j)
      %y = load %B[%idy] : memref<111 x i32>
    }
  }

  for %k = 0 to 9 {
      // In bound.
      %u = load %B[%zero] : memref<111 x i32>
      // Out of bounds.
      %v = load %B[%sym] : memref<111 x i32> // expected-error {{'load' op memref out of upper bound access along dimension #1}}
      // Out of bounds.
      %w = load %B[%minusone] : memref<111 x i32>  // expected-error {{'load' op memref out of lower bound access along dimension #1}}
  }
  return
}
