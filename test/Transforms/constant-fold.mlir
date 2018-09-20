// RUN: mlir-opt %s -constant-fold | FileCheck %s

mlfunc @test(%p : memref<f32>) {
  for %i0 = 0 to 128 {
    for %i1 = 0 to 8 { // CHECK: for %i1 = 0 to 8 {
      %0 = constant 4.5 : f32
      %1 = constant 1.5 : f32

      // CHECK-NEXT: %cst = constant 6.000000e+00 : f32
      %2 = "addf"(%0, %1) : (f32,f32) -> f32

      // CHECK-NEXT: store %cst, %arg0[]
      store %2, %p[] : memref<f32>
    }
  }
  return
}

// CHECK-LABEL: cfgfunc @simple_add
cfgfunc @simple_add() -> f32 {
bb0:   // CHECK: bb0:
  %0 = constant 4.5 : f32
  %1 = constant 1.5 : f32

  // CHECK-NEXT: %cst = constant 6.000000e+00 : f32
  %2 = "addf"(%0, %1) : (f32,f32) -> f32

  // CHECK-NEXT: return %cst
  return %2 : f32
}