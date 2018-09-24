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

mlfunc @constant_fold_affine_apply() {
  %0 = alloc() : memref<8x8xf32>

  %c177 = constant 177 : affineint
  %c211 = constant 211 : affineint
  %N = constant 1075 : affineint

  // CHECK: %c1159 = constant 1159 : affineint
  // CHECK: %c1152 = constant 1152 : affineint
  %x  = "affine_apply"(%c177, %c211, %N) {map: (d0, d1)[S0] -> ( (d0 + 128 * S0) floordiv 128 + d1 mod 128, 128 * (S0 ceildiv 128) )} : (affineint, affineint, affineint) -> (affineint, affineint)

  // CHECK: {{[0-9]+}} = load %0[%c1159, %c1152] : memref<8x8xf32>  
  %v = load %0[%x#0, %x#1] : memref<8x8xf32>
  return
}
