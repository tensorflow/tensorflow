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

// CHECK-LABEL: cfgfunc @simple_addf
cfgfunc @simple_addf() -> f32 {
bb0:   // CHECK: bb0:
  %0 = constant 4.5 : f32
  %1 = constant 1.5 : f32

  // CHECK-NEXT: %cst = constant 6.000000e+00 : f32
  %2 = "addf"(%0, %1) : (f32,f32) -> f32

  // CHECK-NEXT: return %cst
  return %2 : f32
}

// CHECK-LABEL: cfgfunc @simple_addi
cfgfunc @simple_addi() -> i32 {
bb0:   // CHECK: bb0:
  %0 = constant 1 : i32
  %1 = constant 5 : i32

  // CHECK-NEXT: %c6_i32 = constant 6 : i32
  %2 = "addi"(%0, %1) : (i32,i32) -> i32

  // CHECK-NEXT: return %c6_i32
  return %2 : i32
}

// CHECK-LABEL: cfgfunc @simple_subf
cfgfunc @simple_subf() -> f32 {
bb0:   // CHECK: bb0:
  %0 = constant 4.5 : f32
  %1 = constant 1.5 : f32

  // CHECK-NEXT: %cst = constant 3.000000e+00 : f32
  %2 = "subf"(%0, %1) : (f32,f32) -> f32

  // CHECK-NEXT: return %cst
  return %2 : f32
}

// CHECK-LABEL: cfgfunc @simple_subi
cfgfunc @simple_subi() -> i32 {
bb0:   // CHECK: bb0:
  %0 = constant 4 : i32
  %1 = constant 1 : i32

  // CHECK-NEXT: %c3_i32 = constant 3 : i32
  %2 = "subi"(%0, %1) : (i32,i32) -> i32

  // CHECK-NEXT: return %c3_i32
  return %2 : i32
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

// CHECK-LABEL: cfgfunc @simple_mulf
cfgfunc @simple_mulf() -> f32 {
bb0:   // CHECK: bb0:
  %0 = constant 4.5 : f32
  %1 = constant 1.5 : f32

  // CHECK-NEXT: %cst = constant 6.750000e+00 : f32
  %2 = "mulf"(%0, %1) : (f32,f32) -> f32

  // CHECK-NEXT: return %cst
  return %2 : f32
}

// CHECK-LABEL: cfgfunc @simple_muli
cfgfunc @simple_muli() -> i32 {
bb0:   // CHECK: bb0:
  %0 = constant 4 : i32
  %1 = constant 2 : i32

  // CHECK-NEXT: %c8_i32 = constant 8 : i32
  %2 = "muli"(%0, %1) : (i32,i32) -> i32

  // CHECK-NEXT: return %c8_i32
  return %2 : i32
}
