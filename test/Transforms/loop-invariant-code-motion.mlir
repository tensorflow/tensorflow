// RUN: mlir-opt %s -affine-loop-invariant-code-motion -split-input-file -verify | FileCheck %s

func @nested_loops_both_having_invariant_code() {
  %m = alloc() : memref<10xf32>
  %cf7 = constant 7.0 : f32
  %cf8 = constant 8.0 : f32

  affine.for %i0 = 0 to 10 {
    %v0 = addf %cf7, %cf8 : f32
    affine.for %i1 = 0 to 10 {
      store %v0, %m[%i0] : memref<10xf32>
    }
  }

  // CHECK: %0 = alloc() : memref<10xf32>
  // CHECK-NEXT: %cst = constant 7.000000e+00 : f32
  // CHECK-NEXT: %cst_0 = constant 8.000000e+00 : f32
  // CHECK-NEXT: %1 = addf %cst, %cst_0 : f32
  // CHECK-NEXT: affine.for %i0 = 0 to 10 {
  // CHECK-NEXT: store %1, %0[%i0] : memref<10xf32>
  // CHECK-NEXT: }
  // CHECK-NEXT: return
  return
}

// The store-load forwarding can see through affine apply's since it relies on
// dependence information.
// CHECK-LABEL: func @store_affine_apply
func @store_affine_apply() -> memref<10xf32> {
  %cf7 = constant 7.0 : f32
  %m = alloc() : memref<10xf32>
  affine.for %i0 = 0 to 10 {
      %t0 = affine.apply (d1) -> (d1 + 1)(%i0)
      store %cf7, %m[%t0] : memref<10xf32>
  }
  return %m : memref<10xf32>
// CHECK:       %cst = constant 7.000000e+00 : f32
// CHECK-NEXT:  %0 = alloc() : memref<10xf32>
// CHECK-NEXT:  affine.for %i0 = 0 to 10 {
// CHECK-NEXT:      %1 = affine.apply #map2(%i0)
// CHECK-NEXT:      store %cst, %0[%1] : memref<10xf32>
// CHECK-NEXT:  }
// CHECK-NEXT:  return %0 : memref<10xf32>
}

func @nested_loops_code_invariant_to_both() {
  %m = alloc() : memref<10xf32>
  %cf7 = constant 7.0 : f32
  %cf8 = constant 8.0 : f32

  affine.for %i0 = 0 to 10 {
    affine.for %i1 = 0 to 10 {
      %v0 = addf %cf7, %cf8 : f32
    }
  }

  // CHECK: %0 = alloc() : memref<10xf32>
  // CHECK-NEXT: %cst = constant 7.000000e+00 : f32
  // CHECK-NEXT: %cst_0 = constant 8.000000e+00 : f32
  // CHECK-NEXT: %1 = addf %cst, %cst_0 : f32
  // CHECK-NEXT: return
  return
}

func @single_loop_nothing_invariant() {
  %m1 = alloc() : memref<10xf32>
  %m2 = alloc() : memref<10xf32>
  affine.for %i0 = 0 to 10 {
    %v0 = load %m1[%i0] : memref<10xf32>
    %v1 = load %m2[%i0] : memref<10xf32>
    %v2 = addf %v0, %v1 : f32
    store %v2, %m1[%i0] : memref<10xf32>
  }

  // CHECK: %0 = alloc() : memref<10xf32>
  // CHECK-NEXT: %1 = alloc() : memref<10xf32>
  // CHECK-NEXT: affine.for %i0 = 0 to 10 {
  // CHECK-NEXT: %2 = load %0[%i0] : memref<10xf32>
  // CHECK-NEXT: %3 = load %1[%i0] : memref<10xf32>
  // CHECK-NEXT: %4 = addf %2, %3 : f32
  // CHECK-NEXT: store %4, %0[%i0] : memref<10xf32>
  // CHECK-NEXT: }
  // CHECK-NEXT: return
  return
}


func @invariant_code_inside_affine_if() {
  %m = alloc() : memref<10xf32>
  %cf8 = constant 8.0 : f32

  affine.for %i0 = 0 to 10 {
    %t0 = affine.apply (d1) -> (d1 + 1)(%i0)
    affine.if (d0, d1) : (d1 - d0 >= 0) (%i0, %t0) {
        %cf9 = addf %cf8, %cf8 : f32
        store %cf9, %m[%i0] : memref<10xf32>

    }
  }

  // CHECK: %0 = alloc() : memref<10xf32>
  // CHECK-NEXT: %cst = constant 8.000000e+00 : f32
  // CHECK-NEXT: affine.for %i0 = 0 to 10 {
  // CHECK-NEXT: %1 = affine.apply #map2(%i0)
  // CHECK-NEXT: affine.if #set0(%i0, %1) {
  // CHECK-NEXT: %2 = addf %cst, %cst : f32
  // CHECK-NEXT: store %2, %0[%i0] : memref<10xf32>
  // CHECK-NEXT: }
  // CHECK-NEXT: }
  // CHECK-NEXT: return

  return
}


func @nested_loops_with_common_and_uncommon_invariant_code() {
  %m = alloc() : memref<10xf32>
  %cf7 = constant 7.0 : f32
  %cf8 = constant 8.0 : f32

  affine.for %i0 = 0 to 10 {
    %v0 = addf %cf7, %cf8 : f32
    affine.for %i1 = 0 to 10 {
      %v1 = addf %cf7, %cf7 : f32
      store %v0, %m[%i1] : memref<10xf32>
      store %v0, %m[%i0] : memref<10xf32>
    }
  }

  // CHECK: %0 = alloc() : memref<10xf32>
  // CHECK-NEXT: %cst = constant 7.000000e+00 : f32
  // CHECK-NEXT: %cst_0 = constant 8.000000e+00 : f32
  // CHECK-NEXT: %1 = addf %cst, %cst_0 : f32
  // CHECK-NEXT: %2 = addf %cst, %cst : f32
  // CHECK-NEXT: affine.for %i0 = 0 to 10 {
  // CHECK-NEXT: store %1, %0[%i0] : memref<10xf32>
  // CHECK-NEXT: affine.for %i1 = 0 to 10 {
  // CHECK-NEXT:   store %1, %0[%i1] : memref<10xf32>
  // CHECK-NEXT:  }
  // CHECK-NEXT: }
  // CHECK-NEXT: return
  return
}

func @invariant_affine_if() {
  %m = alloc() : memref<10xf32>
  %cf8 = constant 8.0 : f32
  affine.for %i0 = 0 to 10 {
    affine.for %i1 = 0 to 10 {
      affine.if (d0, d1) : (d1 - d0 >= 0) (%i0, %i0) {
          %cf9 = addf %cf8, %cf8 : f32
          store %cf9, %m[%i0] : memref<10xf32>

      }
    }
  }

  // CHECK: %0 = alloc() : memref<10xf32>
  // CHECK-NEXT: %cst = constant 8.000000e+00 : f32
  // CHECK-NEXT: affine.for %i0 = 0 to 10 {
  // CHECK-NEXT: affine.if #set0(%i0, %i0) {
  // CHECK-NEXT: %1 = addf %cst, %cst : f32
  // CHECK-NEXT: store %1, %0[%i0] : memref<10xf32>
  // CHECK-NEXT: }
  // CHECK-NEXT: }
  // CHECK-NEXT: return

  return
}

func @invariant_constant_and_load() {
  %m = alloc() : memref<100xf32>
  affine.for %i0 = 0 to 5 {
    %c0 = constant 0 : index
    %v = load %m[%c0] : memref<100xf32>
    store %v, %m[%i0] : memref<100xf32>
  }

  // CHECK: %0 = alloc() : memref<100xf32>
  // CHECK-NEXT: %c0 = constant 0 : index
  // CHECK-NEXT: %1 = load %0[%c0] : memref<100xf32>
  // CHECK-NEXT: affine.for %i0 = 0 to 5 {
  // CHECK-NEXT:  store %1, %0[%i0] : memref<100xf32>
  // CHECK-NEXT: }
  // CHECK-NEXT: return

  return
}

