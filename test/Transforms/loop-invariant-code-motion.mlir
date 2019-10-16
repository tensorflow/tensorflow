// RUN: mlir-opt %s -loop-invariant-code-motion -split-input-file | FileCheck %s

func @nested_loops_both_having_invariant_code() {
  %m = alloc() : memref<10xf32>
  %cf7 = constant 7.0 : f32
  %cf8 = constant 8.0 : f32

  affine.for %arg0 = 0 to 10 {
    %v0 = addf %cf7, %cf8 : f32
    affine.for %arg1 = 0 to 10 {
      %v1 = addf %v0, %cf8 : f32
      affine.store %v0, %m[%arg0] : memref<10xf32>
    }
  }

  // CHECK: %0 = alloc() : memref<10xf32>
  // CHECK-NEXT: %[[CST0:.*]] = constant 7.000000e+00 : f32
  // CHECK-NEXT: %[[CST1:.*]] = constant 8.000000e+00 : f32
  // CHECK-NEXT: %[[ADD0:.*]] = addf %[[CST0]], %[[CST1]] : f32
  // CHECK-NEXT: addf %[[ADD0]], %[[CST1]] : f32
  // CHECK-NEXT: affine.for
  // CHECK-NEXT: affine.for
  // CHECK-NEXT: affine.store

  return
}

func @nested_loops_code_invariant_to_both() {
  %m = alloc() : memref<10xf32>
  %cf7 = constant 7.0 : f32
  %cf8 = constant 8.0 : f32

  affine.for %arg0 = 0 to 10 {
    affine.for %arg1 = 0 to 10 {
      %v0 = addf %cf7, %cf8 : f32
    }
  }

  // CHECK: %0 = alloc() : memref<10xf32>
  // CHECK-NEXT: %cst = constant 7.000000e+00 : f32
  // CHECK-NEXT: %cst_0 = constant 8.000000e+00 : f32
  // CHECK-NEXT: %1 = addf %cst, %cst_0 : f32

  return
}

func @single_loop_nothing_invariant() {
  %m1 = alloc() : memref<10xf32>
  %m2 = alloc() : memref<10xf32>
  affine.for %arg0 = 0 to 10 {
    %v0 = affine.load %m1[%arg0] : memref<10xf32>
    %v1 = affine.load %m2[%arg0] : memref<10xf32>
    %v2 = addf %v0, %v1 : f32
    affine.store %v2, %m1[%arg0] : memref<10xf32>
  }

  // CHECK: %0 = alloc() : memref<10xf32>
  // CHECK-NEXT: %1 = alloc() : memref<10xf32>
  // CHECK-NEXT: affine.for %arg0 = 0 to 10 {
  // CHECK-NEXT: %2 = affine.load %0[%arg0] : memref<10xf32>
  // CHECK-NEXT: %3 = affine.load %1[%arg0] : memref<10xf32>
  // CHECK-NEXT: %4 = addf %2, %3 : f32
  // CHECK-NEXT: affine.store %4, %0[%arg0] : memref<10xf32>

  return
}

func @invariant_code_inside_affine_if() {
  %m = alloc() : memref<10xf32>
  %cf8 = constant 8.0 : f32

  affine.for %arg0 = 0 to 10 {
    %t0 = affine.apply (d1) -> (d1 + 1)(%arg0)
    affine.if (d0, d1) : (d1 - d0 >= 0) (%arg0, %t0) {
        %cf9 = addf %cf8, %cf8 : f32
        affine.store %cf9, %m[%arg0] : memref<10xf32>

    }
  }

  // CHECK: %0 = alloc() : memref<10xf32>
  // CHECK-NEXT: %cst = constant 8.000000e+00 : f32
  // CHECK-NEXT: affine.for %arg0 = 0 to 10 {
  // CHECK-NEXT: %1 = affine.apply #map3(%arg0)
  // CHECK-NEXT: affine.if #set0(%arg0, %1) {
  // CHECK-NEXT: %2 = addf %cst, %cst : f32
  // CHECK-NEXT: affine.store %2, %0[%arg0] : memref<10xf32>
  // CHECK-NEXT: }


  return
}

func @invariant_affine_if() {
  %m = alloc() : memref<10xf32>
  %cf8 = constant 8.0 : f32
  affine.for %arg0 = 0 to 10 {
    affine.for %arg1 = 0 to 10 {
      affine.if (d0, d1) : (d1 - d0 >= 0) (%arg0, %arg0) {
          %cf9 = addf %cf8, %cf8 : f32
      }
    }
  }

  // CHECK: %0 = alloc() : memref<10xf32>
  // CHECK-NEXT: %[[CST:.*]] = constant 8.000000e+00 : f32
  // CHECK-NEXT: affine.for %[[ARG:.*]] = 0 to 10 {
  // CHECK-NEXT: affine.if #set0(%[[ARG]], %[[ARG]]) {
  // CHECK-NEXT: addf %[[CST]], %[[CST]] : f32
  // CHECK-NEXT: }

  return
}

func @invariant_affine_if2() {
  %m = alloc() : memref<10xf32>
  %cf8 = constant 8.0 : f32
  affine.for %arg0 = 0 to 10 {
    affine.for %arg1 = 0 to 10 {
      affine.if (d0, d1) : (d1 - d0 >= 0) (%arg0, %arg0) {
          %cf9 = addf %cf8, %cf8 : f32
          affine.store %cf9, %m[%arg1] : memref<10xf32>
      }
    }
  }

  // CHECK: alloc
  // CHECK-NEXT: constant
  // CHECK-NEXT: affine.for
  // CHECK-NEXT: affine.for
  // CHECK-NEXT: affine.if
  // CHECK-NEXT: addf
  // CHECK-NEXT: affine.store
  // CHECK-NEXT: }
  // CHECK-NEXT: }

  return
}

func @invariant_affine_nested_if() {
  %m = alloc() : memref<10xf32>
  %cf8 = constant 8.0 : f32
  affine.for %arg0 = 0 to 10 {
    affine.for %arg1 = 0 to 10 {
      affine.if (d0, d1) : (d1 - d0 >= 0) (%arg0, %arg0) {
          %cf9 = addf %cf8, %cf8 : f32
          affine.if (d0, d1) : (d1 - d0 >= 0) (%arg0, %arg0) {
            %cf10 = addf %cf9, %cf9 : f32
          }
      }
    }
  }

  // CHECK: alloc
  // CHECK-NEXT: constant
  // CHECK-NEXT: affine.for
  // CHECK-NEXT: affine.for
  // CHECK-NEXT: affine.if
  // CHECK-NEXT: addf
  // CHECK-NEXT: affine.if
  // CHECK-NEXT: addf
  // CHECK-NEXT: }
  // CHECK-NEXT: }
  // CHECK-NEXT: }


  return
}

func @invariant_affine_nested_if_else() {
  %m = alloc() : memref<10xf32>
  %cf8 = constant 8.0 : f32
  affine.for %arg0 = 0 to 10 {
    affine.for %arg1 = 0 to 10 {
      affine.if (d0, d1) : (d1 - d0 >= 0) (%arg0, %arg0) {
          %cf9 = addf %cf8, %cf8 : f32
          affine.store %cf9, %m[%arg0] : memref<10xf32>
          affine.if (d0, d1) : (d1 - d0 >= 0) (%arg0, %arg0) {
            %cf10 = addf %cf9, %cf9 : f32
          } else {
            affine.store %cf9, %m[%arg1] : memref<10xf32>
          }
      }
    }
  }

  // CHECK: alloc
  // CHECK-NEXT: constant
  // CHECK-NEXT: affine.for
  // CHECK-NEXT: affine.for
  // CHECK-NEXT: affine.if
  // CHECK-NEXT: addf
  // CHECK-NEXT: affine.store
  // CHECK-NEXT: affine.if
  // CHECK-NEXT: addf
  // CHECK-NEXT: } else {
  // CHECK-NEXT: affine.store
  // CHECK-NEXT: }
  // CHECK-NEXT: }
  // CHECK-NEXT: }


  return
}

func @invariant_loop_dialect() {
  %ci0 = constant 0 : index
  %ci10 = constant 10 : index
  %ci1 = constant 1 : index
  %m = alloc() : memref<10xf32>
  %cf7 = constant 7.0 : f32
  %cf8 = constant 8.0 : f32
  loop.for %arg0 = %ci0 to %ci10 step %ci1 {
    loop.for %arg1 = %ci0 to %ci10 step %ci1 {
      %v0 = addf %cf7, %cf8 : f32
    }
  }

  // CHECK: %0 = alloc() : memref<10xf32>
  // CHECK-NEXT: %cst = constant 7.000000e+00 : f32
  // CHECK-NEXT: %cst_0 = constant 8.000000e+00 : f32
  // CHECK-NEXT: %1 = addf %cst, %cst_0 : f32

  return
}

func @variant_loop_dialect() {
  %ci0 = constant 0 : index
  %ci10 = constant 10 : index
  %ci1 = constant 1 : index
  %m = alloc() : memref<10xf32>
  loop.for %arg0 = %ci0 to %ci10 step %ci1 {
    loop.for %arg1 = %ci0 to %ci10 step %ci1 {
      %v0 = addi %arg0, %arg1 : index
    }
  }

  // CHECK: %0 = alloc() : memref<10xf32>
  // CHECK-NEXT: loop.for
  // CHECK-NEXT: loop.for
  // CHECK-NEXT: addi

  return
}
