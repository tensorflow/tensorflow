// RUN: mlir-opt %s -affine-loop-invariant-code-motion -split-input-file | FileCheck %s

func @nested_loops_both_having_invariant_code() {
  %m = alloc() : memref<10xf32>
  %cf7 = constant 7.0 : f32
  %cf8 = constant 8.0 : f32

  affine.for %arg0 = 0 to 10 {
    %v0 = addf %cf7, %cf8 : f32
    affine.for %arg1 = 0 to 10 {
      affine.store %v0, %m[%arg0] : memref<10xf32>
    }
  }

  // CHECK: %0 = alloc() : memref<10xf32>
  // CHECK-NEXT: %cst = constant 7.000000e+00 : f32
  // CHECK-NEXT: %cst_0 = constant 8.000000e+00 : f32
  // CHECK-NEXT: %1 = addf %cst, %cst_0 : f32
  // CHECK-NEXT: affine.for %arg0 = 0 to 10 {
  // CHECK-NEXT: affine.store %1, %0[%arg0] : memref<10xf32>

  return
}

// The store-load forwarding can see through affine apply's since it relies on
// dependence information.
// CHECK-LABEL: func @store_affine_apply
func @store_affine_apply() -> memref<10xf32> {
  %cf7 = constant 7.0 : f32
  %m = alloc() : memref<10xf32>
  affine.for %arg0 = 0 to 10 {
      %t0 = affine.apply (d1) -> (d1 + 1)(%arg0)
      affine.store %cf7, %m[%t0] : memref<10xf32>
  }
  return %m : memref<10xf32>
// CHECK:       %cst = constant 7.000000e+00 : f32
// CHECK-NEXT:  %0 = alloc() : memref<10xf32>
// CHECK-NEXT:  affine.for %arg0 = 0 to 10 {
// CHECK-NEXT:      %1 = affine.apply #map3(%arg0)
// CHECK-NEXT:      affine.store %cst, %0[%1] : memref<10xf32>
// CHECK-NEXT:  }
// CHECK-NEXT:  return %0 : memref<10xf32>
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


func @dependent_stores() {
  %m = alloc() : memref<10xf32>
  %cf7 = constant 7.0 : f32
  %cf8 = constant 8.0 : f32

  affine.for %arg0 = 0 to 10 {
    %v0 = addf %cf7, %cf8 : f32
    affine.for %arg1 = 0 to 10 {
      %v1 = addf %cf7, %cf7 : f32
      affine.store %v1, %m[%arg1] : memref<10xf32>
      affine.store %v0, %m[%arg0] : memref<10xf32>
    }
  }

  // CHECK: %0 = alloc() : memref<10xf32>
  // CHECK-NEXT: %cst = constant 7.000000e+00 : f32
  // CHECK-NEXT: %cst_0 = constant 8.000000e+00 : f32
  // CHECK-NEXT: %1 = addf %cst, %cst_0 : f32
  // CHECK-NEXT: %2 = addf %cst, %cst : f32
  // CHECK-NEXT: affine.for %arg0 = 0 to 10 {

  // CHECK-NEXT: affine.for %arg1 = 0 to 10 {
  // CHECK-NEXT:   affine.store %2, %0[%arg1] : memref<10xf32>
  // CHECK-NEXT:   affine.store %1, %0[%arg0] : memref<10xf32>

  return
}

func @independent_stores() {
  %m = alloc() : memref<10xf32>
  %cf7 = constant 7.0 : f32
  %cf8 = constant 8.0 : f32

  affine.for %arg0 = 0 to 10 {
    %v0 = addf %cf7, %cf8 : f32
    affine.for %arg1 = 0 to 10 {
      %v1 = addf %cf7, %cf7 : f32
      affine.store %v0, %m[%arg0] : memref<10xf32>
      affine.store %v1, %m[%arg1] : memref<10xf32>
    }
  }

  // CHECK: %0 = alloc() : memref<10xf32>
  // CHECK-NEXT: %cst = constant 7.000000e+00 : f32
  // CHECK-NEXT: %cst_0 = constant 8.000000e+00 : f32
  // CHECK-NEXT: %1 = addf %cst, %cst_0 : f32
  // CHECK-NEXT: %2 = addf %cst, %cst : f32
  // CHECK-NEXT: affine.for %arg0 = 0 to 10 {
  // CHECK-NEXT:   affine.for %arg1 = 0 to 10 {
  // CHECK-NEXT:     affine.store %1, %0[%arg0] : memref<10xf32>
  // CHECK-NEXT:     affine.store %2, %0[%arg1] : memref<10xf32>
  // CHECK-NEXT:    }

  return
}

func @load_dependent_store() {
  %m = alloc() : memref<10xf32>
  %cf7 = constant 7.0 : f32
  %cf8 = constant 8.0 : f32

  affine.for %arg0 = 0 to 10 {
    %v0 = addf %cf7, %cf8 : f32
    affine.for %arg1 = 0 to 10 {
      %v1 = addf %cf7, %cf7 : f32
      affine.store %v0, %m[%arg1] : memref<10xf32>
      %v2 = affine.load %m[%arg0] : memref<10xf32>
    }
  }

  // CHECK: %0 = alloc() : memref<10xf32>
  // CHECK-NEXT: %cst = constant 7.000000e+00 : f32
  // CHECK-NEXT: %cst_0 = constant 8.000000e+00 : f32
  // CHECK-NEXT: %1 = addf %cst, %cst_0 : f32
  // CHECK-NEXT: %2 = addf %cst, %cst : f32
  // CHECK-NEXT: affine.for %arg0 = 0 to 10 {
  // CHECK-NEXT: affine.for %arg1 = 0 to 10 {
  // CHECK-NEXT:   affine.store %1, %0[%arg1] : memref<10xf32>
  // CHECK-NEXT:   %3 = affine.load %0[%arg0] : memref<10xf32>

  return
}

func @load_after_load() {
  %m = alloc() : memref<10xf32>
  %cf7 = constant 7.0 : f32
  %cf8 = constant 8.0 : f32

  affine.for %arg0 = 0 to 10 {
    %v0 = addf %cf7, %cf8 : f32
    affine.for %arg1 = 0 to 10 {
      %v1 = addf %cf7, %cf7 : f32
      %v3 = affine.load %m[%arg1] : memref<10xf32>
      %v2 = affine.load %m[%arg0] : memref<10xf32>
    }
  }

  // CHECK: %0 = alloc() : memref<10xf32>
  // CHECK-NEXT: %cst = constant 7.000000e+00 : f32
  // CHECK-NEXT: %cst_0 = constant 8.000000e+00 : f32
  // CHECK-NEXT: %1 = addf %cst, %cst_0 : f32
  // CHECK-NEXT: %2 = addf %cst, %cst : f32
  // CHECK-NEXT: affine.for %arg0 = 0 to 10 {
  // CHECK-NEXT: %3 = affine.load %0[%arg0] : memref<10xf32>
  // CHECK-NEXT: affine.for %arg1 = 0 to 10 {
  // CHECK-NEXT: %4 = affine.load %0[%arg1] : memref<10xf32>

  return
}

func @invariant_affine_if() {
  %m = alloc() : memref<10xf32>
  %cf8 = constant 8.0 : f32
  affine.for %arg0 = 0 to 10 {
    affine.for %arg1 = 0 to 10 {
      affine.if (d0, d1) : (d1 - d0 >= 0) (%arg0, %arg0) {
          %cf9 = addf %cf8, %cf8 : f32
          affine.store %cf9, %m[%arg0] : memref<10xf32>

      }
    }
  }

  // CHECK: %0 = alloc() : memref<10xf32>
  // CHECK-NEXT: %cst = constant 8.000000e+00 : f32
  // CHECK-NEXT: affine.for %arg0 = 0 to 10 {
  // CHECK-NEXT: affine.if #set0(%arg0, %arg0) {
  // CHECK-NEXT: %1 = addf %cst, %cst : f32
  // CHECK-NEXT: affine.store %1, %0[%arg0] : memref<10xf32>
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

  // CHECK: %0 = alloc() : memref<10xf32>
  // CHECK-NEXT: %cst = constant 8.000000e+00 : f32
  // CHECK-NEXT: affine.for %arg0 = 0 to 10 {
  // CHECK-NEXT: affine.for %arg1 = 0 to 10 {
  // CHECK-NEXT: affine.if #set0(%arg0, %arg0) {
  // CHECK-NEXT: %1 = addf %cst, %cst : f32
  // CHECK-NEXT: affine.store %1, %0[%arg1] : memref<10xf32>
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
          affine.store %cf9, %m[%arg0] : memref<10xf32>
          affine.if (d0, d1) : (d1 - d0 >= 0) (%arg0, %arg0) {
            affine.store %cf9, %m[%arg1] : memref<10xf32>
          }
      }
    }
  }

  // CHECK: %0 = alloc() : memref<10xf32>
  // CHECK-NEXT: %cst = constant 8.000000e+00 : f32
  // CHECK-NEXT: affine.for %arg0 = 0 to 10 {
  // CHECK-NEXT: affine.for %arg1 = 0 to 10 {
  // CHECK-NEXT: affine.if #set0(%arg0, %arg0) {
  // CHECK-NEXT: %1 = addf %cst, %cst : f32
  // CHECK-NEXT: affine.store %1, %0[%arg0] : memref<10xf32>
  // CHECK-NEXT: affine.if #set0(%arg0, %arg0) {
  // CHECK-NEXT: affine.store %1, %0[%arg1] : memref<10xf32>
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
            affine.store %cf9, %m[%arg0] : memref<10xf32>
          } else {
            affine.store %cf9, %m[%arg1] : memref<10xf32>
          }
      }
    }
  }

  // CHECK: %0 = alloc() : memref<10xf32>
  // CHECK-NEXT: %cst = constant 8.000000e+00 : f32
  // CHECK-NEXT: affine.for %arg0 = 0 to 10 {
  // CHECK-NEXT: affine.for %arg1 = 0 to 10 {
  // CHECK-NEXT: affine.if #set0(%arg0, %arg0) {
  // CHECK-NEXT: %1 = addf %cst, %cst : f32
  // CHECK-NEXT: affine.store %1, %0[%arg0] : memref<10xf32>
  // CHECK-NEXT: affine.if #set0(%arg0, %arg0) {
  // CHECK-NEXT: affine.store %1, %0[%arg0] : memref<10xf32>
  // CHECK-NEXT: } else {
  // CHECK-NEXT: affine.store %1, %0[%arg1] : memref<10xf32>
  // CHECK-NEXT: }
  // CHECK-NEXT: }
  // CHECK-NEXT: }


  return
}

func @invariant_affine_nested_if_else2() {
  %m = alloc() : memref<10xf32>
  %m2 = alloc() : memref<10xf32>
  %cf8 = constant 8.0 : f32
  affine.for %arg0 = 0 to 10 {
    affine.for %arg1 = 0 to 10 {
      affine.if (d0, d1) : (d1 - d0 >= 0) (%arg0, %arg0) {
          %cf9 = addf %cf8, %cf8 : f32
          %tload1 = affine.load %m[%arg0] : memref<10xf32>
          affine.if (d0, d1) : (d1 - d0 >= 0) (%arg0, %arg0) {
            affine.store %cf9, %m2[%arg0] : memref<10xf32>
          } else {
            %tload2 = affine.load %m[%arg0] : memref<10xf32>
          }
      }
    }
  }

  // CHECK: %0 = alloc() : memref<10xf32>
  // CHECK-NEXT: %1 = alloc() : memref<10xf32>
  // CHECK-NEXT: %cst = constant 8.000000e+00 : f32
  // CHECK-NEXT: affine.for %arg0 = 0 to 10 {
  // CHECK-NEXT: affine.if #set0(%arg0, %arg0) {
  // CHECK-NEXT: %2 = addf %cst, %cst : f32
  // CHECK-NEXT: %3 = affine.load %0[%arg0] : memref<10xf32>
  // CHECK-NEXT: affine.if #set0(%arg0, %arg0) {
  // CHECK-NEXT: affine.store %2, %1[%arg0] : memref<10xf32>
  // CHECK-NEXT: } else {
  // CHECK-NEXT: %4 = affine.load %0[%arg0] : memref<10xf32>
  // CHECK-NEXT: }
  // CHECK-NEXT: }


  return
}


func @invariant_affine_nested_if2() {
  %m = alloc() : memref<10xf32>
  %cf8 = constant 8.0 : f32
  affine.for %arg0 = 0 to 10 {
    affine.for %arg1 = 0 to 10 {
      affine.if (d0, d1) : (d1 - d0 >= 0) (%arg0, %arg0) {
          %cf9 = addf %cf8, %cf8 : f32
          %v1 = affine.load %m[%arg0] : memref<10xf32>
          affine.if (d0, d1) : (d1 - d0 >= 0) (%arg0, %arg0) {
            %v2 = affine.load %m[%arg0] : memref<10xf32>
          }
      }
    }
  }

  // CHECK: %0 = alloc() : memref<10xf32>
  // CHECK-NEXT: %cst = constant 8.000000e+00 : f32
  // CHECK-NEXT: affine.for %arg0 = 0 to 10 {
  // CHECK-NEXT: affine.if #set0(%arg0, %arg0) {
  // CHECK-NEXT: %1 = addf %cst, %cst : f32
  // CHECK-NEXT: %2 = affine.load %0[%arg0] : memref<10xf32>
  // CHECK-NEXT: affine.if #set0(%arg0, %arg0) {
  // CHECK-NEXT: %3 = affine.load %0[%arg0] : memref<10xf32>
  // CHECK-NEXT: }
  // CHECK-NEXT: }


  return
}

func @invariant_affine_for_inside_affine_if() {
  %m = alloc() : memref<10xf32>
  %cf8 = constant 8.0 : f32
  affine.for %arg0 = 0 to 10 {
    affine.for %arg1 = 0 to 10 {
      affine.if (d0, d1) : (d1 - d0 >= 0) (%arg0, %arg0) {
          %cf9 = addf %cf8, %cf8 : f32
          affine.store %cf9, %m[%arg0] : memref<10xf32>
          affine.for %arg2 = 0 to 10 {
            affine.store %cf9, %m[%arg2] : memref<10xf32>
          }
      }
    }
  }

  // CHECK: %0 = alloc() : memref<10xf32>
  // CHECK-NEXT: %cst = constant 8.000000e+00 : f32
  // CHECK-NEXT: affine.for %arg0 = 0 to 10 {
  // CHECK-NEXT: affine.for %arg1 = 0 to 10 {
  // CHECK-NEXT: affine.if #set0(%arg0, %arg0) {
  // CHECK-NEXT: %1 = addf %cst, %cst : f32
  // CHECK-NEXT: affine.store %1, %0[%arg0] : memref<10xf32>
  // CHECK-NEXT: affine.for %arg2 = 0 to 10 {
  // CHECK-NEXT: affine.store %1, %0[%arg2] : memref<10xf32>
  // CHECK-NEXT: }
  // CHECK-NEXT: }
  // CHECK-NEXT: }


  return
}


func @invariant_constant_and_load() {
  %m = alloc() : memref<100xf32>
  %m2 = alloc() : memref<100xf32>
  affine.for %arg0 = 0 to 5 {
    %c0 = constant 0 : index
    %v = affine.load %m2[%c0] : memref<100xf32>
    affine.store %v, %m[%arg0] : memref<100xf32>
  }

  // CHECK: %0 = alloc() : memref<100xf32>
  // CHECK-NEXT: %1 = alloc() : memref<100xf32>
  // CHECK-NEXT: %c0 = constant 0 : index
  // CHECK-NEXT: %2 = affine.load %1[%c0] : memref<100xf32>
  // CHECK-NEXT: affine.for %arg0 = 0 to 5 {
  // CHECK-NEXT:  affine.store %2, %0[%arg0] : memref<100xf32>


  return
}


func @nested_load_store_same_memref() {
  %m = alloc() : memref<10xf32>
  %cst = constant 8.0 : f32
  %c0 = constant 0 : index
   affine.for %arg0 = 0 to 10 {
    %v0 = affine.load %m[%c0] : memref<10xf32>
    affine.for %arg1 = 0 to 10 {
      affine.store %cst, %m[%arg1] : memref<10xf32>
    }
  }

  // CHECK: %0 = alloc() : memref<10xf32>
  // CHECK-NEXT: %cst = constant 8.000000e+00 : f32
  // CHECK-NEXT: %c0 = constant 0 : index
  // CHECK-NEXT: affine.for %arg0 = 0 to 10 {
  // CHECK-NEXT:  %1 = affine.load %0[%c0] : memref<10xf32>
  // CHECK-NEXT:   affine.for %arg1 = 0 to 10 {
  // CHECK-NEXT:    affine.store %cst, %0[%arg1] : memref<10xf32>


  return
}


func @nested_load_store_same_memref2() {
  %m = alloc() : memref<10xf32>
  %cst = constant 8.0 : f32
  %c0 = constant 0 : index
   affine.for %arg0 = 0 to 10 {
     affine.store %cst, %m[%c0] : memref<10xf32>
      affine.for %arg1 = 0 to 10 {
        %v0 = affine.load %m[%arg0] : memref<10xf32>
    }
  }

  // CHECK: %0 = alloc() : memref<10xf32>
  // CHECK-NEXT: %cst = constant 8.000000e+00 : f32
  // CHECK-NEXT: %c0 = constant 0 : index
  // CHECK-NEXT: affine.for %arg0 = 0 to 10 {
  // CHECK-NEXT:   affine.store %cst, %0[%c0] : memref<10xf32>
  // CHECK-NEXT:   %1 = affine.load %0[%arg0] : memref<10xf32>


  return
}
