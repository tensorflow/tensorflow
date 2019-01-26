// RUN: mlir-opt %s -memref-dataflow-opt -verify | FileCheck %s

// CHECK-LABEL: func @simple_store_load() {
func @simple_store_load() {
  %cf7 = constant 7.0 : f32
  %m = alloc() : memref<10xf32>
  for %i0 = 0 to 10 {
    store %cf7, %m[%i0] : memref<10xf32>
    %v0 = load %m[%i0] : memref<10xf32>
    %v1 = addf %v0, %v0 : f32
  }
  return
// CHECK:       %cst = constant 7.000000e+00 : f32
// CHECK-NEXT:  for %i0 = 0 to 10 {
// CHECK-NEXT:    %0 = addf %cst, %cst : f32
// CHECK-NEXT:  }
// CHECK-NEXT:  return
}

// CHECK-LABEL: func @multi_store_load() {
func @multi_store_load() {
  %c0 = constant 0 : index
  %cf7 = constant 7.0 : f32
  %cf8 = constant 8.0 : f32
  %cf9 = constant 9.0 : f32
  %m = alloc() : memref<10xf32>
  for %i0 = 0 to 10 {
    store %cf7, %m[%i0] : memref<10xf32>
    %v0 = load %m[%i0] : memref<10xf32>
    %v1 = addf %v0, %v0 : f32
    store %cf8, %m[%i0] : memref<10xf32>
    store %cf9, %m[%i0] : memref<10xf32>
    %v2 = load %m[%i0] : memref<10xf32>
    %v3 = load %m[%i0] : memref<10xf32>
    %v4 = mulf %v2, %v3 : f32
  }
  return
// CHECK:       %c0 = constant 0 : index
// CHECK-NEXT:  %cst = constant 7.000000e+00 : f32
// CHECK-NEXT:  %cst_0 = constant 8.000000e+00 : f32
// CHECK-NEXT:  %cst_1 = constant 9.000000e+00 : f32
// CHECK-NEXT:  for %i0 = 0 to 10 {
// CHECK-NEXT:    %0 = addf %cst, %cst : f32
// CHECK-NEXT:    %1 = mulf %cst_1, %cst_1 : f32
// CHECK-NEXT:  }
// CHECK-NEXT:  return

}

// The store-load forwarding can see through affine apply's since it relies on
// dependence information.
// CHECK-LABEL: func @store_load_affine_apply
func @store_load_affine_apply() -> memref<10x10xf32> {
  %cf7 = constant 7.0 : f32
  %m = alloc() : memref<10x10xf32>
  for %i0 = 0 to 10 {
    for %i1 = 0 to 10 {
      %t0 = affine_apply (d0, d1) -> (d1 + 1)(%i0, %i1)
      %t1 = affine_apply (d0, d1) -> (d0)(%i0, %i1)
      %idx0 = affine_apply (d0, d1) -> (d1) (%t0, %t1)
      %idx1 = affine_apply (d0, d1) -> (d0 - 1) (%t0, %t1)
      store %cf7, %m[%idx0, %idx1] : memref<10x10xf32>
      // CHECK-NOT: load %{{[0-9]+}}
      %v0 = load %m[%i0, %i1] : memref<10x10xf32>
      %v1 = addf %v0, %v0 : f32
    }
  }
  // The memref and its stores won't be erased due to this memref return.
  return %m : memref<10x10xf32>
// CHECK:       %cst = constant 7.000000e+00 : f32
// CHECK-NEXT:  %0 = alloc() : memref<10x10xf32>
// CHECK-NEXT:  for %i0 = 0 to 10 {
// CHECK-NEXT:    for %i1 = 0 to 10 {
// CHECK-NEXT:      %1 = affine_apply #map0(%i0, %i1)
// CHECK-NEXT:      %2 = affine_apply #map1(%i0, %i1)
// CHECK-NEXT:      %3 = affine_apply #map2(%1, %2)
// CHECK-NEXT:      %4 = affine_apply #map3(%1, %2)
// CHECK-NEXT:      store %cst, %0[%3, %4] : memref<10x10xf32>
// CHECK-NEXT:      %5 = addf %cst, %cst : f32
// CHECK-NEXT:    }
// CHECK-NEXT:  }
// CHECK-NEXT:  return %0 : memref<10x10xf32>
}

// CHECK-LABEL: func @store_load_nested
func @store_load_nested(%N : index) {
  %cf7 = constant 7.0 : f32
  %m = alloc() : memref<10xf32>
  for %i0 = 0 to 10 {
    store %cf7, %m[%i0] : memref<10xf32>
    for %i1 = 0 to %N {
      %v0 = load %m[%i0] : memref<10xf32>
      %v1 = addf %v0, %v0 : f32
    }
  }
  return
// CHECK:       %cst = constant 7.000000e+00 : f32
// CHECK-NEXT:  for %i0 = 0 to 10 {
// CHECK-NEXT:    for %i1 = 0 to %arg0 {
// CHECK-NEXT:      %0 = addf %cst, %cst : f32
// CHECK-NEXT:    }
// CHECK-NEXT:  }
// CHECK-NEXT:  return
}

// No forwarding happens here since either of the two stores could be the last
// writer; store/load forwarding will however be possible here once loop live
// out SSA scalars are available.
// CHECK-LABEL: func @multi_store_load_nested_no_fwd
func @multi_store_load_nested_no_fwd(%N : index) {
  %cf7 = constant 7.0 : f32
  %cf8 = constant 8.0 : f32
  %m = alloc() : memref<10xf32>
  for %i0 = 0 to 10 {
    store %cf7, %m[%i0] : memref<10xf32>
    for %i1 = 0 to %N {
      store %cf8, %m[%i1] : memref<10xf32>
    }
    for %i2 = 0 to %N {
      // CHECK: %{{[0-9]+}} = load %0[%i0] : memref<10xf32>
      %v0 = load %m[%i0] : memref<10xf32>
      %v1 = addf %v0, %v0 : f32
    }
  }
  return
}

// No forwarding happens here since both stores have a value going into
// the load.
// CHECK-LABEL: func @store_load_store_nested_no_fwd
func @store_load_store_nested_no_fwd(%N : index) {
  %cf7 = constant 7.0 : f32
  %cf9 = constant 9.0 : f32
  %m = alloc() : memref<10xf32>
  for %i0 = 0 to 10 {
    store %cf7, %m[%i0] : memref<10xf32>
    for %i1 = 0 to %N {
      // CHECK: %{{[0-9]+}} = load %0[%i0] : memref<10xf32>
      %v0 = load %m[%i0] : memref<10xf32>
      %v1 = addf %v0, %v0 : f32
      store %cf9, %m[%i0] : memref<10xf32>
    }
  }
  return
}

// Forwarding happens here since the last store postdominates all other stores
// and other forwarding criteria are satisfied.
// CHECK-LABEL: func @multi_store_load_nested_fwd
func @multi_store_load_nested_fwd(%N : index) {
  %cf7 = constant 7.0 : f32
  %cf8 = constant 8.0 : f32
  %cf9 = constant 9.0 : f32
  %cf10 = constant 10.0 : f32
  %m = alloc() : memref<10xf32>
  for %i0 = 0 to 10 {
    store %cf7, %m[%i0] : memref<10xf32>
    for %i1 = 0 to %N {
      store %cf8, %m[%i1] : memref<10xf32>
    }
    for %i2 = 0 to %N {
      store %cf9, %m[%i2] : memref<10xf32>
    }
    store %cf10, %m[%i0] : memref<10xf32>
    for %i3 = 0 to %N {
      // CHECK-NOT: %{{[0-9]+}} = load
      %v0 = load %m[%i0] : memref<10xf32>
      %v1 = addf %v0, %v0 : f32
    }
  }
  return
}

// There is no unique load location for the store to forward to.
// CHECK-LABEL: func @store_load_no_fwd
func @store_load_no_fwd() {
  %cf7 = constant 7.0 : f32
  %m = alloc() : memref<10xf32>
  for %i0 = 0 to 10 {
    store %cf7, %m[%i0] : memref<10xf32>
    for %i1 = 0 to 10 {
      for %i2 = 0 to 10 {
        // CHECK: load %{{[0-9]+}}
        %v0 = load %m[%i2] : memref<10xf32>
        %v1 = addf %v0, %v0 : f32
      }
    }
  }
  return
}

// Forwarding happens here as there is a one-to-one store-load correspondence.
// CHECK-LABEL: func @store_load_fwd
func @store_load_fwd() {
  %cf7 = constant 7.0 : f32
  %c0 = constant 0 : index
  %m = alloc() : memref<10xf32>
  store %cf7, %m[%c0] : memref<10xf32>
  for %i0 = 0 to 10 {
    for %i1 = 0 to 10 {
      for %i2 = 0 to 10 {
        // CHECK-NOT: load %{{[0-9]}}+
        %v0 = load %m[%c0] : memref<10xf32>
        %v1 = addf %v0, %v0 : f32
      }
    }
  }
  return
}

// Although there is a dependence from the second store to the load, it is
// satisfied by the outer surrounding loop, and does not prevent the first
// store to be forwarded to the load.
func @store_load_store_nested_fwd(%N : index) -> f32 {
  %cf7 = constant 7.0 : f32
  %cf9 = constant 9.0 : f32
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %m = alloc() : memref<10xf32>
  for %i0 = 0 to 10 {
    store %cf7, %m[%i0] : memref<10xf32>
    for %i1 = 0 to %N {
      %v0 = load %m[%i0] : memref<10xf32>
      %v1 = addf %v0, %v0 : f32
      %idx = affine_apply (d0) -> (d0 + 1) (%i0)
      store %cf9, %m[%idx] : memref<10xf32>
    }
  }
  // Due to this load, the memref isn't optimized away.
  %v3 = load %m[%c1] : memref<10xf32>
  return %v3 : f32
// CHECK:       %0 = alloc() : memref<10xf32>
// CHECK-NEXT:  for %i0 = 0 to 10 {
// CHECK-NEXT:    store %cst, %0[%i0] : memref<10xf32>
// CHECK-NEXT:    for %i1 = 0 to %arg0 {
// CHECK-NEXT:      %1 = addf %cst, %cst : f32
// CHECK-NEXT:      %2 = affine_apply #map4(%i0)
// CHECK-NEXT:      store %cst_0, %0[%2] : memref<10xf32>
// CHECK-NEXT:    }
// CHECK-NEXT:  }
// CHECK-NEXT:  %3 = load %0[%c1] : memref<10xf32>
// CHECK-NEXT:  return %3 : f32
}
