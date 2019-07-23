// RUN: mlir-opt %s -test-detect-parallel -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL: func @loop_nest_3d_outer_two_parallel
func @loop_nest_3d_outer_two_parallel(%N : index) {
  %0 = alloc() : memref<1024 x 1024 x vector<64xf32>>
  %1 = alloc() : memref<1024 x 1024 x vector<64xf32>>
  %2 = alloc() : memref<1024 x 1024 x vector<64xf32>>
  affine.for %i = 0 to %N {
    // expected-remark@-1 {{parallel loop}}
    affine.for %j = 0 to %N {
      // expected-remark@-1 {{parallel loop}}
      affine.for %k = 0 to %N {
        // expected-remark@-1 {{sequential loop}}
        %5 = affine.load %0[%i, %k] : memref<1024x1024xvector<64xf32>>
        %6 = affine.load %1[%k, %j] : memref<1024x1024xvector<64xf32>>
        %7 = affine.load %2[%i, %j] : memref<1024x1024xvector<64xf32>>
        %8 = mulf %5, %6 : vector<64xf32>
        %9 = addf %7, %8 : vector<64xf32>
        affine.store %9, %2[%i, %j] : memref<1024x1024xvector<64xf32>>
      }
    }
  }
  return
}

// -----

// CHECK-LABEL: unknown_op_conservative
func @unknown_op_conservative() {
  affine.for %i = 0 to 10 {
    // expected-remark@-1 {{sequential loop}}
    "unknown"() : () -> ()
  }
  return
}

// -----

// CHECK-LABEL: non_affine_load
func @non_affine_load() {
  %0 = alloc() : memref<100 x f32>
  affine.for %i = 0 to 100 {
    // expected-remark@-1 {{sequential loop}}
    load %0[%i] : memref<100 x f32>
  }
  return
}
