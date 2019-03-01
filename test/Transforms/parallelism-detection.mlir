// RUN: mlir-opt %s -detect-parallel | mlir-opt | FileCheck %s

func @loop_nest_3d_outer_two_parallel(%N : index) {
  %0 = alloc() : memref<1024 x 1024 x vector<64xf32>>
  %1 = alloc() : memref<1024 x 1024 x vector<64xf32>>
  %2 = alloc() : memref<1024 x 1024 x vector<64xf32>>
  for %i = 0 to %N {
    for %j = 0 to %N {
      for %k = 0 to %N {
        %5 = load %0[%i, %k] : memref<1024x1024xvector<64xf32>>
        %6 = load %1[%k, %j] : memref<1024x1024xvector<64xf32>>
        %7 = load %2[%i, %j] : memref<1024x1024xvector<64xf32>>
        %8 = mulf %5, %6 : vector<64xf32>
        %9 = addf %7, %8 : vector<64xf32>
        store %9, %2[%i, %j] : memref<1024x1024xvector<64xf32>>
      }  // CHECK: } {parallel: false}
    } // CHECK: } {parallel: true}
  }  // CHECK: } {parallel: true}
  return
}
