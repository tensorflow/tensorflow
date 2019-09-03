// RUN: mlir-opt %s -split-input-file -affine-data-copy-generate -affine-data-copy-generate-dma=false -affine-data-copy-generate-fast-mem-space=0 -affine-data-copy-generate-skip-non-unit-stride-loops | FileCheck %s
// Small buffer size to trigger fine copies.
// RUN: mlir-opt %s -affine-data-copy-generate -affine-data-copy-generate-dma=false -affine-data-copy-generate-fast-mem-space=0 -affine-data-copy-generate-fast-mem-capacity=1 | FileCheck --check-prefix=CHECK-SMALL %s

// -copy-skip-non-stride-loops forces the copies to be placed right inside the
// tile space loops, avoiding the sensitivity of copy placement depth to memory
// footprint -- so that one could write a definite test case and not have to
// update it each time something related to the cost functions change.

#map0 = (d0) -> (d0)
#map1 = (d0) -> (d0 + 128)

// Map used to index the original memref while copying.
// CHECK-DAG: [[MEM_IDX_MAP:map[0-9]+]] = (d0, d1) -> (d0 + d1)
// Map used to index the buffer while computing.
// CHECK-DAG: [[BUF_IDX_MAP:map[0-9]+]] = (d0, d1, d2, d3) -> (-d0 + d2, -d1 + d3)

// CHECK-LABEL: func @matmul
func @matmul(%A: memref<4096x4096xf32>, %B: memref<4096x4096xf32>, %C: memref<4096x4096xf32>) -> memref<4096x4096xf32> {
  affine.for %i = 0 to 4096 step 128 {
    affine.for %j = 0 to 4096 step 128 {
      affine.for %k = 0 to 4096 step 128 {
        affine.for %ii = #map0(%i) to #map1(%i) {
          affine.for %jj = #map0(%j) to #map1(%j) {
            affine.for %kk = #map0(%k) to #map1(%k) {
              %5 = affine.load %A[%ii, %kk] : memref<4096x4096xf32>
              %6 = affine.load %B[%kk, %jj] : memref<4096x4096xf32>
              %7 = affine.load %C[%ii, %jj] : memref<4096x4096xf32>
              %8 = mulf %5, %6 : f32
              %9 = addf %7, %8 : f32
              affine.store %9, %C[%ii, %jj] : memref<4096x4096xf32>
            }
          }
        }
      }
    }
  }
  return %C : memref<4096x4096xf32>
}

// Buffers of size 128x128 get created here for all three matrices.

// CHECK: affine.for %{{.*}} = 0 to 4096 step 128 {
// CHECK:   affine.for %{{.*}} = 0 to 4096 step 128 {
// CHECK:     [[BUFC:%[0-9]+]] = alloc() : memref<128x128xf32>

// The result matrix's copy gets hoisted out.
// Result matrix copy-in.
// CHECK:     affine.for %{{.*}} = 0 to 128 {
// CHECK:       %{{.*}} = affine.apply #[[MEM_IDX_MAP]](%{{.*}}, %{{.*}})
// CHECK:       affine.for %{{.*}} = 0 to 128 {
// CHECK:         %{{.*}} = affine.apply #[[MEM_IDX_MAP]](%{{.*}}, %{{.*}})
// CHECK:         %{{.*}} = affine.load %{{.*}}[%{{.*}}, %{{.*}}] : memref<4096x4096xf32>
// CHECK:         affine.store %{{.*}}, [[BUFC]][%{{.*}}, %{{.*}}] : memref<128x128xf32>
// CHECK:       }
// CHECK:     }

// LHS matrix copy-in.
// CHECK:     affine.for %{{.*}} = 0 to 4096 step 128 {
// CHECK:      [[BUFA:%[0-9]+]] = alloc() : memref<128x128xf32>
// CHECK:       affine.for %{{.*}} = 0 to 128 {
// CHECK:         %{{.*}} = affine.apply #[[MEM_IDX_MAP]](%{{.*}}, %{{.*}})
// CHECK:         affine.for %{{.*}} = 0 to 128 {
// CHECK:           %{{.*}} = affine.apply #[[MEM_IDX_MAP]](%{{.*}}, %{{.*}})
// CHECK:           %{{.*}} = affine.load %{{.*}}[%{{.*}}, %{{.*}}] : memref<4096x4096xf32>
// CHECK:           affine.store %{{.*}}, [[BUFA]][%{{.*}}, %{{.*}}] : memref<128x128xf32>
// CHECK:         }
// CHECK:       }

// RHS matrix copy-in.
// CHECK:       [[BUFB:%[0-9]+]] = alloc() : memref<128x128xf32>
// CHECK:       affine.for %{{.*}} = 0 to 128 {
// CHECK:         %{{.*}} = affine.apply #[[MEM_IDX_MAP]](%{{.*}}, %{{.*}})
// CHECK:         affine.for %{{.*}} = 0 to 128 {
// CHECK:           %{{.*}} = affine.apply #[[MEM_IDX_MAP]](%{{.*}}, %{{.*}})
// CHECK:           %{{.*}} = affine.load %{{.*}}[%{{.*}}, %{{.*}}] : memref<4096x4096xf32>
// CHECK:           affine.store %{{.*}}, [[BUFB]][%{{.*}}, %{{.*}}] : memref<128x128xf32>
// CHECK:         }
// CHECK:       }

// Computation on the fast buffers.
// CHECK:       affine.for %{{.*}} = #map7(%{{.*}}) to #map8(%{{.*}}) {
// CHECK:         affine.for %{{.*}} = #map7(%{{.*}}) to #map8(%{{.*}}) {
// CHECK:           affine.for %{{.*}} = #map7(%{{.*}}) to #map8(%{{.*}}) {
// CHECK:             %{{.*}} = affine.load [[BUFA]][-%{{.*}} + %{{.*}}, -%{{.*}} + %{{.*}}] : memref<128x128xf32>
// CHECK:             %{{.*}} = affine.load [[BUFB]][-%{{.*}} + %{{.*}}, -%{{.*}} + %{{.*}}] : memref<128x128xf32>
// CHECK:             %{{.*}} = affine.load [[BUFC]][-%{{.*}} + %{{.*}}, -%{{.*}} + %{{.*}}] : memref<128x128xf32>
// CHECK:             %{{.*}} = mulf %{{.*}}, %{{.*}} : f32
// CHECK:             %{{.*}} = addf %{{.*}}, %{{.*}} : f32
// CHECK:             affine.store %{{.*}}, [[BUFC]][-%{{.*}} + %{{.*}}, -%{{.*}} + %{{.*}}] : memref<128x128xf32>
// CHECK:           }
// CHECK:         }
// CHECK:       }
// CHECK:       dealloc [[BUFB]] : memref<128x128xf32>
// CHECK:       dealloc [[BUFA]] : memref<128x128xf32>
// CHECK:     }
// CHECK:     %{{.*}} = affine.apply #map0(%{{.*}}, %{{.*}})
// CHECK:     %{{.*}} = affine.apply #map1(%{{.*}}, %{{.*}})

// Result matrix copy out.
// CHECK:     affine.for %{{.*}} = 0 to 128 {
// CHECK:       %{{.*}} = affine.apply #[[MEM_IDX_MAP]](%{{.*}}, %{{.*}})
// CHECK:       affine.for %{{.*}} = 0 to 128 {
// CHECK:         %{{.*}} = affine.apply #[[MEM_IDX_MAP]](%{{.*}}, %{{.*}})
// CHECK:         [[BUFA]] = affine.load [[BUFC]][%{{.*}}, %{{.*}}] : memref<128x128xf32>
// CHECK:         store [[BUFA]], %{{.*}}[%{{.*}}, %{{.*}}] : memref<4096x4096xf32>
// CHECK:       }
// CHECK:     }
// CHECK:     dealloc [[BUFC]] : memref<128x128xf32>
// CHECK:   }
// CHECK: }

//
// This test case will lead to single element buffers. These are eventually
// expected to be turned into registers via alloca and mem2reg.
//
// CHECK-SMALL: func @foo
func @foo(%arg0: memref<1024x1024xf32>, %arg1: memref<1024x1024xf32>, %arg2: memref<1024x1024xf32>) -> memref<1024x1024xf32> {
  affine.for %i = 0 to 1024 {
    affine.for %j = 0 to 1024 {
      affine.for %k = 0 to 1024 {
        %6 = affine.load %arg1[%k, %j] : memref<1024x1024xf32>
        %7 = affine.load %arg2[%i, %j] : memref<1024x1024xf32>
        %9 = addf %6, %7 : f32
        affine.store %9, %arg2[%i, %j] : memref<1024x1024xf32>
      }
    }
  }
  return %arg2 : memref<1024x1024xf32>
}
// CHECK-SMALL: affine.for %arg{{.*}} = 0 to 1024 {
// CHECK-SMALL:   affine.for %arg{{.*}} = 0 to 1024 {
// CHECK-SMALL:     %{{.*}} = affine.apply #map{{.*}}(%arg{{.*}}, %arg{{.*}})
// CHECK-SMALL:     %{{.*}} = affine.apply #map{{.*}}(%arg{{.*}}, %arg{{.*}})
// CHECK-SMALL:     %{{.*}} = alloc() : memref<1x1xf32>
// CHECK-SMALL:     %{{.*}} = affine.apply #map{{.*}}(%arg{{.*}}, %c0{{.*}})
// CHECK-SMALL:     %{{.*}} = affine.apply #map{{.*}}(%arg{{.*}}, %c0{{.*}})
// CHECK-SMALL:     %{{.*}} = affine.load %arg{{.*}}[%{{.*}}, %{{.*}}] : memref<1024x1024xf32>
// CHECK-SMALL:     affine.store %{{.*}}, %{{.*}}[%c0{{.*}}, %c0{{.*}}] : memref<1x1xf32>
// CHECK-SMALL:     affine.for %arg{{.*}} = 0 to 1024 {
// CHECK-SMALL:       %{{.*}} = affine.apply #map{{.*}}(%arg{{.*}}, %arg{{.*}})
// CHECK-SMALL:       %{{.*}} = affine.apply #map{{.*}}(%arg{{.*}}, %arg{{.*}})
// CHECK-SMALL:       %{{.*}} = alloc() : memref<1x1xf32>
// CHECK-SMALL:       %{{.*}} = affine.apply #map{{.*}}(%arg{{.*}}, %c0{{.*}})
// CHECK-SMALL:       %{{.*}} = affine.apply #map{{.*}}(%arg{{.*}}, %c0{{.*}})
// CHECK-SMALL:       %{{.*}} = affine.load %arg{{.*}}[%{{.*}}, %{{.*}}] : memref<1024x1024xf32>
// CHECK-SMALL:       affine.store %{{.*}}, %{{.*}}[%c0{{.*}}, %c0{{.*}}] : memref<1x1xf32>
// CHECK-SMALL:       %{{.*}} = affine.load %{{.*}}[0, 0] : memref<1x1xf32>
// CHECK-SMALL:       %{{.*}} = affine.load %{{.*}}[0, 0] : memref<1x1xf32>
// CHECK-SMALL:       %{{.*}} = addf %{{.*}}, %{{.*}} : f32
// CHECK-SMALL:       affine.store %{{.*}}, %{{.*}}[0, 0] : memref<1x1xf32>
// CHECK-SMALL:       dealloc %{{.*}} : memref<1x1xf32>
// CHECK-SMALL:     }
// CHECK-SMALL:     %{{.*}} = affine.apply #map{{.*}}(%arg{{.*}}, %arg{{.*}})
// CHECK-SMALL:     %{{.*}} = affine.apply #map{{.*}}(%arg{{.*}}, %arg{{.*}})
// CHECK-SMALL:     %{{.*}} = affine.apply #map{{.*}}(%arg{{.*}}, %c0{{.*}})
// CHECK-SMALL:     %{{.*}} = affine.apply #map{{.*}}(%arg{{.*}}, %c0{{.*}})
// CHECK-SMALL:     %{{.*}} = affine.load %{{.*}}[%c0{{.*}}, %c0{{.*}}] : memref<1x1xf32>
// CHECK-SMALL:     affine.store %{{.*}}, %arg{{.*}}[%{{.*}}, %{{.*}}] : memref<1024x1024xf32>
// CHECK-SMALL:     dealloc %{{.*}} : memref<1x1xf32>
// CHECK-SMALL:   }
// CHECK-SMALL: }
// CHECK-SMALL: return
