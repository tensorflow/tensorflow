// RUN: mlir-opt %s -convert-loop-op-to-gpu -gpu-num-workgroups=2,16 -gpu-workgroup-size=32,4 | FileCheck %s

#map0 = (d0) -> (d0 + 2)
#map1 = (d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)
module {
  func @fmul(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>, %arg2: memref<?x?xf32>) {
    %c1 = constant 1 : index
    %c0 = constant 0 : index
    %c2 = constant 2 : index
    %0 = dim %arg0, 0 : memref<?x?xf32>
    %1 = dim %arg0, 1 : memref<?x?xf32>
    // CHECK-LABEL: gpu.launch
    // CHECK:   loop.for
    // CHECK:     loop.for
    // CHECK:       loop.for
    // CHECK:         loop.for
    // CHECK:           load
    // CHECK:           load
    // CHECK:           load
    // CHECK:           mulf
    // CHECK:           store
    loop.for %arg3 = %c0 to %0 step %c2 {
      loop.for %arg4 = %c0 to %1 step %c2 {
        %2 = affine.apply #map0(%arg3)
        %3 = affine.apply #map0(%arg4)
        %4 = linalg.subview %arg0[%arg3, %2, %c1, %arg4, %3, %c1] : memref<?x?xf32>
        %5 = affine.apply #map0(%arg3)
        %6 = affine.apply #map0(%arg4)
        %7 = linalg.subview %arg1[%arg3, %5, %c1, %arg4, %6, %c1] : memref<?x?xf32>
        %8 = affine.apply #map0(%arg3)
        %9 = affine.apply #map0(%arg4)
        %10 = linalg.subview %arg2[%arg3, %8, %c1, %arg4, %9, %c1] : memref<?x?xf32>
        %11 = dim %4, 0 : memref<?x?xf32, #map1>
        %12 = dim %4, 1 : memref<?x?xf32, #map1>
        loop.for %arg5 = %c0 to %11 step %c1 {
          loop.for %arg6 = %c0 to %12 step %c1 {
            %13 = load %4[%arg5, %arg6] : memref<?x?xf32, #map1>
            %14 = load %7[%arg5, %arg6] : memref<?x?xf32, #map1>
            %15 = load %10[%arg5, %arg6] : memref<?x?xf32, #map1>
            %16 = mulf %13, %14 : f32
            store %16, %10[%arg5, %arg6] : memref<?x?xf32, #map1>
          }
        }
      }
    }
    return
  }
}
