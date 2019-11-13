// RUN: mlir-opt %s -convert-loop-op-to-gpu -gpu-num-workgroups=2,16 -gpu-workgroup-size=32,4 | FileCheck %s

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
        %4 = std.subview %arg0[%arg3, %arg4][%c2, %c2][%c1, %c1] : memref<?x?xf32> to memref<?x?xf32, offset: ?, strides: [?, ?]>
        %7 = std.subview %arg1[%arg3, %arg4][%c2, %c2][%c1, %c1] : memref<?x?xf32> to memref<?x?xf32, offset: ?, strides: [?, ?]>
        %10 = std.subview %arg2[%arg3, %arg4][%c2, %c2][%c1, %c1] : memref<?x?xf32> to memref<?x?xf32, offset: ?, strides: [?, ?]>
        %11 = dim %4, 0 : memref<?x?xf32, offset: ?, strides: [?, ?]>
        %12 = dim %4, 1 : memref<?x?xf32, offset: ?, strides: [?, ?]>
        loop.for %arg5 = %c0 to %11 step %c1 {
          loop.for %arg6 = %c0 to %12 step %c1 {
            %13 = load %4[%arg5, %arg6] : memref<?x?xf32, offset: ?, strides: [?, ?]>
            %14 = load %7[%arg5, %arg6] : memref<?x?xf32, offset: ?, strides: [?, ?]>
            %15 = load %10[%arg5, %arg6] : memref<?x?xf32, offset: ?, strides: [?, ?]>
            %16 = mulf %13, %14 : f32
            store %16, %10[%arg5, %arg6] : memref<?x?xf32, offset: ?, strides: [?, ?]>
          }
        }
      }
    }
    return
  }
}
