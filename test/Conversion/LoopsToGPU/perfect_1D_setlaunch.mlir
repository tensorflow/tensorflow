// RUN: mlir-opt -convert-loop-op-to-gpu -gpu-num-workgroups=2 -gpu-workgroup-size=32 %s | FileCheck %s

module {
  func @foo(%arg0: memref<?x?xf32>, %arg1 : memref<?x?xf32>, %arg2 : memref<?x?xf32>) {
    %0 = dim %arg0, 0 : memref<?x?xf32>
    %1 = dim %arg0, 1 : memref<?x?xf32>
    %c0 = constant 0 : index
    %c1 = constant 1 : index
    // CHECK: gpu.launch
    // CHECK:   loop.for
    // CHECK:     loop.for
    // CHECK:       load
    // CHECK:       load
    // CHECK:       add
    // CHECK:       store
    loop.for %iv1 = %c0 to %0 step %c1 {
      loop.for %iv2 = %c0 to %1 step %c1 {
         %12 = load %arg0[%iv1, %iv2] : memref<?x?xf32>
         %13 = load %arg1[%iv2, %iv1] : memref<?x?xf32>
         %14 = addf %12, %13 : f32
         store %12, %arg2[%iv1, %iv2] : memref<?x?xf32>
       }
     }
    return
  }
}