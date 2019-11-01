// RUN: mlir-opt -convert-loop-op-to-gpu -gpu-num-workgroups=4,2,2 -gpu-workgroup-size=32,2,2 %s | FileCheck %s

module {
  func @imperfect_3D(%arg0 : memref<?x?x?x?xf32>, %arg1 : memref<?x?x?x?xf32>, %arg2 : memref<?x?x?x?xf32>, %arg3 : memref<?x?x?x?xf32>, %t1 : index, %t2 : index, %t3 : index, %t4 : index, %step1 : index, %step2 : index, %step3 : index, %step4 : index) {
    %0 = dim %arg0, 0 : memref<?x?x?x?xf32>
    %1 = dim %arg0, 1 : memref<?x?x?x?xf32>
    %2 = dim %arg0, 2 : memref<?x?x?x?xf32>
    %3 = dim %arg0, 3 : memref<?x?x?x?xf32>
    %c0 = constant 0 : index
    // CHECK: gpu.launch
    // CHECK:   loop.for
    // CHECK:     loop.for
    // CHECK:       loop.for
    // CHECK:         alloc
    // CHECK:         loop.for
    // CHECK:           loop.for
    // CHECK:             loop.for
    // CHECK:               loop.for
    // CHECK:                 load
    // CHECK:                 load
    // CHECK:                 addf
    // CHECK:                 store
    // CHECK:         loop.for
    // CHECK:           loop.for
    // CHECK:             loop.for
    // CHECK:               loop.for
    // CHECK:                 load
    // CHECK:                 load
    // CHECK:                 mulf
    // CHECK:                 store
    // CHECK:         dealloc
    loop.for %iv1 = %c0 to %0 step %t1 {
      loop.for %iv2 = %c0 to %1 step %t2 {
        loop.for %iv3 = %c0 to %2 step %t3 {
          %6 = alloc(%t1, %t2, %t3, %3) : memref<?x?x?x?xf32>
          %ubcmp1 = cmpi "slt", %0, %t1 : index
          %ub1 = select %ubcmp1, %0, %t1 : index
          %ubcmp2 = cmpi "slt", %1, %t2 : index
          %ub2 = select %ubcmp2, %1, %t2 : index
          %ubcmp3 = cmpi "slt", %2, %t3 : index
          %ub3 = select %ubcmp3, %2, %t3 : index
          %ubcmp4 = cmpi "slt", %3, %t4 : index
          %ub4 = select %ubcmp3, %3, %t4 : index
          loop.for %iv5 = %iv1 to %ub1 step %step1 {
            loop.for %iv6 = %iv2 to %ub2 step %step2 {
              loop.for %iv7 = %iv3 to %ub3 step %step3 {
                loop.for %iv8 = %c0 to %3 step %step4 {
                  %7 = load %arg0[%iv5, %iv6, %iv7, %iv8] : memref<?x?x?x?xf32>
                  %8 = load %arg1[%iv5, %iv6, %iv7, %iv8] : memref<?x?x?x?xf32>
                  %9 = addf %7, %8 : f32
                  %10 = subi %iv5, %iv1 : index
                  %11 = divis %10, %step1 : index
                  %12 = subi %iv6, %iv2 : index
                  %13 = divis %12, %step2 : index
                  %14 = subi %iv7, %iv3 : index
                  %15 = divis %14, %step3 : index
                  store %9, %6[%11, %13, %15, %iv8] : memref<?x?x?x?xf32>
                }
              }
            }
          }
          loop.for %iv9 = %iv1 to %ub1 step %step1 {
            loop.for %iv10 = %iv2 to %ub2 step %step2 {
              loop.for %iv11 = %iv3 to %ub3 step %step3 {
                loop.for %iv12 = %c0 to %3 step %step4 {
                  %18 = subi %iv9, %iv1 : index
                  %19 = divis %18, %step1 : index
                  %20 = subi %iv10, %iv2 : index
                  %21 = divis %20, %step2 : index
                  %22 = subi %iv11, %iv3 : index
                  %23 = divis %22, %step3 : index
                  %26 = load %6[%19, %21, %23, %iv12] : memref<?x?x?x?xf32>
                  %27 = load %arg2[%iv9, %iv10, %iv12, %iv11] : memref<?x?x?x?xf32>
                  %28 = mulf %26, %27 : f32
                  store %28, %arg3[%iv9, %iv10, %iv11, %iv12] : memref<?x?x?x?xf32>
                }
              }
            }
          }
          dealloc %6 : memref<?x?x?x?xf32>
        }
      }
    }
    return
  }
}