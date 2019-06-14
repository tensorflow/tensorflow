// RUN: mlir-opt -convert-affine-to-gpu -gpu-block-dims=1 -gpu-thread-dims=1 %s | FileCheck --check-prefix=CHECK-11 %s
// RUN: mlir-opt -convert-affine-to-gpu -gpu-block-dims=2 -gpu-thread-dims=2 %s | FileCheck --check-prefix=CHECK-22 %s

// CHECK-11-LABEL: @step_1
// CHECK-22-LABEL: @step_1
func @step_1(%A : memref<?x?x?x?xf32>, %B : memref<?x?x?x?xf32>) {
  // Bounds of the loop and its range.
  // CHECK-11-NEXT: %c0 = constant 0 : index
  // CHECK-11-NEXT: %c42 = constant 42 : index
  // CHECK-11-NEXT: %0 = subi %c42, %c0 : index
  //
  // CHECK-22-NEXT: %c0 = constant 0 : index
  // CHECK-22-NEXT: %c42 = constant 42 : index
  // CHECK-22-NEXT: %0 = subi %c42, %c0 : index
  affine.for %i = 0 to 42 {

    // Bounds of the loop and its range.
    // CHECK-11-NEXT: %c0_0 = constant 0 : index
    // CHECK-11-NEXT: %c10 = constant 10 : index
    // CHECK-11-NEXT: %1 = subi %c10, %c0_0 : index
    //
    // CHECK-22-NEXT: %c0_0 = constant 0 : index
    // CHECK-22-NEXT: %c10 = constant 10 : index
    // CHECK-22-NEXT: %1 = subi %c10, %c0_0 : index
    affine.for %j = 0 to 10 {
    // CHECK-11: gpu.launch
    // CHECK-11-SAME: blocks(%i0, %i1, %i2) in (%i6 = %0, %i7 = %c1, %i8 = %c1)
    // CHECK-11-SAME: threads(%i3, %i4, %i5) in (%i9 = %1, %i10 = %c1, %i11 = %c1)
    // CHECK-11-SAME: args(%i12 = %arg0, %i13 = %arg1, %i14 = %c0, %i15 = %c0_0)

      // Remapping of the loop induction variables.
      // CHECK-11:        %[[i:.*]] = addi %i14, %i0 : index
      // CHECK-11-NEXT:   %[[j:.*]] = addi %i15, %i3 : index

      // This loop is not converted if mapping to 1, 1 dimensions.
      // CHECK-11-NEXT: affine.for %[[ii:.*]] = 2 to 16
      //
      // Bounds of the loop and its range.
      // CHECK-22-NEXT: %c2 = constant 2 : index
      // CHECK-22-NEXT: %c16 = constant 16 : index
      // CHECK-22-NEXT: %2 = subi %c16, %c2 : index
      affine.for %ii = 2 to 16 {
        // This loop is not converted if mapping to 1, 1 dimensions.
        // CHECK-11-NEXT: affine.for %[[jj:.*]] = 5 to 17
        //
        // Bounds of the loop and its range.
        // CHECK-22-NEXT: %c5 = constant 5 : index
        // CHECK-22-NEXT: %c17 = constant 17 : index
        // CHECK-22-NEXT: %3 = subi %c17, %c5 : index
        affine.for %jj = 5 to 17 {
        // CHECK-22: gpu.launch
        // CHECK-22-SAME: blocks(%i0, %i1, %i2) in (%i6 = %0, %i7 = %1, %i8 = %c1)
        // CHECK-22-SAME: threads(%i3, %i4, %i5) in (%i9 = %2, %i10 = %3, %i11 = %c1)
        // CHECK-22-SAME: args(%i12 = %arg0, %i13 = %arg1, %i14 = %c0, %i15 = %c0_0, %i16 = %c2, %i17 = %c5)

          // Remapping of the loop induction variables in the last mapped loop.
          // CHECK-22:        %[[i:.*]] = addi %i14, %i0 : index
          // CHECK-22-NEXT:   %[[j:.*]] = addi %i15, %i1 : index
          // CHECK-22-NEXT:   %[[ii:.*]] = addi %i16, %i3 : index
          // CHECK-22-NEXT:   %[[jj:.*]] = addi %i17, %i4 : index

          // Using remapped values instead of loop iterators.
          // CHECK-11:        {{.*}} = load %i12[%[[i]], %[[j]], %[[ii]], %[[jj]]] : memref<?x?x?x?xf32>
          // CHECK-22:        {{.*}} = load %i12[%[[i]], %[[j]], %[[ii]], %[[jj]]] : memref<?x?x?x?xf32>
          %0 = load %A[%i, %j, %ii, %jj] : memref<?x?x?x?xf32>
          // CHECK-11-NEXT:   store {{.*}}, %i13[%[[i]], %[[j]], %[[ii]], %[[jj]]] : memref<?x?x?x?xf32>
          // CHECK-22-NEXT:   store {{.*}}, %i13[%[[i]], %[[j]], %[[ii]], %[[jj]]] : memref<?x?x?x?xf32>
          store %0, %B[%i, %j, %ii, %jj] : memref<?x?x?x?xf32>

          // CHECK-11: gpu.return
          // CHECK-22: gpu.return
        }
      }
    }
  }
  return
}

