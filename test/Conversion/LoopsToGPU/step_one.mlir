// RUN: mlir-opt -convert-loops-to-gpu -gpu-block-dims=1 -gpu-thread-dims=1 %s | FileCheck --check-prefix=CHECK-11 %s
// RUN: mlir-opt -convert-loops-to-gpu -gpu-block-dims=2 -gpu-thread-dims=2 %s | FileCheck --check-prefix=CHECK-22 %s

// CHECK-11-LABEL: @step_1
// CHECK-22-LABEL: @step_1
func @step_1(%A : memref<?x?x?x?xf32>, %B : memref<?x?x?x?xf32>) {
  // Bounds of the loop, its range and step.
  // CHECK-11-NEXT: %{{.*}} = constant 0 : index
  // CHECK-11-NEXT: %{{.*}} = constant 42 : index
  // CHECK-11-NEXT: %{{.*}} = subi %{{.*}}, %{{.*}} : index
  // CHECK-11-NEXT: %{{.*}} = constant 1 : index
  //
  // CHECK-22-NEXT: %{{.*}} = constant 0 : index
  // CHECK-22-NEXT: %{{.*}} = constant 42 : index
  // CHECK-22-NEXT: %{{.*}} = subi %{{.*}}, %{{.*}} : index
  // CHECK-22-NEXT: %{{.*}} = constant 1 : index
  affine.for %i = 0 to 42 {

    // Bounds of the loop, its range and step.
    // CHECK-11-NEXT: %{{.*}} = constant 0 : index
    // CHECK-11-NEXT: %{{.*}} = constant 10 : index
    // CHECK-11-NEXT: %{{.*}} = subi %{{.*}}, %{{.*}} : index
    // CHECK-11-NEXT: %{{.*}} = constant 1 : index
    //
    // CHECK-22-NEXT: %{{.*}} = constant 0 : index
    // CHECK-22-NEXT: %{{.*}} = constant 10 : index
    // CHECK-22-NEXT: %{{.*}} = subi %{{.*}}, %{{.*}} : index
    // CHECK-22-NEXT: %{{.*}} = constant 1 : index
    affine.for %j = 0 to 10 {
    // CHECK-11: gpu.launch
    // CHECK-11-SAME: blocks
    // CHECK-11-SAME: threads
    // CHECK-11-SAME: args

      // Remapping of the loop induction variables.
      // CHECK-11:        %[[i:.*]] = addi %{{.*}}, %{{.*}} : index
      // CHECK-11-NEXT:   %[[j:.*]] = addi %{{.*}}, %{{.*}} : index

      // This loop is not converted if mapping to 1, 1 dimensions.
      // CHECK-11-NEXT: affine.for %[[ii:.*]] = 2 to 16
      //
      // Bounds of the loop, its range and step.
      // CHECK-22-NEXT: %{{.*}} = constant 2 : index
      // CHECK-22-NEXT: %{{.*}} = constant 16 : index
      // CHECK-22-NEXT: %{{.*}} = subi %{{.*}}, %{{.*}} : index
      // CHECK-22-NEXT: %{{.*}} = constant 1 : index
      affine.for %ii = 2 to 16 {
        // This loop is not converted if mapping to 1, 1 dimensions.
        // CHECK-11-NEXT: affine.for %[[jj:.*]] = 5 to 17
        //
        // Bounds of the loop, its range and step.
        // CHECK-22-NEXT: %{{.*}} = constant 5 : index
        // CHECK-22-NEXT: %{{.*}} = constant 17 : index
        // CHECK-22-NEXT: %{{.*}} = subi %{{.*}}, %{{.*}} : index
        // CHECK-22-NEXT: %{{.*}} = constant 1 : index
        affine.for %jj = 5 to 17 {
        // CHECK-22: gpu.launch
        // CHECK-22-SAME: blocks
        // CHECK-22-SAME: threads
        // CHECK-22-SAME: args

          // Remapping of the loop induction variables in the last mapped loop.
          // CHECK-22:        %[[i:.*]] = addi %{{.*}}, %{{.*}} : index
          // CHECK-22-NEXT:   %[[j:.*]] = addi %{{.*}}, %{{.*}} : index
          // CHECK-22-NEXT:   %[[ii:.*]] = addi %{{.*}}, %{{.*}} : index
          // CHECK-22-NEXT:   %[[jj:.*]] = addi %{{.*}}, %{{.*}} : index

          // Using remapped values instead of loop iterators.
          // CHECK-11:        {{.*}} = load %{{.*}}[%[[i]], %[[j]], %[[ii]], %[[jj]]] : memref<?x?x?x?xf32>
          // CHECK-22:        {{.*}} = load %{{.*}}[%[[i]], %[[j]], %[[ii]], %[[jj]]] : memref<?x?x?x?xf32>
          %0 = load %A[%i, %j, %ii, %jj] : memref<?x?x?x?xf32>
          // CHECK-11-NEXT:   store {{.*}}, %{{.*}}[%[[i]], %[[j]], %[[ii]], %[[jj]]] : memref<?x?x?x?xf32>
          // CHECK-22-NEXT:   store {{.*}}, %{{.*}}[%[[i]], %[[j]], %[[ii]], %[[jj]]] : memref<?x?x?x?xf32>
          store %0, %B[%i, %j, %ii, %jj] : memref<?x?x?x?xf32>

          // CHECK-11: gpu.return
          // CHECK-22: gpu.return
        }
      }
    }
  }
  return
}

