// RUN: mlir-opt -convert-loops-to-gpu -gpu-block-dims=1 -gpu-thread-dims=1 %s | FileCheck %s

// CHECK-LABEL: @step_var
func @step_var(%A : memref<?x?xf32>, %B : memref<?x?xf32>) {
  // Check that we divide by step.
  // CHECK:  %[[range_i:.*]] = divis {{.*}}, %{{.*}}
  // CHECK:  %[[range_j:.*]] = divis {{.*}}, %{{.*}}

  // CHECK: gpu.launch
  // CHECK-SAME: blocks(%{{[^)]*}}, %{{[^)]*}}, %{{[^)]*}}) in (%{{[^)]*}} = %[[range_i]], %{{[^)]*}} = %{{[^)]*}}, %{{[^)]*}} = %{{[^)]*}})
  // CHECK-SAME: threads(%{{[^)]*}}, %{{[^)]*}}, %{{[^)]*}}) in (%{{[^)]*}} = %[[range_j]], %{{[^)]*}} = %{{[^)]*}}, %{{[^)]*}} = %{{[^)]*}})
  affine.for %i = 5 to 15 step 4 {
    affine.for %j = 3 to 19 step 7 {
      // Loop induction variable remapping:
      //     iv = thread(block)_id * step + lower_bound
      // CHECK:      %[[prod_i:.*]] = muli %{{.*}}, %{{.*}} : index
      // CHECK-NEXT: %[[i:.*]] = addi %{{.*}}, %[[prod_i]] : index
      // CHECK-NEXT: %[[prod_j:.*]] = muli %{{.*}}, %{{.*}} : index
      // CHECK-NEXT: %[[j:.*]] = addi %{{.*}}, %[[prod_j]] : index

      // CHECK:     {{.*}} = load %{{.*}}[%[[i]], %[[j]]] : memref<?x?xf32>
      %0 = load %A[%i, %j] : memref<?x?xf32>
      // CHECK:     store {{.*}}, %{{.*}}[%[[i]], %[[j]]] : memref<?x?xf32>
      store %0, %B[%i, %j] : memref<?x?xf32>
    }
  }
  return
}
