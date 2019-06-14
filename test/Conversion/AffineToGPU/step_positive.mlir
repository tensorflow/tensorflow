// RUN: mlir-opt -convert-affine-to-gpu -gpu-block-dims=1 -gpu-thread-dims=1 %s | FileCheck %s

// CHECK-LABEL: @step_var
func @step_var(%A : memref<?x?xf32>, %B : memref<?x?xf32>) {
  // The loop range computation is performed by lowering the affine expression
  // floordiv(upper - lower, step).  The lowering of affine expressions has its
  // own test, here we only check the fact of division by step.
  // CHECK:  divis {{.*}}, %c4
  // CHECK:  %[[range_i:.*]] = select
  // CHECK:  divis {{.*}}, %c7
  // CHECK:  %[[range_j:.*]] = select

  // CHECK: gpu.launch
  // CHECK-SAME: blocks(%i0, %i1, %i2) in (%i6 = %[[range_i]], %i7 = %c1, %i8 = %c1)
  // CHECK-SAME: threads(%i3, %i4, %i5) in (%i9 = %[[range_j]], %i10 = %c1, %i11 = %c1)
  affine.for %i = 5 to 15 step 4 {
    affine.for %j = 3 to 19 step 7 {
      // Loop induction variable remapping:
      //     iv = thread(block)_id * step + lower_bound
      // CHECK:      %[[c4:.*]] = constant 4 : index
      // CHECK-NEXT: %[[prod_i:.*]] = muli %[[c4]], %i0 : index
      // CHECK-NEXT: %[[i:.*]] = addi %i14, %[[prod_i]] : index
      // CHECK-NEXT: %[[c7:.*]] = constant 7 : index
      // CHECK-NEXT: %[[prod_j:.*]] = muli %[[c7]], %i3 : index
      // CHECK-NEXT: %[[j:.*]] = addi %i15, %[[prod_j]] : index

      // CHECK:     {{.*}} = load %i12[%[[i]], %[[j]]] : memref<?x?xf32>
      %0 = load %A[%i, %j] : memref<?x?xf32>
      // CHECK:     store {{.*}}, %i13[%[[i]], %[[j]]] : memref<?x?xf32>
      store %0, %B[%i, %j] : memref<?x?xf32>
    }
  }
  return
}
