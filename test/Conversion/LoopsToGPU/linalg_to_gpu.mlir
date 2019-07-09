// RUN: mlir-opt -convert-loops-to-gpu %s | FileCheck %s

// CHECK-LABEL: @foo
func @foo(%arg0: !linalg.buffer<?xf32>, %arg1 : index) {
  %c0 = constant 0 : index
  %c42 = constant 42 : index
  %c3 = constant 3 : index
  // CHECK:      subi %c42, %c0 : index
  // CHECK-NEXT: %[[range_i:.*]] = divis {{.*}}, %c3 : index
  linalg.for %i0 = %c0 to %c42 step %c3 {
    // CHECK:      subi %c42, %c3 : index
    // CHECK-NEXT: %[[range_j:.*]] = divis {{.*}}, %arg1 : index
    linalg.for %i1 = %c3 to %c42 step %arg1 {
      // CHECK:      gpu.launch
      // CHECK-SAME: blocks(%i0, %i1, %i2) in (%i6 = %[[range_i]], %i7 = %c1, %i8 = %c1)
      // CHECK-SAME: threads(%i3, %i4, %i5) in (%i9 = %[[range_j]], %i10 = %c1, %i11 = %c1)
      // CHECK-SAME: args(%i12 = %c0, %i13 = %c3, %i14 = %c3, %i15 = %arg1)

      // Replacements of loop induction variables.  Take a product with the
      // step and add the lower bound.
      // CHECK: %[[prod_i:.*]] = muli %i14, %i0 : index
      // CHECK: addi %i12, %[[prod_i]] : index
      // CHECK: %[[prod_j:.*]] = muli %i15, %i3 : index
      // CHECK: addi %i13, %[[prod_j]] : index

      // CHECK: gpu.return
    }
  }
  return
}
