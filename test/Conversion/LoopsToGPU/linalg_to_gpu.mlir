// RUN: mlir-opt -convert-loops-to-gpu %s | FileCheck %s

// CHECK-LABEL: @foo
func @foo(%arg0: !linalg.buffer<?xf32>, %arg1 : index) {
  %c0 = constant 0 : index
  %c42 = constant 42 : index
  %c3 = constant 3 : index
  // CHECK:      subi %{{.*}}, %{{.*}} : index
  // CHECK-NEXT: %[[range_i:.*]] = divis {{.*}}, %{{.*}} : index
  loop.for %i0 = %c0 to %c42 step %c3 {
    // CHECK:      subi %{{.*}}, %{{.*}} : index
    // CHECK-NEXT: %[[range_j:.*]] = divis {{.*}}, %{{.*}} : index
    loop.for %i1 = %c3 to %c42 step %arg1 {
      // CHECK:      gpu.launch
      // CHECK-SAME: blocks
      // CHECK-SAME: threads
      // CHECK-SAME: args

      // Replacements of loop induction variables.  Take a product with the
      // step and add the lower bound.
      // CHECK: %[[prod_i:.*]] = muli %{{.*}}, %{{.*}} : index
      // CHECK: addi %{{.*}}, %[[prod_i]] : index
      // CHECK: %[[prod_j:.*]] = muli %{{.*}}, %{{.*}} : index
      // CHECK: addi %{{.*}}, %[[prod_j]] : index

      // CHECK: gpu.return
    }
  }
  return
}
