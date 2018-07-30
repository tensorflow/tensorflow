// RUN: %S/../../mlir-opt %s -o - -unroll-innermost-loops | FileCheck %s

// CHECK-LABEL: mlfunc @loops() {
mlfunc @loops() {
  // CHECK: for %i0 = 1 to 100 step 2 {
  for %i = 1 to 100 step 2 {
    // CHECK: "custom"(){value: 1} : () -> ()
    // CHECK-NEXT: "custom"(){value: 1} : () -> ()
    // CHECK-NEXT: "custom"(){value: 1} : () -> ()
    // CHECK-NEXT: "custom"(){value: 1} : () -> ()
    for %j = 1 to 4 {
      "custom"(){value: 1} : () -> f32
    }
  }       // CHECK:  }
  return  // CHECK:  return
}         // CHECK }
