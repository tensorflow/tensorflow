// RUN: mlir-opt -lower-edsc-test %s | FileCheck %s

func @t1(%lhs: memref<3x4x5x6xf32>, %rhs: memref<3x4x5x6xf32>, %result: memref<3x4x5x6xf32>) -> () {
  return
}

func @fn() {
  "dump"() {fn: @t1: (memref<3x4x5x6xf32>, memref<3x4x5x6xf32>, memref<3x4x5x6xf32>) -> ()} : () -> ()
  return
}

// CHECK: block {
// CHECK-NEXT:   for(idx($8)=$12 to $4 step $13) {
// CHECK-NEXT:     for(idx($9)=$12 to $5 step $13) {
// CHECK-NEXT:       for(idx($10)=$12 to $6 step $13) {
// CHECK-NEXT:         for(idx($11)=$12 to $7 step $13) {
// CHECK-NEXT:           lhs($14) = store( ... );
// CHECK-NEXT:         };
// CHECK-NEXT:       };
// CHECK-NEXT:     };
// CHECK-NEXT:   }
// CHECK-NEXT: }