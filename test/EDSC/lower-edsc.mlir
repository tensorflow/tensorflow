// RUN: mlir-opt -lower-edsc-test %s | FileCheck %s

func @t1(%lhs: memref<3x4x5x6xf32>, %rhs: memref<3x4x5x6xf32>, %result: memref<3x4x5x6xf32>) -> () { return }
func @t2(%lhs: memref<3x4xf32>, %rhs: memref<3x4xf32>, %result: memref<3x4xf32>) -> () { return }

func @fn() {
  "print"() {op: "x.add", fn: @t1: (memref<3x4x5x6xf32>, memref<3x4x5x6xf32>, memref<3x4x5x6xf32>) -> ()} : () -> ()
  "print"() {op: "x.add", fn: @t2: (memref<3x4xf32>, memref<3x4xf32>, memref<3x4xf32>) -> ()} : () -> ()
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
// CHECK: block {
// CHECK-NEXT:   for(idx($21)=$23 to $19 step $24) {
// CHECK-NEXT:     for(idx($22)=$23 to $20 step $24) {
// CHECK-NEXT:       lhs($25) = store( ... );
// CHECK-NEXT:     };
// CHECK-NEXT:   }
// CHECK-NEXT: }
