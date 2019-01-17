// RUN: mlir-opt -lower-edsc-test %s | FileCheck %s

func @t1(%lhs: memref<3x4x5x6xf32>, %rhs: memref<3x4x5x6xf32>, %result: memref<3x4x5x6xf32>) -> () { return }
func @t2(%lhs: memref<3x4xf32>, %rhs: memref<3x4xf32>, %result: memref<3x4xf32>) -> () { return }

func @fn() {
  "print"() {op: "x.add", fn: @t1: (memref<3x4x5x6xf32>, memref<3x4x5x6xf32>, memref<3x4x5x6xf32>) -> ()} : () -> ()
  "print"() {op: "x.add", fn: @t2: (memref<3x4xf32>, memref<3x4xf32>, memref<3x4xf32>) -> ()} : () -> ()
  return
}

// CHECK: block {
// CHECK:   for(idx($12)=$16 to $4 step $17) {
// CHECK:     for(idx($13)=$16 to $5 step $17) {
// CHECK:       for(idx($14)=$16 to $6 step $17) {
// CHECK:         for(idx($15)=$16 to $7 step $17) {
// CHECK:           lhs($18) = store( ... );
// CHECK:         };
// CHECK:       };
// CHECK:     };
// CHECK:   }
// CHECK: }
// CHECK: block {
// CHECK:   for(idx($27)=$29 to $23 step $30) {
// CHECK:     for(idx($28)=$29 to $24 step $30) {
// CHECK:       lhs($31) = store( ... );
// CHECK:     };
// CHECK:   }
// CHECK: }
