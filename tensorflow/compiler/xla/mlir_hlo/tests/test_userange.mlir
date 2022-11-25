// RUN: mlir-hlo-opt -test-print-userange -split-input-file %s | FileCheck %s

// CHECK-LABEL: Testing : func_empty
func.func @func_empty() {
  func.return
}
//      CHECK:  ---- UserangeAnalysis -----
// CHECK-NEXT:  ---------------------------

// -----

// CHECK-LABEL: Testing : useRangeGap
func.func @useRangeGap(%arg0: i1, %arg1: memref<2xf32>, %arg2: memref<2xf32>)
{
  %0 = memref.alloc() : memref<2xf32>
  %1 = memref.alloc() : memref<2xf32>
  cf.cond_br %arg0, ^bb1, ^bb2
^bb1:
  "lmhlo.negate"(%arg1, %0) : (memref<2xf32>, memref<2xf32>) -> ()
  "lmhlo.negate"(%arg1, %1) : (memref<2xf32>, memref<2xf32>) -> ()
  cf.br ^bb3
^bb2:
  "lmhlo.negate"(%arg2, %0) : (memref<2xf32>, memref<2xf32>) -> ()
  "lmhlo.negate"(%arg2, %1) : (memref<2xf32>, memref<2xf32>) -> ()
  cf.br ^bb3
^bb3:
  func.return
}
//      CHECK:  Value: %[[A0:.*]] = memref.alloc
// CHECK-NEXT:  Userange: {(7, 7), (13, 13)}
//      CHECK:  Value: %[[A1:.*]] = memref.alloc
// CHECK-NEXT:  Userange: {(9, 9), (15, 15)}
//      CHECK:  %[[A0]] = memref.alloc
//      CHECK:  %[[A1]] = memref.alloc

// -----

// CHECK-LABEL: Testing : loopWithNestedRegion
func.func @loopWithNestedRegion(%arg0: i1, %arg1: memref<2xf32>, %arg2: memref<2xf32>)
{
  %0 = memref.alloc() : memref<2xf32>
  %1 = memref.alloc() : memref<2xf32>
  %2 = memref.alloc() : memref<2xf32>
  %3 = memref.alloc() : memref<2xf32>
  cf.br ^bb1
^bb1:
  %4 = scf.if %arg0 -> (memref<2xf32>) {
    "lmhlo.negate"(%arg1, %0) : (memref<2xf32>, memref<2xf32>) -> ()
    scf.yield %2 : memref<2xf32>
  } else {
    "lmhlo.negate"(%arg1, %1) : (memref<2xf32>, memref<2xf32>) -> ()
    scf.yield %2 : memref<2xf32>
  }
  cf.br ^bb2
^bb2:
  cf.cond_br %arg0, ^bb1, ^bb3
^bb3:
  "lmhlo.negate"(%arg1, %2) : (memref<2xf32>, memref<2xf32>) -> ()
  "lmhlo.negate"(%arg1, %3) : (memref<2xf32>, memref<2xf32>) -> ()
  func.return
}
//      CHECK:  Value: %[[A0:.*]] = memref.alloc
// CHECK-NEXT:  Userange: {(11, 23)}
//      CHECK:  Value: %[[A1:.*]] = memref.alloc
// CHECK-NEXT:  Userange: {(11, 23)}
//      CHECK:  Value: %[[A2:.*]] = memref.alloc
// CHECK-NEXT:  Userange: {(11, 25)}
//      CHECK:  Value: %[[A3:.*]] = memref.alloc
// CHECK-NEXT:  Userange: {(27, 27)}
//      CHECK:  Value: %[[A4:.*]] = scf.if
//      CHECK:  Userange: {(19, 19)}
//      CHECK:  %[[A0]] = memref.alloc
//      CHECK:  %[[A1]] = memref.alloc
//      CHECK:  %[[A2]] = memref.alloc
//      CHECK:  %[[A3]] = memref.alloc
//      CHECK:  %[[A4]] = scf.if

// -----

// CHECK-LABEL: Testing : condBranchWithAlias
func.func @condBranchWithAlias(%arg0: i1, %arg1: memref<2xf32>, %arg2: memref<2xf32>)
{
  %0 = memref.alloc() : memref<2xf32>
  cf.cond_br %arg0, ^bb1, ^bb2
^bb1:
  "lmhlo.negate"(%arg1, %0) : (memref<2xf32>, memref<2xf32>) -> ()
  cf.br ^bb3(%0 : memref<2xf32>)
^bb2:
  %1 = memref.alloc() : memref<2xf32>
  "lmhlo.negate"(%arg1, %1) : (memref<2xf32>, memref<2xf32>) -> ()
  cf.br ^bb3(%1 : memref<2xf32>)
^bb3(%2 : memref<2xf32>):
  %3 = memref.alloc() : memref<2xf32>
  "lmhlo.copy"(%2, %arg2) : (memref<2xf32>, memref<2xf32>) -> ()
  "lmhlo.copy"(%3, %arg2) : (memref<2xf32>, memref<2xf32>) -> ()
  %4 = memref.alloc() : memref<2xf32>
  "lmhlo.copy"(%4, %arg2) : (memref<2xf32>, memref<2xf32>) -> ()
  cf.br ^bb4(%0 : memref<2xf32>)
^bb4(%5 : memref<2xf32>):
  "lmhlo.copy"(%5, %arg2) : (memref<2xf32>, memref<2xf32>) -> ()
  func.return
}
//      CHECK:  Value: %[[A0:.*]] = memref.alloc
// CHECK-NEXT:  Userange: {(5, 7), (15, 27)}
//      CHECK:  Value: %[[A1:.*]] = memref.alloc
// CHECK-NEXT:  Userange: {(11, 17)}
//      CHECK:  Value: %[[A2:.*]] = memref.alloc
// CHECK-NEXT:  Userange: {(19, 19)}
//      CHECK:  Value: %[[A3:.*]] = memref.alloc
// CHECK-NEXT:  Userange: {(23, 23)}
//      CHECK:  Value: <block argument> of type 'memref<2xf32>' at index: 0
// CHECK-SAME:  Userange: {(15, 17)}
//      CHECK:  Value: <block argument> of type 'memref<2xf32>' at index: 0
// CHECK-SAME:  Userange: {(27, 27)}
//      CHECK:  %[[A0]] = memref.alloc
//      CHECK:  %[[A1]] = memref.alloc
//      CHECK:  %[[A2]] = memref.alloc
//      CHECK:  %[[A3]] = memref.alloc
