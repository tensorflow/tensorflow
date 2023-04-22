// RUN: mlir-hlo-opt -test-print-userange -split-input-file %s | FileCheck %s

// CHECK-LABEL: Testing : func_empty
func @func_empty() {
  return
}
//      CHECK:  ---- UserangeAnalysis -----
// CHECK-NEXT:  ---------------------------

// -----

// CHECK-LABEL: Testing : useRangeGap
func @useRangeGap(%arg0: i1, %arg1: memref<2xf32>, %arg2: memref<2xf32>)
{
  %0 = memref.alloc() : memref<2xf32>
  %1 = memref.alloc() : memref<2xf32>
  cond_br %arg0, ^bb1, ^bb2
^bb1:
  "lmhlo.negate"(%arg1, %0) : (memref<2xf32>, memref<2xf32>) -> ()
  "lmhlo.negate"(%arg1, %1) : (memref<2xf32>, memref<2xf32>) -> ()
  br ^bb3
^bb2:
  "lmhlo.negate"(%arg2, %0) : (memref<2xf32>, memref<2xf32>) -> ()
  "lmhlo.negate"(%arg2, %1) : (memref<2xf32>, memref<2xf32>) -> ()
  br ^bb3
^bb3:
  return
}
//      CHECK:  Value: %0 {{ *}}
// CHECK-NEXT:  Userange: {(6, 6), (12, 12)}
//      CHECK:  Value: %1 {{ *}}
// CHECK-NEXT:  Userange: {(8, 8), (14, 14)}

// -----

// CHECK-LABEL: Testing : loopWithNestedRegion
func @loopWithNestedRegion(%arg0: i1, %arg1: memref<2xf32>, %arg2: memref<2xf32>)
{
  %0 = memref.alloc() : memref<2xf32>
  %1 = memref.alloc() : memref<2xf32>
  %2 = memref.alloc() : memref<2xf32>
  %3 = memref.alloc() : memref<2xf32>
  br ^bb1
^bb1:
  %4 = scf.if %arg0 -> (memref<2xf32>) {
    "lmhlo.negate"(%arg1, %0) : (memref<2xf32>, memref<2xf32>) -> ()
    scf.yield %2 : memref<2xf32>
  } else {
    "lmhlo.negate"(%arg1, %1) : (memref<2xf32>, memref<2xf32>) -> ()
    scf.yield %2 : memref<2xf32>
  }
  br ^bb2
^bb2:
  cond_br %arg0, ^bb1, ^bb3
^bb3:
  "lmhlo.negate"(%arg1, %2) : (memref<2xf32>, memref<2xf32>) -> ()
  "lmhlo.negate"(%arg1, %3) : (memref<2xf32>, memref<2xf32>) -> ()
  return
}
//      CHECK:  Value: %0 {{ *}}
// CHECK-NEXT:  Userange: {(10, 22)}
//      CHECK:  Value: %1 {{ *}}
// CHECK-NEXT:  Userange: {(10, 22)}
//      CHECK:  Value: %2 {{ *}}
// CHECK-NEXT:  Userange: {(10, 24)}
//      CHECK:  Value: %3 {{ *}}
// CHECK-NEXT:  Userange: {(26, 26)}
//      CHECK:  Value: %4 {{ *}}
//      CHECK:  Userange: {(18, 18)}

// -----

// CHECK-LABEL: Testing : condBranchWithAlias
func @condBranchWithAlias(%arg0: i1, %arg1: memref<2xf32>, %arg2: memref<2xf32>)
{
  %0 = memref.alloc() : memref<2xf32>
  cond_br %arg0, ^bb1, ^bb2
^bb1:
  "lmhlo.negate"(%arg1, %0) : (memref<2xf32>, memref<2xf32>) -> ()
  br ^bb3(%0 : memref<2xf32>)
^bb2:
  %1 = memref.alloc() : memref<2xf32>
  "lmhlo.negate"(%arg1, %1) : (memref<2xf32>, memref<2xf32>) -> ()
  br ^bb3(%1 : memref<2xf32>)
^bb3(%2 : memref<2xf32>):
  %3 = memref.alloc() : memref<2xf32>
  "lmhlo.copy"(%2, %arg2) : (memref<2xf32>, memref<2xf32>) -> ()
  "lmhlo.copy"(%3, %arg2) : (memref<2xf32>, memref<2xf32>) -> ()
  %4 = memref.alloc() : memref<2xf32>
  "lmhlo.copy"(%4, %arg2) : (memref<2xf32>, memref<2xf32>) -> ()
  br ^bb4(%0 : memref<2xf32>)
^bb4(%5 : memref<2xf32>):
  "lmhlo.copy"(%5, %arg2) : (memref<2xf32>, memref<2xf32>) -> ()
  return
}
//      CHECK:  Value: %0 {{ *}}
// CHECK-NEXT:  Userange: {(4, 6), (14, 26)}
//      CHECK:  Value: %1 {{ *}}
// CHECK-NEXT:  Userange: {(10, 16)}
//      CHECK:  Value: %3 {{ *}}
// CHECK-NEXT:  Userange: {(18, 18)}
//      CHECK:  Value: %4 {{ *}}
// CHECK-NEXT:  Userange: {(22, 22)}
//      CHECK:  Value: <block argument> of type 'memref<2xf32>' at index: 0
// CHECK-SAME:  Userange: {(14, 16)}
//      CHECK:  Value: <block argument> of type 'memref<2xf32>' at index: 0
// CHECK-SAME:  Userange: {(26, 26)}
