// RUN: mlir-hlo-opt -buffer-reuse -split-input-file %s | FileCheck %s

// Expected behavior: %0 replaces %1 and %2 replaces %3 and %4, since %2 is an
// alias of %0, %3 & %4 are also replaced by %0. %0 and %1 do not
// interfere with each other despite their shared alias.
// CHECK-LABEL: func @condBranchWithAlias
func.func @condBranchWithAlias(%arg0: i1, %arg1: memref<2xf32>, %arg2: memref<2xf32>)
{
  %0 = memref.alloc() : memref<2xf32>
  cf.cond_br %arg0, ^bb1, ^bb2
^bb1:
  memref.copy %arg1, %0 : memref<2xf32> to memref<2xf32>
  cf.br ^bb3(%0 : memref<2xf32>)
^bb2:
  %1 = memref.alloc() : memref<2xf32>
  memref.copy %arg1, %1 : memref<2xf32> to memref<2xf32>
  cf.br ^bb3(%1 : memref<2xf32>)
^bb3(%2 : memref<2xf32>):
  %3 = memref.alloc() : memref<2xf32>
  memref.copy %2, %arg2 : memref<2xf32> to memref<2xf32>
  memref.copy %3, %arg2 : memref<2xf32> to memref<2xf32>
  %4 = memref.alloc() : memref<2xf32>
  memref.copy %4, %arg2 : memref<2xf32> to memref<2xf32>
  func.return
}

// CHECK-SAME: %[[ARG0:.*]]: {{.*}}, %[[ARG1:.*]]: {{.*}}, %[[ARG2:.*]]: {{.*}}
// CHECK-NEXT: %[[ALLOC0:.*]] = memref.alloc()
// CHECK-NEXT: cf.cond_br %[[ARG0]], ^[[BB1:.*]], ^[[BB2:.*]]
//      CHECK: ^[[BB1]]:
// CHECK-NEXT: memref.copy %[[ARG1]], %[[ALLOC0]]
// CHECK-NEXT: cf.br ^[[BB3:.*]](%[[ALLOC0]]{{.*}}
//      CHECK: ^[[BB2]]:
// CHECK-NEXT: memref.copy %[[ARG1]], %[[ALLOC0]]
// CHECK-NEXT: cf.br ^[[BB3]](%[[ALLOC0]]{{.*}}
//      CHECK: ^[[BB3]](%[[BLOCKARG0:.*]]: {{.*}}):
// CHECK-NEXT: memref.copy %[[BLOCKARG0]], %[[ARG2]]
// CHECK-NEXT: memref.copy %[[ALLOC0]], %[[ARG2]]
// CHECK-NEXT: memref.copy %[[ALLOC0]], %[[ARG2]]
// CHECK-NEXT: return

// -----

// Expected behavior: Every alloc can be replaced by %0.
// CHECK-LABEL: func @allReuseSimple
func.func @allReuseSimple(%arg0: memref<2xf32>) {
  %0 = memref.alloc() : memref<2xf32>
  %1 = memref.alloc() : memref<2xf32>
  %2 = memref.alloc() : memref<2xf32>
  %3 = memref.alloc() : memref<2xf32>
  memref.copy %arg0, %0 : memref<2xf32> to memref<2xf32>
  memref.copy %arg0, %1 : memref<2xf32> to memref<2xf32>
  memref.copy %arg0, %2 : memref<2xf32> to memref<2xf32>
  memref.copy %arg0, %3 : memref<2xf32> to memref<2xf32>
  func.return
}

// CHECK-SAME: %[[ARG0:.*]]: {{.*}}
// CHECK-NEXT: %[[ALLOC0:.*]] = memref.alloc()
// CHECK-NEXT: memref.copy %[[ARG0]], %[[ALLOC0]]
// CHECK-NEXT: memref.copy %[[ARG0]], %[[ALLOC0]]
// CHECK-NEXT: memref.copy %[[ARG0]], %[[ALLOC0]]
// CHECK-NEXT: memref.copy %[[ARG0]], %[[ALLOC0]]
// CHECK-NEXT: return

// -----

// Expected behavior: %0 can't replace %1 as its alloc OP does not dominate the
// first use of %1.
// CHECK-LABEL: func @allocDominance
func.func @allocDominance(%arg0: i1, %arg1: memref<2xf32>) {
  cf.cond_br %arg0, ^bb1, ^bb2
 ^bb1:
  %0 = memref.alloc() : memref<2xf32>
  memref.copy %arg1, %0 : memref<2xf32> to memref<2xf32>
  cf.br ^bb2
 ^bb2:
  %1 = memref.alloc() : memref<2xf32>
  memref.copy %arg1, %1 : memref<2xf32> to memref<2xf32>
  func.return
}

// CHECK-SAME: %[[ARG0:.*]]: {{.*}}, %[[ARG1:.*]]: {{.*}}
// CHECK-NEXT: cf.cond_br %[[ARG0]], ^[[BB1:.*]], ^[[BB2:.*]]
//      CHECK: ^[[BB1]]:
// CHECK-NEXT: %[[ALLOC0:.*]] = memref.alloc()
// CHECK-NEXT: memref.copy %[[ARG1]], %[[ALLOC0]]
// CHECK-NEXT: cf.br ^[[BB2]]
//      CHECK: ^[[BB2]]:
// CHECK-NEXT: %[[ALLOC1:.*]] = memref.alloc()
// CHECK-NEXT: memref.copy %[[ARG1]], %[[ALLOC1]]
// CHECK-NEXT: return

// -----

// Expected behavior: Nothing can be replaced as there is an alias interference.
// CHECK-LABEL: func @aliasInterference
func.func @aliasInterference(%arg0: i1, %arg1: memref<2xf32>) {
  %0 = memref.alloc() : memref<2xf32>
  %1 = memref.alloc() : memref<2xf32>
  memref.copy %arg1, %0 : memref<2xf32> to memref<2xf32>
  cf.br ^bb1(%0 : memref<2xf32>)
^bb1(%2 : memref<2xf32>):
  memref.copy %arg1, %1 : memref<2xf32> to memref<2xf32>
  memref.copy %arg1, %2 : memref<2xf32> to memref<2xf32>
  func.return
}

// CHECK-SAME: %[[ARG0:.*]]: {{.*}}, %[[ARG1:.*]]: {{.*}}
// CHECK-NEXT: %[[ALLOC0:.*]] = memref.alloc()
// CHECK-NEXT: %[[ALLOC1:.*]] = memref.alloc()
// CHECK-NEXT: memref.copy %[[ARG1]], %[[ALLOC0]]
// CHECK-NEXT: cf.br ^[[BB1:.*]](%[[ALLOC0]]{{.*}}
//      CHECK: ^[[BB1]](%[[BLOCKARG0:.*]]: {{.*}}):
// CHECK-NEXT: memref.copy %[[ARG1]], %[[ALLOC1]]
// CHECK-NEXT: memref.copy %[[ARG1]], %[[BLOCKARG0]]
// CHECK-NEXT: return

// -----

// Expected behavior: %1 should be replaced by %2 as there is no interference.
// %2 is an alias of %0, so %0 replaces %1.
// CHECK-LABEL: func @aliasReuse
func.func @aliasReuse(%arg0: memref<2xf32>) {
  %0 = memref.alloc() : memref<2xf32>
  %1 = memref.alloc() : memref<2xf32>
  memref.copy %arg0, %0 : memref<2xf32> to memref<2xf32>
  cf.br ^bb1(%0 : memref<2xf32>)
^bb1(%2 : memref<2xf32>):
  memref.copy %arg0, %2 : memref<2xf32> to memref<2xf32>
  memref.copy %arg0, %1 : memref<2xf32> to memref<2xf32>
  func.return
}
// CHECK-SAME: %[[ARG0:.*]]: {{.*}}
// CHECK-NEXT: %[[ALLOC0:.*]] = memref.alloc()
// CHECK-NEXT: memref.copy %[[ARG0]], %[[ALLOC0]]
// CHECK-NEXT: cf.br ^[[BB1:.*]](%[[ALLOC0]]{{.*}}
//      CHECK: ^[[BB1]](%[[BLOCKARG0:.*]]: {{.*}}):
// CHECK-NEXT: memref.copy %[[ARG0]], %[[BLOCKARG0]]
// CHECK-NEXT: memref.copy %[[ARG0]], %[[ALLOC0]]
// CHECK-NEXT: return

// -----

// Expected behavior: Nothing should be replaced as both buffers interfere
// within a single OP. There is no replace in place, because copy is not
// elementwise.
// CHECK-LABEL: func @sameOperation
func.func @sameOperation() {
  %0 = memref.alloc() : memref<2xf32>
  %1 = memref.alloc() : memref<2xf32>
  memref.copy %1, %0 : memref<2xf32> to memref<2xf32>
  func.return
}

// CHECK-NEXT: %[[ALLOC0:.*]] = memref.alloc()
// CHECK-NEXT: %[[ALLOC1:.*]] = memref.alloc()
// CHECK-NEXT: memref.copy %[[ALLOC1]], %[[ALLOC0]]
// CHECK-NEXT: return

// -----

// Expected behavior: %0 replaces both %1 and %2.
// CHECK-LABEL: func @branchReuse
func.func @branchReuse(%arg0: i1, %arg1: memref<2xf32>) {
  %0 = memref.alloc() : memref<2xf32>
  %1 = memref.alloc() : memref<2xf32>
  %2 = memref.alloc() : memref<2xf32>
  cf.cond_br %arg0, ^bb1, ^bb2
^bb1:
  memref.copy %0, %arg1 : memref<2xf32> to memref<2xf32>
  memref.copy %arg1, %2 : memref<2xf32> to memref<2xf32>
  cf.br ^bb3
^bb2:
  memref.copy %arg1, %1 : memref<2xf32> to memref<2xf32>
  cf.br ^bb3
^bb3:
  func.return
}

// CHECK-SAME: %[[ARG0:.*]]: {{.*}}, %[[ARG1:.*]]: {{.*}}
// CHECK-NEXT: %[[ALLOC0:.*]] = memref.alloc()
// CHECK-NEXT: cf.cond_br %[[ARG0]], ^[[BB1:.*]], ^[[BB2:.*]]
//      CHECK: ^[[BB1]]:
// CHECK-NEXT: memref.copy %[[ALLOC0]], %[[ARG1]]
// CHECK-NEXT: memref.copy %[[ARG1]], %[[ALLOC0]]
// CHECK-NEXT: cf.br ^[[BB3:.*]]
//      CHECK: ^[[BB2]]:
// CHECK-NEXT: memref.copy %[[ARG1]], %[[ALLOC0]]
// CHECK-NEXT: cf.br ^[[BB3]]
//      CHECK: ^[[BB3]]:
// CHECK-NEXT: return

// -----

// Expected behavior: No replacement due to type mismatch.
// CHECK-LABEL: func @typeMismatch
func.func @typeMismatch(%arg0: memref<2xf32>, %arg1: memref<4xf16>) {
  %0 = memref.alloc() : memref<2xf32>
  %1 = memref.alloc() : memref<4xf16>
  memref.copy %arg0, %0 : memref<2xf32> to memref<2xf32>
  memref.copy %arg1, %1 : memref<4xf16> to memref<4xf16>
  func.return
}

// CHECK-SAME: %[[ARG0:.*]]: {{.*}}, %[[ARG1:.*]]: {{.*}}
// CHECK-NEXT: %[[ALLOC0:.*]] = memref.alloc()
// CHECK-NEXT: %[[ALLOC1:.*]] = memref.alloc()
// CHECK-NEXT: memref.copy %[[ARG0]], %[[ALLOC0]]
// CHECK-NEXT: memref.copy %[[ARG1]], %[[ALLOC1]]
// CHECK-NEXT: return

// -----

// Expected behavior: %2 replaces %5 due to the same allocation with the same
// operands, otherwise no replacement due to type mismatch.
// CHECK-LABEL: func @complexTypeMatching
func.func @complexTypeMatching(%arg0: i1, %arg1: index, %arg2: index, %arg3 : index) {
  %0 = memref.alloc() : memref<2xf32>
  %1 = memref.alloc(%arg1) : memref<?xf32>
  %2 = memref.alloc(%arg2) : memref<?xf32>
  %3 = memref.alloc(%arg1, %arg2) : memref<?x?xf32>
  %4 = memref.alloc(%arg1, %arg3) : memref<?x?xf32>
  %5 = memref.alloc(%arg2) : memref<?xf32>
  memref.copy %0, %0 : memref<2xf32> to memref<2xf32>
  cf.cond_br %arg0, ^bb1, ^bb4
^bb1:
  memref.copy %1, %1 : memref<?xf32> to memref<?xf32>
  cf.cond_br %arg0, ^bb2, ^bb3
^bb2:
  memref.copy %3, %3 : memref<?x?xf32> to memref<?x?xf32>
  memref.copy %4, %4 : memref<?x?xf32> to memref<?x?xf32>
  cf.br ^bb4
^bb3:
  memref.copy %5, %5 : memref<?xf32> to memref<?xf32>
  cf.br ^bb2
^bb4:
  memref.copy %2, %2 : memref<?xf32> to memref<?xf32>
  func.return
}

// CHECK-SAME: %[[ARG0:.*]]: {{.*}}, %[[ARG1:.*]]: {{.*}}, %[[ARG2:.*]]: {{.*}},
// CHECK-SAME: %[[ARG3:.*]]: {{.*}}
// CHECK-NEXT: %[[ALLOC0:.*]] = memref.alloc()
// CHECK-NEXT: %[[ALLOC1:.*]] = memref.alloc(%[[ARG1]])
// CHECK-NEXT: %[[ALLOC2:.*]] = memref.alloc(%[[ARG2]])
// CHECK-NEXT: %[[ALLOC3:.*]] = memref.alloc(%[[ARG1]], %[[ARG2]])
// CHECK-NEXT: %[[ALLOC4:.*]] = memref.alloc(%[[ARG1]], %[[ARG3]])
// CHECK-NEXT: memref.copy %[[ALLOC0]], %[[ALLOC0]]
// CHECK-NEXT: cf.cond_br %[[ARG0]], ^[[BB1:.*]], ^[[BB4:.*]]
//      CHECK: ^[[BB1]]:
// CHECK-NEXT: memref.copy %[[ALLOC1]], %[[ALLOC1]]
// CHECK-NEXT: cf.cond_br %[[ARG0]], ^[[BB2:.*]], ^[[BB3:.*]]
//      CHECK: ^[[BB2]]:
// CHECK-NEXT: memref.copy %[[ALLOC3]], %[[ALLOC3]]
// CHECK-NEXT: memref.copy %[[ALLOC4]], %[[ALLOC4]]
//      CHECK: ^[[BB3]]:
// CHECK-NEXT: memref.copy %[[ALLOC2]], %[[ALLOC2]]
//      CHECK: ^[[BB4]]:
// CHECK-NEXT: memref.copy %[[ALLOC2]], %[[ALLOC2]]
// CHECK-NEXT: return

// -----

// Expected behavior: In this case %2 can replace %0 and %0 can replace %1.
// However, %2 cannot replace %1. Due to the ordering of the allocs the only
// valid replacement is %0 replaces %2.
// CHECK-LABEL: func @nonTransitive
func.func @nonTransitive(%arg0: i1, %arg1: memref<2xf32>) {
  %0 = memref.alloc() : memref<2xf32>
  %1 = memref.alloc() : memref<2xf32>
  %2 = memref.alloc() : memref<2xf32>
  cf.cond_br %arg0, ^bb1, ^bb2
 ^bb1:
  memref.copy %arg1, %2 : memref<2xf32> to memref<2xf32>
  memref.copy %arg1, %1 : memref<2xf32> to memref<2xf32>
  memref.copy %arg1, %2 : memref<2xf32> to memref<2xf32>
  cf.br ^bb3
 ^bb2:
  memref.copy %arg1, %0 : memref<2xf32> to memref<2xf32>
  cf.br ^bb3
 ^bb3:
  func.return
}

// CHECK-SAME: %[[ARG0:.*]]: {{.*}}, %[[ARG1:.*]]: {{.*}}
// CHECK-NEXT: %[[ALLOC0:.*]] = memref.alloc()
// CHECK-NEXT: %[[ALLOC1:.*]] = memref.alloc()
// CHECK-NEXT: cf.cond_br %[[ARG0]], ^[[BB1:.*]], ^[[BB2:.*]]
//      CHECK: ^[[BB1]]:
// CHECK-NEXT: memref.copy %[[ARG1]], %[[ALLOC0]]
// CHECK-NEXT: memref.copy %[[ARG1]], %[[ALLOC1]]
// CHECK-NEXT: memref.copy %[[ARG1]], %[[ALLOC0]]
// CHECK-NEXT: cf.br ^[[BB3:.*]]
//      CHECK: ^[[BB2]]:
// CHECK-NEXT: memref.copy %[[ARG1]], %[[ALLOC0]]
// CHECK-NEXT: cf.br ^[[BB3]]
//      CHECK: ^[[BB3]]:
// CHECK-NEXT: return

// -----

// Expected behavior: %1 can replace %2 and %5.
// CHECK-LABEL: func @nestedRegionControlFlow
func.func @nestedRegionControlFlow(%arg0 : index, %arg1 : index) -> memref<2xf32> {
  %0 = arith.cmpi "eq", %arg0, %arg1 : index
  %1 = memref.alloc() : memref<2xf32>
  %2 = memref.alloc() : memref<2xf32>
  %3 = scf.if %0 -> (memref<2xf32>) {
    %4 = scf.if %0 -> (memref<2xf32>) {
      scf.yield %1 : memref<2xf32>
    } else {
      %5 = memref.alloc() : memref<2xf32>
      scf.yield %5 : memref<2xf32>
    }
    scf.yield %4 : memref<2xf32>
  } else {
    memref.copy %1, %1 : memref<2xf32> to memref<2xf32>
    scf.yield %2 : memref<2xf32>
  }
  func.return %3 : memref<2xf32>
}

//      CHECK: %[[ALLOC1:.*]] = memref.alloc()
// CHECK-NEXT: %[[ALLOC2:.*]] = scf.if
// CHECK-NEXT: %[[ALLOC3:.*]] = scf.if
//      CHECK: scf.yield %[[ALLOC1]]
//      CHECK: scf.yield %[[ALLOC1]]
//      CHECK: scf.yield %[[ALLOC3]]
//      CHECK: memref.copy %[[ALLOC1]], %[[ALLOC1]]
// CHECK-NEXT: scf.yield %[[ALLOC1]]
//      CHECK: return %[[ALLOC2]]

// -----

// Expected behavior: Nothing should be reused here.
// CHECK-LABEL: func @nestedRegionControlFlowNoReuse
func.func @nestedRegionControlFlowNoReuse(%arg0 : index,
                                     %arg1 : index) -> memref<2xf32> {
  %0 = arith.cmpi "eq", %arg0, %arg1 : index
  %1 = memref.alloc() : memref<2xf32>
  %2 = memref.alloc() : memref<2xf32>
  %3 = scf.if %0 -> (memref<2xf32>) {
    %4 = scf.if %0 -> (memref<2xf32>) {
      scf.yield %1 : memref<2xf32>
    } else {
      scf.yield %2 : memref<2xf32>
    }
    scf.yield %4 : memref<2xf32>
  } else {
    memref.copy %1, %2 : memref<2xf32> to memref<2xf32>
    scf.yield %2 : memref<2xf32>
  }
  func.return %3 : memref<2xf32>
}

//      CHECK: %[[ALLOC1:.*]] = memref.alloc()
// CHECK-NEXT: %[[ALLOC2:.*]] = memref.alloc()
// CHECK-NEXT: %[[ALLOC3:.*]] = scf.if
// CHECK-NEXT: %[[ALLOC4:.*]] = scf.if
//      CHECK: scf.yield %[[ALLOC1]]
//      CHECK: scf.yield %[[ALLOC2]]
//      CHECK: scf.yield %[[ALLOC4]]
//      CHECK: memref.copy %[[ALLOC1]], %[[ALLOC2]]
// CHECK-NEXT: scf.yield %[[ALLOC2]]
//      CHECK: return %[[ALLOC3]]

// -----

// Expected behavior: No reuse is possible here as both buffers are used inside
//                    a loop and thus interfere.
// CHECK-LABEL: func @noReuseInNestedRegionLoop
func.func @noReuseInNestedRegionLoop(
  %arg0: memref<2xf32>,
  %lb: index,
  %ub: index,
  %step: index,
  %buf: memref<2xf32>) -> memref<2xf32> {
  %0 = memref.alloc() : memref<2xf32>
  %1 = memref.alloc() : memref<2xf32>
  %2 = scf.for %i = %lb to %ub step %step
    iter_args(%iterBuf = %buf) -> memref<2xf32> {
    %3 = arith.cmpi "eq", %lb, %ub : index
    %4 = scf.if %3 -> (memref<2xf32>) {
      scf.yield %arg0 : memref<2xf32>
    } else {
      memref.copy %arg0, %arg0 : memref<2xf32> to memref<2xf32>
      memref.copy %0, %0 : memref<2xf32> to memref<2xf32>
      scf.yield %arg0 : memref<2xf32>
    }
    memref.copy %arg0, %1 : memref<2xf32> to memref<2xf32>
    scf.yield %arg0 : memref<2xf32>
  }
  func.return %2 : memref<2xf32>
}

//      CHECK: %[[ALLOC0:.*]] = memref.alloc()
// CHECK-NEXT: %[[ALLOC1:.*]] = memref.alloc()

// -----

// Expected behavior: %0 and %1 cannot replace each other as they are used
// inside a loop. However, %0 can replace %2 as it is only used after the loop.
// CHECK-LABEL: func @replaceAfterLoop
func.func @replaceAfterLoop(%arg0: i1, %arg1: memref<2xf32>, %arg2: memref<2xf32>)
{
  %0 = memref.alloc() : memref<2xf32>
  %1 = memref.alloc() : memref<2xf32>
  %2 = memref.alloc() : memref<2xf32>
  cf.br ^bb1
^bb1:
  cf.cond_br %arg0, ^bb2, ^bb3
^bb2:
  memref.copy %arg1, %0 : memref<2xf32> to memref<2xf32>
  memref.copy %arg1, %1 : memref<2xf32> to memref<2xf32>
  cf.br ^bb4
^bb3:
  memref.copy %arg2, %0 : memref<2xf32> to memref<2xf32>
  memref.copy %arg2, %1 : memref<2xf32> to memref<2xf32>
  cf.br ^bb4
^bb4:
  cf.cond_br %arg0, ^bb1, ^bb5
^bb5:
  memref.copy %arg1, %2 : memref<2xf32> to memref<2xf32>
  func.return
}

// CHECK-SAME: %[[ARG0:.*]]: {{.*}}, %[[ARG1:.*]]: {{.*}}, %[[ARG2:.*]]: {{.*}}
//      CHECK: %[[ALLOC0:.*]] = memref.alloc()
//      CHECK: %[[ALLOC1:.*]] = memref.alloc()
//      CHECK: memref.copy %[[ARG1]], %[[ALLOC0]]
// CHECK-NEXT: memref.copy %[[ARG1]], %[[ALLOC1]]
//      CHECK: memref.copy %[[ARG2]], %[[ALLOC0]]
// CHECK-NEXT: memref.copy %[[ARG2]], %[[ALLOC1]]
//      CHECK: memref.copy %[[ARG1]], %[[ALLOC0]]

// -----

// Expected behavior: Due to the gap in the userange %0 can replace %1.
// CHECK-LABEL: func @useRangeGap
func.func @useRangeGap(%arg0: i1, %arg1: memref<2xf32>, %arg2: memref<2xf32>)
{
  %0 = memref.alloc() : memref<2xf32>
  %1 = memref.alloc() : memref<2xf32>
  cf.cond_br %arg0, ^bb1, ^bb2
^bb1:
  memref.copy %arg1, %0 : memref<2xf32> to memref<2xf32>
  memref.copy %arg1, %1 : memref<2xf32> to memref<2xf32>
  cf.br ^bb3
^bb2:
  memref.copy %arg2, %0 : memref<2xf32> to memref<2xf32>
  memref.copy %arg2, %1 : memref<2xf32> to memref<2xf32>
  cf.br ^bb3
^bb3:
  func.return
}

// CHECK-SAME: %[[ARG0:.*]]: {{.*}}, %[[ARG1:.*]]: {{.*}}, %[[ARG2:.*]]: {{.*}}
//      CHECK: %[[ALLOC0:.*]] = memref.alloc()
//      CHECK: memref.copy %[[ARG1]], %[[ALLOC0]]
// CHECK-NEXT: memref.copy %[[ARG1]], %[[ALLOC0]]
//      CHECK: memref.copy %[[ARG2]], %[[ALLOC0]]
// CHECK-NEXT: memref.copy %[[ARG2]], %[[ALLOC0]]

// -----

// Expected behavior: Due to the loop from block ^bb2 to ^bb1 it is not possible
// to replace anything inside the nested region. In addion %2 cannot be replaced
// outside the loop, because of the interference inside the loop. %3 can be
// replaced by %0.
// CHECK-LABEL: func @loopWithNestedRegion
func.func @loopWithNestedRegion(%arg0: i1, %arg1: memref<2xf32>, %arg2: memref<2xf32>)
{
  %0 = memref.alloc() : memref<2xf32>
  %1 = memref.alloc() : memref<2xf32>
  %2 = memref.alloc() : memref<2xf32>
  %3 = memref.alloc() : memref<2xf32>
  cf.br ^bb1
^bb1:
  %4 = scf.if %arg0 -> (memref<2xf32>) {
    memref.copy %arg1, %0 : memref<2xf32> to memref<2xf32>
    scf.yield %2 : memref<2xf32>
  } else {
    memref.copy %arg1, %1 : memref<2xf32> to memref<2xf32>
    scf.yield %2 : memref<2xf32>
  }
  cf.br ^bb2
^bb2:
  cf.cond_br %arg0, ^bb1, ^bb3
^bb3:
  memref.copy %arg1, %2 : memref<2xf32> to memref<2xf32>
  memref.copy %arg1, %3 : memref<2xf32> to memref<2xf32>
  func.return
}

// CHECK-SAME: %[[ARG0:.*]]: {{.*}}, %[[ARG1:.*]]: {{.*}}, %[[ARG2:.*]]: {{.*}}
//      CHECK: %[[ALLOC0:.*]] = memref.alloc()
// CHECK-NEXT: %[[ALLOC1:.*]] = memref.alloc()
// CHECK-NEXT: %[[ALLOC2:.*]] = memref.alloc()
// CHECK-NEXT: cf.br ^[[BB1:.*]]
//      CHECK: memref.copy %[[ARG1]], %[[ALLOC0]]
//      CHECK: memref.copy %[[ARG1]], %[[ALLOC1]]
//      CHECK: memref.copy %[[ARG1]], %[[ALLOC2]]
//      CHECK: memref.copy %[[ARG1]], %[[ALLOC0]]

// -----

// Expected behavior: %2 cannot replace %1
// CHECK-LABEL: func @noReuseOuterRegion
func.func @noReuseOuterRegion(%arg0 : i1, %arg1 : memref<2xf32>) -> memref<2xf32> {
  scf.if %arg0 {
    %1 = memref.alloc() : memref<2xf32>
    memref.copy %arg1, %1 : memref<2xf32> to memref<2xf32>
  }
  %0 = memref.alloc() : memref<2xf32>
  memref.copy %arg1, %0 : memref<2xf32> to memref<2xf32>
  func.return %0 : memref<2xf32>
}

// CHECK-SAME: %[[ARG0:.*]]: {{.*}}, %[[ARG1:.*]]: {{.*}})
//      CHECK: scf.if %[[ARG0]] {
// CHECK-NEXT: %[[ALLOC0:.*]] = memref.alloc()
// CHECK-NEXT: memref.copy %[[ARG1]], %[[ALLOC0]]
// CHECK-NEXT: }
// CHECK-NEXT: %[[ALLOC1:.*]] = memref.alloc()
// CHECK-NEXT: memref.copy %[[ARG1]], %[[ALLOC1]]
// CHECK-NEXT: return %[[ALLOC1]]
