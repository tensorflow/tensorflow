// RUN: mlir-opt -extract-fixed-outer-loops -outer-loop-sizes=7 %s | FileCheck %s --check-prefixes=COMMON,TILE_7
// RUN: mlir-opt -extract-fixed-outer-loops -outer-loop-sizes=7,4 %s | FileCheck %s --check-prefixes=COMMON,TILE_74

// COMMON-LABEL: @foo
func @foo(%arg0: memref<?x?xf32>) {
  %c2 = constant 2 : index
  %c44 = constant 44 : index
  %c1 = constant 1 : index
  // Range of the original loop:
  //   (upper - lower + step - 1) / step
  // where step is known to be %c1.
  // COMMON:      %[[diff:.*]] = subi %c44, %c2
  // COMMON:      %[[adjustment:.*]] = subi %c1, %c1_{{.*}}
  // COMMON-NEXT: %[[diff_adj:.*]] = addi %[[diff]], %[[adjustment]]
  // COMMON-NEXT: %[[range:.*]] = divis %[[diff_adj]], %c1

  // Ceildiv to get the parametric tile size.
  // COMMON:       %[[sum:.*]] = addi %[[range]], %c6
  // COMMON-NEXT:  %[[size:.*]] = divis %[[sum]], %c7

  // Range of the second original loop
  //   (upper - lower + step - 1) / step
  // where step is known to be %c2.
  // TILE_74:      %[[diff2:.*]] = subi %c44, %c1
  // TILE_74:      %[[adjustment2:.*]] = subi %c2, %c1_{{.*}}
  // TILE_74-NEXT: %[[diff2_adj:.*]] = addi %[[diff2]], %[[adjustment2]]
  // TILE_74-NEXT: %[[range2:.*]] = divis %[[diff2_adj]], %c2

  // Ceildiv to get the parametric tile size for the second original loop.
  // TILE_74:      %[[sum2:.*]] = addi %[[range2]], %c3
  // TILE_74-NEXT: %[[size2:.*]] = divis %[[sum2]], %c4

  // New step(s) (original is %c1 and %c2).
  // COMMON:      %[[step:.*]] = muli %c1, %[[size]]
  // TILE_74:     %[[step2:.*]] = muli %c2, %[[size2]]

  // Updated outer loop(s) use new steps.
  // COMMON:  for %[[i:.*]] = %c2 to %c44 step %[[step]]
  // TILE_74: for %[[j:.*]] = %c1 to %c44 step %[[step2]]
  for %i = %c2 to %c44 step %c1 {
    // Upper bound for the inner loop min(%i + %step, %c44).
    // COMMON:      %[[stepped:.*]] = addi %[[i]], %[[step]]
    // COMMON-NEXT: cmpi "slt", %c44, %[[stepped]]
    // COMMON-NEXT: %[[ub:.*]] = select {{.*}}, %c44, %[[stepped]]
    //
    // TILE_74:      %[[stepped2:.*]] = addi %[[j]], %[[step2]]
    // TILE_74-NEXT: cmpi "slt", %c44, %[[stepped2]]
    // TILE_74-NEXT: %[[ub2:.*]] = select {{.*}}, %c44, %[[stepped2]]

    // Created inner loop.
    // COMMON: for %[[ii:.*]] = %[[i]] to %[[ub:.*]] step %c1

    // This loop is not modified in TILE_7 case.
    // TILE_7:  for %[[j:.*]] = %c1 to %c44 step %c2
    //
    // But is modified in TILE_74 case.
    // TILE_74: for %[[jj:.*]] = %[[j]] to %[[ub2]] step %c2
    for %j = %c1 to %c44 step %c2 {
      // The right iterator are used.
      // TILE_7:  load %arg0[%[[ii]], %[[j]]]
      // TILE_74: load %arg0[%[[ii]], %[[jj]]]
      load %arg0[%i, %j]: memref<?x?xf32>
    }
  }
  return
}
