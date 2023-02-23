// RUN: mlir-hlo-opt -thlo-legalize-sort -canonicalize %s | FileCheck %s

func.func @sort(%input1: memref<?x?xf32>, %input2: memref<?x?xi32>,
                %init1: memref<?x?xf32>, %init2: memref<?x?xi32>) {
  thlo.sort
      ins(%input1: memref<?x?xf32>, %input2: memref<?x?xi32>)
      outs(%init1: memref<?x?xf32>, %init2: memref<?x?xi32>)
      dimension = 0
      is_stable = true
      (%e11: f32, %e12: f32, %e21: i32, %e22: i32) {
        %gt = arith.cmpf ogt, %e11, %e12: f32
        thlo.yield %gt : i1
      }
  func.return
}

// CHECK-LABEL:   func.func @sort(
// CHECK-SAME:                    %[[INPUT1:[A-Za-z0-9]*]]: memref<?x?xf32>,
// CHECK-SAME:                    %[[INPUT2:[A-Za-z0-9]*]]: memref<?x?xi32>,
// CHECK-SAME:                    %[[INIT1:[A-Za-z0-9]*]]: memref<?x?xf32>,
// CHECK-SAME:                    %[[INIT2:[A-Za-z0-9]*]]: memref<?x?xi32>) {
// CHECK:           %[[CTRUE:.*]] = arith.constant true
// CHECK:           %[[C16:.*]] = arith.constant 16 : index
// CHECK:           %[[CFALSE:.*]] = arith.constant false
// CHECK:           %[[C1:.*]] = arith.constant 1 : index
// CHECK:           %[[C0:.*]] = arith.constant 0 : index
// CHECK:           %[[SORT_DIM:.*]] = memref.dim %[[INPUT1]], %[[C0]]
// CHECK:           %[[DYN_DIM0:.*]] = memref.dim %[[INPUT1]], %[[C0]]
// CHECK:           %[[DYN_DIM1:.*]] = memref.dim %[[INPUT1]], %[[C1]]
// CHECK:           %[[SCRATCH1:.*]] = memref.alloc(%[[DYN_DIM0]], %[[DYN_DIM1]])
// CHECK:           %[[SCRATCH2:.*]] = memref.alloc(%[[DYN_DIM0]], %[[DYN_DIM1]])
// CHECK:           %[[BATCH_DIM_SIZE:.*]] = memref.dim %[[INPUT1]], %[[C1]]
// CHECK:           %[[PARITY:.*]] = scf.for
// CHECK-SAME:          %[[SUBVIEW_INDEX:.*]] = %[[C0]] to %[[BATCH_DIM_SIZE]]
// CHECK-SAME:          step %[[C1]]
// CHECK-SAME:          iter_args(%[[ARG5:.*]] = %[[CFALSE]]) -> (i1) {
// CHECK:             %[[SUBVIEW_INPUT1:.*]] = memref.subview
// CHECK-SAME:            %[[INPUT1]][0, %[[SUBVIEW_INDEX]]]
// CHECK-SAME             [%[[SORT_DIM]], 1] [1, 1]
// CHECK:             %[[SUBVIEW_INPUT2:.*]] = memref.subview
// CHECK-SAME:            %[[INPUT2]][0, %[[SUBVIEW_INDEX]]]
// CHECK-SAME             [%[[SORT_DIM]], 1] [1, 1]
// CHECK:             %[[SUBVIEW_INIT1:.*]] = memref.subview
// CHECK-SAME:            %[[INIT1]][0, %[[SUBVIEW_INDEX]]]
// CHECK-SAME             [%[[SORT_DIM]], 1] [1, 1]
// CHECK:             %[[SUBVIEW_INIT2:.*]] = memref.subview
// CHECK-SAME:            %[[INIT2]][0, %[[SUBVIEW_INDEX]]]
// CHECK-SAME             [%[[SORT_DIM]], 1] [1, 1]
// CHECK:             %[[SUBVIEW_SCRATCH1:.*]] = memref.subview
// CHECK-SAME:            %[[SCRATCH1]][0, %[[SUBVIEW_INDEX]]]
// CHECK-SAME             [%[[SORT_DIM]], 1] [1, 1]
// CHECK:             %[[SUBVIEW_SCRATCH2:.*]] = memref.subview
// CHECK-SAME:            %[[SCRATCH2]][0, %[[SUBVIEW_INDEX]]]
// CHECK-SAME             [%[[SORT_DIM]], 1] [1, 1]
// COM:               // We first sort ELEMs in groups of 16 using an
// COM:               // insertion sort.
// CHECK:             scf.for %[[LO:.*]] = %[[C0]] to %[[SORT_DIM]]
// CHECK-SAME:                step %[[C16]] {
// CHECK:               %[[UPPER_BOUND:.*]] = arith.addi %[[LO]], %[[C16]]
// CHECK:               %[[END:.*]] = arith.minsi %[[UPPER_BOUND]], %[[SORT_DIM]]
// CHECK:               %[[LO_IN1:.*]] = memref.load %[[SUBVIEW_INPUT1]][%[[LO]]]
// CHECK:               %[[LO_IN2:.*]] = memref.load %[[SUBVIEW_INPUT2]][%[[LO]]]
// CHECK:               memref.store %[[LO_IN1]], %[[SUBVIEW_INIT1]][%[[LO]]]
// CHECK:               memref.store %[[LO_IN2]], %[[SUBVIEW_INIT2]][%[[LO]]]
// CHECK:               %[[LO_PLUS_1:.*]] = arith.addi %[[LO]], %[[C1]]
// CHECK:               scf.for %[[START:.*]] = %[[LO_PLUS_1]] to %[[END]]
// CHECK-SAME:                  step %[[C1]] {
// CHECK:                 %[[PIVOT1:.*]] = memref.load %[[SUBVIEW_INPUT1]][%[[START]]]
// CHECK:                 %[[PIVOT2:.*]] = memref.load %[[SUBVIEW_INPUT2]][%[[START]]]
// COM:                   // Binary search of the insertion point.
// CHECK:                 %[[LR:.*]]:2 = scf.while
// CHECK-SAME:                (%[[LEFT:.*]] = %[[LO]], %[[RIGHT:.*]] = %[[START]])
// CHECK-SAME:                : (index, index) -> (index, index) {
// CHECK:                   %[[L_LT_R:.*]] = arith.cmpi slt, %[[LEFT]], %[[RIGHT]]
// CHECK:                   scf.condition(%[[L_LT_R]]) %[[LEFT]], %[[RIGHT]]
// CHECK:                 } do {
// CHECK:                 ^bb0(%[[LEFT_:.*]]: index, %[[RIGHT_:.*]]: index):
// CHECK:                   %[[SUM_LR:.*]] = arith.addi %[[LEFT_]], %[[RIGHT_]]
// CHECK:                   %[[MID:.*]] = arith.shrui %[[SUM_LR]], %[[C1]]
// CHECK:                   %[[MID_PLUS_1:.*]] = arith.addi %[[MID]], %[[C1]]
// CHECK:                   %[[MEDIAN:.*]] = memref.load %[[SUBVIEW_INIT1]][%[[MID]]]
// CHECK:                   %[[CMP_PIVOT_MEDIAN:.*]] = arith.cmpf ogt, %[[PIVOT1]], %[[MEDIAN]] : f32
// CHECK:                   %[[NEW_LEFT:.*]] = arith.select %[[CMP_PIVOT_MEDIAN]], %[[LEFT_]], %[[MID_PLUS_1]]
// CHECK:                   %[[NEW_RIGHT:.*]] = arith.select %[[CMP_PIVOT_MEDIAN]], %[[MID]], %[[RIGHT_]]
// CHECK:                   scf.yield %[[NEW_LEFT]], %[[NEW_RIGHT]]
// CHECK:                 }
// COM:                   // Move the n ELEMs that are larger than the pivot
// COM:                   // once to the right.
// CHECK:                 %[[N:.*]] = arith.subi %[[START]], %[[LR:.*]]#0
// CHECK:                 scf.for %[[I:.*]] = %[[C0]] to %[[N]] step %[[C1]] {
// CHECK:                   %[[CUR_IX:.*]] = arith.subi %[[START]], %[[I]]
// CHECK:                   %[[CUR_IX_MINUS_1:.*]] = arith.subi %[[CUR_IX]], %[[C1]] : index
// CHECK:                   %[[ELEM_TO_MOVE1:.*]] = memref.load %[[SUBVIEW_INIT1]][%[[CUR_IX_MINUS_1]]]
// CHECK:                   %[[ELEM_TO_MOVE2:.*]] = memref.load %[[SUBVIEW_INIT2]][%[[CUR_IX_MINUS_1]]]
// CHECK:                   memref.store %[[ELEM_TO_MOVE1]], %[[SUBVIEW_INIT1]][%[[CUR_IX]]]
// CHECK:                   memref.store %[[ELEM_TO_MOVE2]], %[[SUBVIEW_INIT2]][%[[CUR_IX]]]
// CHECK:                 }
// CHECK:                 memref.store %[[PIVOT1]], %[[SUBVIEW_INIT1]][%[[LR]]#0]
// CHECK:                 memref.store %[[PIVOT2]], %[[SUBVIEW_INIT2]][%[[LR]]#0]
// CHECK:               }
// CHECK:             }
// COM:               // Merge subarrays of each input together until the final
// COM:               // sorted array is computed.
// CHECK:             %[[MERGE_RESULTS:.*]]:6 = scf.while
// CHECK-SAME:            (%[[SUBARRAY_SIZE:[A-Za-z0-9]*]] = %[[C16]],
// CHECK-SAME:             %[[PARITY_:[A-Za-z0-9]*]] = %[[CFALSE]],
// CHECK-SAME:             %[[READ_BUF1:[A-Za-z0-9]*]] = %[[SUBVIEW_INIT1]],
// CHECK-SAME:             %[[READ_BUF2:[A-Za-z0-9]*]] = %[[SUBVIEW_INIT2]],
// CHECK-SAME:             %[[WRITE_BUF1:[A-Za-z0-9]*]] = %[[SUBVIEW_SCRATCH1]],
// CHECK-SAME:             %[[WRITE_BUF2:[A-Za-z0-9]*]] = %[[SUBVIEW_SCRATCH2]])
// CHECK:               %[[ARE_ALL_SUBARRAYS_MERGED:.*]] = arith.cmpi slt, %[[SUBARRAY_SIZE]], %[[SORT_DIM]]
// CHECK:               scf.condition(%[[ARE_ALL_SUBARRAYS_MERGED]]) %[[SUBARRAY_SIZE]], %[[PARITY_]], %[[READ_BUF1]], %[[READ_BUF2]], %[[WRITE_BUF1]], %[[WRITE_BUF2]]
// CHECK:             } do {
// CHECK:             ^bb0(%[[SUBARRAY_SIZE_:[A-Za-z0-9]*]]: index,
// CHECK-SAME:             %[[PARITY__:[A-Za-z0-9]*]]: i1,
// CHECK-SAME:             %[[READ_BUF1_:[A-Za-z0-9]*]]: memref<?xf32, strided<[?], offset: ?>>,
// CHECK-SAME:             %[[READ_BUF2_:[A-Za-z0-9]*]]: memref<?xi32, strided<[?], offset: ?>>,
// CHECK-SAME:             %[[WRITE_BUF1_:[A-Za-z0-9]*]]: memref<?xf32, strided<[?], offset: ?>>,
// CHECK-SAME:             %[[WRITE_BUF2_:[A-Za-z0-9]*]]: memref<?xi32, strided<[?], offset: ?>>):
// CHECK:               %[[DOUBLE_SUBARRAY_SIZE:.*]] = arith.addi %[[SUBARRAY_SIZE_]], %[[SUBARRAY_SIZE_]]
// COM:                 // Merge all successive pairs of subarrays of maximum
// COM:                 // size SUBARRAY_SIZE.
// CHECK:               scf.for
// CHECK-SAME:              %[[DOUBLE_SUBARRAY_START:.*]] = %[[C0]] to %[[SORT_DIM]]
// CHECK-SAME:              step %[[DOUBLE_SUBARRAY_SIZE]] {
// CHECK:                 %[[SUBARRAY1_UPPER_BOUND:.*]] = arith.addi %[[DOUBLE_SUBARRAY_START]], %[[SUBARRAY_SIZE_]]
// CHECK:                 %[[SUBARRAY1_END:.*]] = arith.minsi %[[SORT_DIM]], %[[SUBARRAY1_UPPER_BOUND]]
// CHECK:                 %[[SUBARRAY2_UPPER_BOUND:.*]] = arith.addi %[[DOUBLE_SUBARRAY_START]], %[[DOUBLE_SUBARRAY_SIZE]]
// CHECK:                 %[[SUBARRAY2_END:.*]] = arith.minsi %[[SORT_DIM]], %[[SUBARRAY2_UPPER_BOUND]]
// COM:                   // Merge two subarrays together.
// CHECK:                 %[[POST_MERGE_INDICES:.*]]:3 = scf.while
// CHECK-SAME:                (%[[OUTPUT_INDEX:[A-Za-z0-9]*]] = %[[DOUBLE_SUBARRAY_START]],
// CHECK-SAME:                 %[[SUBARRAY1_INDEX:[A-Za-z0-9]*]] = %[[DOUBLE_SUBARRAY_START]],
// CHECK-SAME:                 %[[SUBARRAY2_INDEX:[A-Za-z0-9]*]] = %[[SUBARRAY1_END]])
// CHECK:                   %[[SUBARRAY1_IS_CONSUMED:.*]] = arith.cmpi slt, %[[SUBARRAY1_INDEX]], %[[SUBARRAY1_END]]
// CHECK:                   %[[SUBARRAY2_IS_CONSUMED:.*]] = arith.cmpi slt, %[[SUBARRAY2_INDEX]], %[[SUBARRAY2_END]]
// CHECK:                   %[[IS_MERGE_OVER:.*]] = arith.andi %[[SUBARRAY1_IS_CONSUMED]], %[[SUBARRAY2_IS_CONSUMED]] : i1
// CHECK:                   scf.condition(%[[IS_MERGE_OVER]]) %[[OUTPUT_INDEX]], %[[SUBARRAY1_INDEX]], %[[SUBARRAY2_INDEX]]
// CHECK:                 } do {
// CHECK:                 ^bb0(%[[OUTPUT_INDEX_:[A-Za-z0-9]*]]: index,
// CHECK-SAME:                 %[[SUBARRAY1_INDEX_:[A-Za-z0-9]*]]: index,
// CHECK-SAME:                 %[[SUBARRAY2_INDEX_:[A-Za-z0-9]*]]: index):
// CHECK:                   %[[RHS_ELEM1:.*]] = memref.load %[[READ_BUF1_]][%[[SUBARRAY1_INDEX_]]]
// CHECK:                   %[[RHS_ELEM2:.*]] = memref.load %[[READ_BUF2_]][%[[SUBARRAY1_INDEX_]]]
// CHECK:                   %[[LHS_ELEM1:.*]] = memref.load %[[READ_BUF1_]][%[[SUBARRAY2_INDEX_]]]
// CHECK:                   %[[LHS_ELEM2:.*]] = memref.load %[[READ_BUF2_]][%[[SUBARRAY2_INDEX_]]]
// CHECK:                   %[[COMPARATOR_RESULT:.*]] = arith.cmpf ogt, %[[LHS_ELEM1]], %[[RHS_ELEM1]] : f32
// CHECK:                   %[[LEFT_ELEM1:.*]] = arith.select %[[COMPARATOR_RESULT]], %[[LHS_ELEM1]], %[[RHS_ELEM1]] : f32
// CHECK:                   %[[LEFT_ELEM2:.*]] = arith.select %[[COMPARATOR_RESULT]], %[[LHS_ELEM2]], %[[RHS_ELEM2]] : i32
// CHECK:                   memref.store %[[LEFT_ELEM1]], %[[WRITE_BUF1_]][%[[OUTPUT_INDEX_]]]
// CHECK:                   memref.store %[[LEFT_ELEM2]], %[[WRITE_BUF2_]][%[[OUTPUT_INDEX_]]]
// CHECK:                   %[[SUBARRAY1_INDEX__PLUS_1:.*]] = arith.addi %[[SUBARRAY1_INDEX_]], %[[C1]]
// CHECK:                   %[[NEW_SUBARRAY1_INDEX:.*]] = arith.select %[[COMPARATOR_RESULT]], %[[SUBARRAY1_INDEX_]], %[[SUBARRAY1_INDEX__PLUS_1]]
// CHECK:                   %[[SUBARRAY2_INDEX__PLUS_1:.*]] = arith.addi %[[SUBARRAY2_INDEX_]], %[[C1]]
// CHECK:                   %[[NEW_SUBARRAY2_INDEX:.*]] = arith.select %[[COMPARATOR_RESULT]], %[[SUBARRAY2_INDEX__PLUS_1]], %[[SUBARRAY2_INDEX_]]
// CHECK:                   %[[NEW_OUTPUT_INDEX:.*]] = arith.addi %[[OUTPUT_INDEX_]], %[[C1]]
// CHECK:                   scf.yield %[[NEW_OUTPUT_INDEX]], %[[NEW_SUBARRAY1_INDEX]], %[[NEW_SUBARRAY2_INDEX]]
// CHECK:                 }
// COM:                   // After the merge, exactly one of the two subarrays
// COM:                   // contains unprocessed (and sorted) ELEMs. This
// COM:                   // appends the corresponding ELEMs to the result
// COM:                   // array.
// CHECK:                 %[[IS_SUBARRAY1_CONSUMED:.*]] = arith.cmpi slt, %[[POST_MERGE_INDICES]]#1, %[[SUBARRAY1_END]]
// CHECK:                 %[[INDEX_TO_UNPROCESSED_ELEMS:.*]] = arith.select %[[IS_SUBARRAY1_CONSUMED]], %[[POST_MERGE_INDICES]]#1, %[[POST_MERGE_INDICES]]#2
// CHECK:                 %[[UNPROCESSED_SUBARRAY_END:.*]] = arith.select %[[IS_SUBARRAY1_CONSUMED]], %[[SUBARRAY1_END]], %[[SUBARRAY2_END]]
// CHECK:                 %[[NUMBER_OF_UNPROCESSED_ELEMS:.*]] = arith.subi %[[UNPROCESSED_SUBARRAY_END]], %[[INDEX_TO_UNPROCESSED_ELEMS]]
// CHECK:                 scf.for
// CHECK-SAME:                %[[I_:.*]] = %[[C0]] to %[[NUMBER_OF_UNPROCESSED_ELEMS]]
// CHECK-SAME:                step %[[C1]] {
// CHECK:                   %[[UNPROCESSED_ELEM_INDEX:.*]] = arith.addi %[[INDEX_TO_UNPROCESSED_ELEMS]], %[[I_]]
// CHECK:                   %[[OUTPUT_INDEX__:.*]] = arith.addi %[[POST_MERGE_INDICES]]#0, %[[I_]]
// CHECK:                   %[[UNPROCESSED_ELEM1:.*]] = memref.load %[[READ_BUF1_]][%[[UNPROCESSED_ELEM_INDEX]]]
// CHECK:                   %[[UNPROCESSED_ELEM2:.*]] = memref.load %[[READ_BUF2_]][%[[UNPROCESSED_ELEM_INDEX]]]
// CHECK:                   memref.store %[[UNPROCESSED_ELEM1]], %[[WRITE_BUF1_]][%[[OUTPUT_INDEX__]]]
// CHECK:                   memref.store %[[UNPROCESSED_ELEM2]], %[[WRITE_BUF2_]][%[[OUTPUT_INDEX__]]]
// CHECK:                 }
// CHECK:               }
// CHECK:               %[[NEW_PARITY:.*]] = arith.subi %[[CTRUE]], %[[PARITY__]] : i1
// CHECK:               scf.yield %[[DOUBLE_SUBARRAY_SIZE]], %[[NEW_PARITY]], %[[WRITE_BUF1_]], %[[WRITE_BUF2_]], %[[READ_BUF1_]], %[[READ_BUF2_]]
// CHECK:             }
// CHECK:             scf.yield %[[MERGE_RESULTS]]#1 : i1
// CHECK:           }
// CHECK:           scf.if %[[PARITY]] {
// CHECK:             memref.copy %[[SCRATCH1]], %[[INIT1]]
// CHECK:             memref.copy %[[SCRATCH2]], %[[INIT2]]
// CHECK:           }
// CHECK:           return
// CHECK:         }
