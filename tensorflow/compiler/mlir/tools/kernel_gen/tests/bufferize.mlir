// RUN: kernel-gen-opt %s --func-bufferize --final-bufferize | FileCheck %s --check-prefixes=CHECK,ALLOC
// RUN: kernel-gen-opt %s --func-bufferize --final-bufferize --promote-buffers-to-stack | FileCheck %s  --check-prefixes=CHECK,ALLOCA


// CHECK-LABEL: @tensor.extract
// CHECK-SAME: (%[[ARG:.*]]: memref<?xf32>) -> f32
func @tensor.extract(%arg : tensor<?xf32>) -> f32 {
  // CHECK: %[[C0:.*]] = constant 0 : index
  // CHECK: %[[RESULT:.*]] = load %[[ARG]][%[[C0]]]
  // CHECK: return %[[RESULT]]
  %c0 = constant 0 : index
  %result = tensor.extract %arg[%c0] : tensor<?xf32>
  return %result : f32
}

// CHECK-LABEL: @tensor.from_elements
// CHECK-SAME: (%[[A:.*]]: f32) -> f32
func @tensor.from_elements(%a : f32) -> f32 {
  // CHECK: %[[B:.*]] = constant 1.2
  // CHECK: %[[C:.*]] = constant 2.3
  // ALLOC: %[[MEM:.*]] = alloc() : memref<3xf32>
  // ALLOCA: %[[MEM:.*]] = alloca() : memref<3xf32>
  // CHECK: %[[C0:.*]] = constant 0 : index
  // CHECK: store %[[A]], %[[MEM]][%[[C0]]] : memref<3xf32>
  // CHECK: %[[C1:.*]] = constant 1 : index
  // CHECK: store %[[B]], %[[MEM]][%[[C1]]] : memref<3xf32>
  // CHECK: %[[C2:.*]] = constant 2 : index
  // CHECK: store %[[C]], %[[MEM]][%[[C2]]] : memref<3xf32>
  %b = constant 1.2 : f32
  %c = constant 2.3 : f32
  %tfe = tensor.from_elements %a, %b, %c : tensor<3xf32>
  %c0 = constant 0 : index
  %result = tensor.extract %tfe[%c0] : tensor<3xf32>
  return %result : f32
}

// CHECK-LABEL: @tensor.generate
// CHECK-SAME: (%[[ARG:.*]]: memref<*xf32>) -> index
func @tensor.generate(%arg : tensor<*xf32>) -> index {
  // CHECK: %[[SIZE:.*]] = rank %[[ARG]] : memref<*xf32>
  // ALLOC: %[[MEM:.*]] = alloc(%[[SIZE]]) : memref<?xindex>
  // ALLOCA: %[[MEM:.*]] = alloca(%[[SIZE]]) : memref<?xindex>
  // CHECK: %[[C0:.*]] = constant 0 : index
  // CHECK: %[[C1:.*]] = constant 1 : index
  // CHECK: scf.parallel (%[[I:.*]]) = (%[[C0]]) to (%[[SIZE]]) step (%[[C1]]) {
  // CHECK:   %[[ELEM:.*]] = dim %[[ARG]], %[[I]] : memref<*xf32>
  // CHECK:   store %[[ELEM]], %[[MEM]][%[[I]]] : memref<?xindex>
  // CHECK:   scf.yield
  // CHECK: }
  %size = rank %arg : tensor<*xf32>
  %tfe = tensor.generate %size {
  ^bb0(%i : index):
    %elem = dim %arg, %i : tensor<*xf32>
    tensor.yield %elem : index
  } : tensor<?xindex>
  %c0 = constant 0 : index
  %result = tensor.extract %tfe[%c0] : tensor<?xindex>
  return %result : index
}

// CHECK-LABEL: @assuming
// CHECK-SAME: (%[[WITNESS:.*]]: !shape.witness, %[[ARG:.*]]: memref<?xf32>)
// CHECK-SAME: -> memref<?xf32>
func @assuming(%witness: !shape.witness, %arg : memref<?xf32>)
              -> tensor<?xf32> {
  // CHECK-NEXT: %[[ASSUMING_RESULT:.*]] = shape.assuming %[[WITNESS]]
  // CHECK-SAME:     -> (memref<?xf32>) {
  // CHECK-NEXT:   shape.assuming_yield %[[ARG]] : memref<?xf32>
  // CHECK-NEXT: }
  // CHECK-NEXT: return %[[ASSUMING_RESULT]] : memref<?xf32>
  %assuming_result = shape.assuming %witness -> (tensor<?xf32>) {
    %result = tensor_load %arg : memref<?xf32>
    shape.assuming_yield %result : tensor<?xf32>
  }
  return %assuming_result : tensor<?xf32>
}

// CHECK-LABEL: @const
// CHECK-SAME: -> memref<3xf32>
func @const() -> tensor<3xf32> {
  // CHECK: %[[MEM:.*]] = alloca() : memref<3xf32>
  // CHECK: %[[C4:.*]] = constant 4.000000e+00 : f32
  // CHECK: %[[C0:.*]] = constant 0 : index
  // CHECK: store %[[C4]], %[[MEM]][%[[C0]]] : memref<3xf32>
  // CHECK: %[[C5:.*]] = constant 5.000000e+00 : f32
  // CHECK: %[[C1:.*]] = constant 1 : index
  // CHECK: store %[[C5]], %[[MEM]][%[[C1]]] : memref<3xf32>
  // CHECK: %[[C6:.*]] = constant 6.000000e+00 : f32
  // CHECK: %[[C2:.*]] = constant 2 : index
  // CHECK: store %[[C6]], %[[MEM]][%[[C2]]] : memref<3xf32>
  // CHECK-NEXT: return %[[MEM]] : memref<3xf32>
  %result = constant dense<[4.0, 5.0, 6.0]> : tensor<3xf32>
  return %result : tensor<3xf32>
}

// CHECK-LABEL: @const_splat
// CHECK-SAME: -> memref<3xf32>
func @const_splat() -> tensor<3xf32> {
  // CHECK: %[[MEM:.*]] = alloca() : memref<3xf32>
  // CHECK: %[[C4:.*]] = constant 4.000000e+00 : f32
  // CHECK: %[[C0:.*]] = constant 0 : index
  // CHECK: store %[[C4]], %[[MEM]][%[[C0]]] : memref<3xf32>
  // CHECK: %[[C1:.*]] = constant 1 : index
  // CHECK: store %[[C4]], %[[MEM]][%[[C1]]] : memref<3xf32>
  // CHECK: %[[C2:.*]] = constant 2 : index
  // CHECK: store %[[C4]], %[[MEM]][%[[C2]]] : memref<3xf32>
  // CHECK-NEXT: return %[[MEM]] : memref<3xf32>
  %result = constant dense<4.0> : tensor<3xf32>
  return %result : tensor<3xf32>
}

// CHECK-LABEL: @minimum_broadcast_shapes
// CHECK-SAME: (%[[LHS:.*]]: memref<?xindex>, %[[RHS:.*]]: memref<?xindex>)
func @minimum_broadcast_shapes(%lhs: tensor<?xindex>, %rhs: tensor<?xindex>) -> (tensor<?xindex>, tensor<?xindex>) {
  // CHECK-NEXT: %[[C0:.*]] = constant 0 : index
  // CHECK-NEXT: %[[RANK_LHS:.*]] = dim %[[LHS]], %[[C0]] : memref<?xindex>
  // CHECK-NEXT: %[[TRUE:.*]] = constant true
  // CHECK-NEXT: %[[C0_0:.*]] = constant 0 : index
  // CHECK-NEXT: %[[C1:.*]] = constant 1 : index
  // CHECK-NEXT: %[[FOR_0:.*]]:2 = scf.for %[[IV:.*]] = %[[C0_0]] to %[[RANK_LHS]] step %[[C1]] iter_args(%[[ALL_ONES:.*]] = %[[TRUE]], %[[ONE_COUNT:.*]] = %[[C0_0]]) -> (i1, index) {
  // CHECK-NEXT:   %[[SIZE:.*]] = load %[[LHS]][%[[IV]]] : memref<?xindex>
  // CHECK-NEXT:   %[[IS_ONE:.*]] = cmpi eq, %[[SIZE]], %[[C1]] : index
  // CHECK-NEXT:   %[[NEXT_ALL_ONES:.*]] = and %[[ALL_ONES]], %[[IS_ONE]] : i1
  // CHECK-NEXT:   %[[ONE_COUNT_PLUS_ONE:.*]] = addi %[[ONE_COUNT]], %[[C1]] : index
  // CHECK-NEXT:   %[[NEXT_ONE_COUNT:.*]] = select %[[NEXT_ALL_ONES]], %[[ONE_COUNT_PLUS_ONE]], %[[ONE_COUNT]] : index
  // CHECK-NEXT:   scf.yield %[[NEXT_ALL_ONES]], %[[NEXT_ONE_COUNT]] : i1, index
  // CHECK-NEXT: }
  // CHECK-NEXT: %[[REDUCED_RANK_LHS:.*]] = subi %[[RANK_LHS]], %[[FOR_0]]#1 : index
  // CHECK-NEXT: %[[RANK_RHS:.*]] = dim %[[RHS]], %[[C0]] : memref<?xindex>
  //      CHECK: %[[REDUCED_RANK_RHS:.*]] = subi %[[RANK_RHS]], %[[FOR_1:.*]]#1 : index
  // CHECK-NEXT: %[[IS_GREATER_RANK:.*]] = cmpi ugt, %[[REDUCED_RANK_RHS]], %[[REDUCED_RANK_LHS]] : index
  // CHECK-NEXT: %[[MAX_RANK:.*]] = select %[[IS_GREATER_RANK]], %[[REDUCED_RANK_RHS]], %[[REDUCED_RANK_LHS]] : index
  // CHECK-NEXT: %[[C1_1:.*]] = constant 1 : index
  // CHECK-NEXT: %[[RESULT_LHS:.*]] = alloca(%[[REDUCED_RANK_LHS]]) : memref<?xindex>
  // CHECK-NEXT: scf.for %[[IV_LHS:.*]] = %[[C0]] to %[[REDUCED_RANK_LHS]] step %[[C1_1]] {
  // CHECK-NEXT:   store %[[C1_1]], %[[RESULT_LHS]][%[[IV_LHS]]] : memref<?xindex>
  // CHECK-NEXT: }
  // CHECK-NEXT: %[[RESULT_RHS:.*]] = alloca(%[[REDUCED_RANK_RHS]]) : memref<?xindex>
  // CHECK-NEXT: scf.for %[[IV_RHS:.*]] = %[[C0]] to %[[REDUCED_RANK_RHS]] step %[[C1_1]] {
  // CHECK-NEXT:   store %[[C1_1]], %[[RESULT_RHS]][%[[IV_RHS]]] : memref<?xindex>
  // CHECK-NEXT:  }
  // CHECK-NEXT: %[[C2:.*]] = constant 2 : index
  // CHECK-NEXT: %[[UPPER_BOUND:.*]] = addi %[[MAX_RANK]], %[[C2]] : index
  // CHECK-NEXT: %[[FALSE:.*]] = constant false
  // CHECK-NEXT: %[[MAIN_FOR:.*]]:5 = scf.for %[[IV:.*]] = %[[C1_1]] to %[[UPPER_BOUND]] step %[[C1_1]]
  // CHECK-SAME:     iter_args(%[[BC0:.*]] = %[[FALSE]], %[[BC1:.*]] = %[[FALSE]], %[[RUNNING_PRODUCT:.*]] = %[[C1_1]], %[[OFFSET:.*]] = %[[C0]], %[[INVALID:.*]] = %[[FALSE]]) -> (i1, i1, index, index, i1) {

  // First shape.
  // CHECK-NEXT:   %[[IS_OUT_OF_BOUNDS:.*]] = cmpi ult, %[[REDUCED_RANK_LHS]], %[[IV]] : index
  // CHECK-NEXT:   %[[DIMENSION:.*]] = subi %[[RANK_LHS]], %[[IV]] : index
  // CHECK-NEXT:   %[[RESULT_DIMENSION0:.*]] = subi %[[DIMENSION]], %[[FOR_0]]#1 : index
  // CHECK-NEXT:   %[[CURRENT_SIZE:.*]] = scf.if %[[IS_OUT_OF_BOUNDS]] -> (index) {
  // CHECK-NEXT:     scf.yield %[[C1_1]] : index
  // CHECK-NEXT:   } else {
  // CHECK-NEXT:     %[[SIZE:.*]] = load %[[LHS]][%[[DIMENSION]]] : memref<?xindex>
  // CHECK-NEXT:     scf.yield %[[SIZE]] : index
  // CHECK-NEXT:   }
  // CHECK-NEXT:   %[[CURRENT_SIZE_NOT_ONE0:.*]] = cmpi ne, %[[CURRENT_SIZE]], %[[C1_1]] : index
  // CHECK-NEXT:   %[[NEW_SAME_SIZE:.*]] = select %[[CURRENT_SIZE_NOT_ONE0]], %[[CURRENT_SIZE]], %[[C1_1]] : index
  // CHECK-NEXT:   %[[SAME_SIZE_WAS_NOT_ONE:.*]] = cmpi ne, %[[C1_1]], %[[C1_1]] : index
  // CHECK-NEXT:   %[[IS_DIFFERENT_SIZE:.*]] = cmpi ne, %[[C1_1]], %[[NEW_SAME_SIZE]] : index
  // CHECK-NEXT:   %[[IS_INVALID:.*]] = and %[[SAME_SIZE_WAS_NOT_ONE]], %[[IS_DIFFERENT_SIZE]] : i1
  // CHECK-NEXT:   %[[HAS_INVALID_BROADCAST:.*]] = or %[[FALSE]], %[[IS_INVALID]] : i1

  // Second shape.
  // CHECK-NEXT:   %[[IS_OUT_OF_BOUNDS:.*]] = cmpi ult, %[[REDUCED_RANK_RHS]], %[[IV]] : index
  // CHECK-NEXT:   %[[DIMENSION:.*]] = subi %[[RANK_RHS]], %[[IV]] : index
  // CHECK-NEXT:   %[[RESULT_DIMENSION1:.*]] = subi %[[DIMENSION]], %[[FOR_1]]#1 : index
  // CHECK-NEXT:   %[[CURRENT_SIZE:.*]] = scf.if %[[IS_OUT_OF_BOUNDS]] -> (index) {
  // CHECK-NEXT:     scf.yield %[[C1_1]] : index
  // CHECK-NEXT:   } else {
  // CHECK-NEXT:     %[[SIZE:.*]] = load %[[RHS]][%[[DIMENSION]]] : memref<?xindex>
  // CHECK-NEXT:     scf.yield %[[SIZE]] : index
  // CHECK-NEXT:   }
  // CHECK-NEXT:   %[[CURRENT_SIZE_NOT_ONE1:.*]] = cmpi ne, %[[CURRENT_SIZE]], %[[C1_1]] : index
  // CHECK-NEXT:   %[[NEW_NEW_SAME_SIZE:.*]] = select %[[CURRENT_SIZE_NOT_ONE1]], %[[CURRENT_SIZE]], %[[NEW_SAME_SIZE]] : index
  // CHECK-NEXT:   %[[SAME_SIZE_WAS_NOT_ONE:.*]] = cmpi ne, %[[NEW_SAME_SIZE]], %[[C1_1]] : index
  // CHECK-NEXT:   %[[IS_DIFFERENT_SIZE:.*]] = cmpi ne, %[[NEW_SAME_SIZE]], %[[NEW_NEW_SAME_SIZE]] : index
  // CHECK-NEXT:   %[[IS_INVALID:.*]] = and %[[SAME_SIZE_WAS_NOT_ONE]], %[[IS_DIFFERENT_SIZE]] : i1
  // CHECK-NEXT:   %[[NEW_HAS_INVALID_BROADCAST:.*]] = or %[[HAS_INVALID_BROADCAST]], %[[IS_INVALID]] : i1

  // CHECK-NEXT:   %[[SAME_SIZE_IS_ONE:.*]] = cmpi eq, %[[NEW_NEW_SAME_SIZE]], %[[C1_1]] : index
  // CHECK-NEXT:   %[[NO_BROADCASTING_0:.*]] = select %[[SAME_SIZE_IS_ONE]], %[[BC0]], %[[CURRENT_SIZE_NOT_ONE0]] : i1
  // CHECK-NEXT:   %[[BCASTING_IS_DIFFERENT0:.*]] = cmpi ne, %[[BC0]], %[[NO_BROADCASTING_0]] : i1
  // CHECK-NEXT:   %[[DIFFERENT_SET0:.*]] = or %[[FALSE]], %[[BCASTING_IS_DIFFERENT0]] : i1
  // CHECK-NEXT:   %[[NO_BROADCASTING_1:.*]] = select %[[SAME_SIZE_IS_ONE]], %[[BC1]], %[[CURRENT_SIZE_NOT_ONE1]] : i1
  // CHECK-NEXT:   %[[BCASTING_IS_DIFFERENT1:.*]] = cmpi ne, %[[BC1]], %[[NO_BROADCASTING_1]] : i1
  // CHECK-NEXT:   %[[DIFFERENT_SET1:.*]] = or %[[DIFFERENT_SET0]], %[[BCASTING_IS_DIFFERENT1]] : i1

  // CHECK-NEXT:   %[[LAST_ITERATION:.*]] = cmpi sgt, %[[IV]], %[[MAX_RANK]] : index
  // CHECK-NEXT:   %[[STOP_COMBINING:.*]] = or %[[LAST_ITERATION]], %[[DIFFERENT_SET1]] : i1
  // CHECK-NEXT:   %[[IF_STOP_COMBINING:.*]]:2 = scf.if %[[STOP_COMBINING]] -> (index, index) {
  // CHECK-NEXT:     %[[RUNNING_PRODUCT_NOT_ONE:.*]] = cmpi ne, %[[RUNNING_PRODUCT]], %[[C1_1]] : index
  // CHECK-NEXT:     %[[NEW_DIMENSION_OFFSET:.*]] = scf.if %[[RUNNING_PRODUCT_NOT_ONE]] -> (index) {
  // CHECK-NEXT:       %[[NEW_DIM_OFFSET:.*]] = addi %[[OFFSET]], %[[C1_1]] : index
  // CHECK-NEXT:       %[[MINUS_ONE:.*]] = constant -1 : index
  // CHECK-NEXT:       %[[WAS_IN_BOUNDS0:.*]] = cmpi sge, %[[RESULT_DIMENSION0]], %[[MINUS_ONE]] : index
  // CHECK-NEXT:       %[[SHOULD_STORE_DIM:.*]] = or %[[WAS_IN_BOUNDS0]], %[[BC0]] : i1
  // CHECK-NEXT:       scf.if %[[SHOULD_STORE_DIM]] {
  // CHECK-NEXT:         %[[OUTPUT_DIM:.*]] = subi %[[REDUCED_RANK_LHS]], %[[NEW_DIM_OFFSET]] : index
  // CHECK-NEXT:         %[[OUTPUT_SIZE:.*]] = select %[[BC0]], %[[RUNNING_PRODUCT]], %[[C1_1]] : index
  // CHECK-NEXT:         store %[[OUTPUT_SIZE]], %[[RESULT_LHS]][%[[OUTPUT_DIM]]] : memref<?xindex>
  // CHECK-NEXT:       }
  // CHECK-NEXT:       %[[WAS_IN_BOUNDS1:.*]] = cmpi sge, %[[RESULT_DIMENSION1]], %[[MINUS_ONE]] : index
  // CHECK-NEXT:       %[[SHOULD_STORE_DIM:.*]] = or %[[WAS_IN_BOUNDS1]], %[[BC1]] : i1
  // CHECK-NEXT:       scf.if %[[SHOULD_STORE_DIM]] {
  // CHECK-NEXT:         %[[OUTPUT_DIM:.*]] = subi %[[REDUCED_RANK_RHS]], %[[NEW_DIM_OFFSET]] : index
  // CHECK-NEXT:         %[[OUTPUT_SIZE:.*]] = select %[[BC1]], %[[RUNNING_PRODUCT]], %[[C1_1]] : index
  // CHECK-NEXT:         store %[[OUTPUT_SIZE]], %[[RESULT_RHS]][%[[OUTPUT_DIM]]] : memref<?xindex>
  // CHECK-NEXT:       }
  // CHECK-NEXT:       scf.yield %[[NEW_DIM_OFFSET]] : index
  // CHECK-NEXT:     } else {
  // CHECK-NEXT:       scf.yield %[[OFFSET]] : index
  // CHECK-NEXT:     }
  // CHECK-NEXT:     scf.yield %[[NEW_NEW_SAME_SIZE]], %[[NEW_DIMENSION_OFFSET]] : index, index
  // CHECK-NEXT:   } else {
  // CHECK-NEXT:     %[[NEW_PRODUCT:.*]] = muli %[[RUNNING_PRODUCT]], %[[NEW_NEW_SAME_SIZE]] : index
  // CHECK-NEXT:     scf.yield %[[NEW_PRODUCT]], %[[OFFSET]] : index, index
  // CHECK-NEXT:   }
  // CHECK-NEXT:   %[[NEW_INVALID:.*]] = or %[[INVALID]], %[[NEW_HAS_INVALID_BROADCAST]] : i1
  // CHECK-NEXT:   scf.yield %[[NO_BROADCASTING_0]], %[[NO_BROADCASTING_1]], %[[IF_STOP_COMBINING]]#0, %[[IF_STOP_COMBINING]]#1, %[[NEW_INVALID]] : i1, i1, index, index, i1
  // CHECK-NEXT: }
  %0, %1 = chlo.minimum_broadcast_shapes %lhs, %rhs :
      tensor<?xindex>, tensor<?xindex> -> tensor<?xindex>, tensor<?xindex>
  return %0, %1 : tensor<?xindex>, tensor<?xindex>
}
