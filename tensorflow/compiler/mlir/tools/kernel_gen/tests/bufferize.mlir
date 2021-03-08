// RUN: kernel-gen-opt %s --func-bufferize --final-bufferize | FileCheck %s  --check-prefixes=CHECK,ALLOC
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
  //      CHECK: %[[REDUCED_RANK_RHS:.*]] = subi %[[RANK_RHS]], %[[LEADING_ONES:.*]]#1 : index
  // CHECK-NEXT: %[[IS_GREATER_RANK:.*]] = cmpi ugt, %[[REDUCED_RANK_RHS]], %[[REDUCED_RANK_LHS]] : index
  // CHECK-NEXT: %[[MAX_RANK:.*]] = select %[[IS_GREATER_RANK]], %[[REDUCED_RANK_RHS]], %[[REDUCED_RANK_LHS]] : index
  // CHECK-NEXT: %[[C1_1:.*]] = constant 1 : index
  // CHECK-NEXT: %[[RESULT_LHS:.*]] = alloca(%[[REDUCED_RANK_LHS]]) : memref<?xindex>
  // CHECK-NEXT: scf.for %[[IV:.*]] = %[[C0]] to %[[REDUCED_RANK_LHS]] step %[[C1_1]] {
  // CHECK-NEXT:   store %[[C1_1]], %[[RESULT_LHS]][%[[IV]]] : memref<?xindex>
  // CHECK-NEXT: }
  // CHECK-NEXT: %[[RESULT_RHS:.*]] = alloca(%[[REDUCED_RANK_RHS]]) : memref<?xindex>
  //      CHECK: %[[C2:.*]] = constant 2 : index
  // CHECK-NEXT: %[[UPPER_BOUND:.*]] = addi %[[MAX_RANK]], %[[C2]] : index
  // CHECK-NEXT: %[[MAIN_FOR:.*]]:2 = scf.for %[[IV:.*]] = %[[C1_1]] to %[[UPPER_BOUND]] step %[[C1_1]] iter_args(%[[RUNNING_PRODUCT:.*]] = %[[C1_1]], %[[OFFSET:.*]] = %[[C0]]) -> (index, index) {
  // CHECK-NEXT:   %[[FALSE:.*]] = constant false
  // CHECK-NEXT:   %[[MINUS_ONE:.*]] = constant -1 : index
  // CHECK-NEXT:   %[[IS_OUT_OF_BOUNDS:.*]] = cmpi ult, %[[REDUCED_RANK_LHS]], %[[IV]] : index
  // CHECK-NEXT:   %[[DIMENSION:.*]] = subi %[[RANK_LHS]], %[[IV]] : index
  // CHECK-NEXT:   %[[RESULT_DIMENSION:.*]] = subi %[[DIMENSION]], %[[FOR_0]]#1 : index
  // CHECK-NEXT:   %[[CURRENT_SIZE:.*]] = scf.if %[[IS_OUT_OF_BOUNDS]] -> (index) {
  // CHECK-NEXT:     scf.yield %[[MINUS_ONE]] : index
  // CHECK-NEXT:   } else {
  // CHECK-NEXT:     %[[SIZE:.*]] = load %[[LHS]][%[[DIMENSION]]] : memref<?xindex>
  // CHECK-NEXT:     scf.yield %[[SIZE]] : index
  // CHECK-NEXT:   }
  // CHECK-NEXT:   %[[IS_INITIALIZED:.*]] = cmpi ne, %[[MINUS_ONE]], %[[MINUS_ONE]] : index
  // CHECK-NEXT:   %[[SAME_SIZE:.*]] = select %[[IS_INITIALIZED]], %[[MINUS_ONE]], %[[CURRENT_SIZE]] : index
  // CHECK-NEXT:   %[[IS_DIFFERENT_SIZE:.*]] = cmpi ne, %[[CURRENT_SIZE]], %[[SAME_SIZE]] : index
  // CHECK-NEXT:   %[[NEW_SAME_SIZE:.*]] = select %[[IS_DIFFERENT_SIZE]], %[[CURRENT_SIZE]], %[[SAME_SIZE]] : index
  // CHECK-NEXT:   %[[DIFFERENT_SIZES:.*]] = or %[[FALSE]], %[[IS_DIFFERENT_SIZE]] : i1
  // CHECK-NEXT:   %[[IS_ONE_OUT_OF_BOUNDS:.*]] = cmpi eq, %[[RESULT_DIMENSION]], %[[MINUS_ONE]] : index
  // CHECK-NEXT:   %[[JUST_OUT_OF_BOUNDS:.*]] = or %[[FALSE]], %[[IS_ONE_OUT_OF_BOUNDS]] : i1
  //      CHECK:   %[[IS_INITIALIZED:.*]] = cmpi ne, %[[NEW_SAME_SIZE]], %[[MINUS_ONE]] : index
  // CHECK-NEXT:   %[[SAME_SIZE:.*]] = select %[[IS_INITIALIZED]], %[[NEW_SAME_SIZE]], %[[CURRENT_SIZE_1:.*]] : index
  // CHECK-NEXT:   %[[IS_DIFFERENT_SIZE:.*]] = cmpi ne, %[[CURRENT_SIZE_1]], %[[SAME_SIZE]] : index
  // CHECK-NEXT:   %[[FINAL_SAME_SIZE:.*]] = select %[[IS_DIFFERENT_SIZE]], %[[CURRENT_SIZE_1]], %[[SAME_SIZE]] : index
  //      CHECK:   %[[FINAL_DIFFERENT_SIZES:.*]] = or %[[DIFFERENT_SIZES]], %[[IS_DIFFERENT_SIZE:.*]] : i1
  //      CHECK:   %[[FINAL_JUST_OUT_OF_BOUNDS:.*]] = or %[[JUST_OUT_OF_BOUNDS]], %[[IS_ONE_OUT_OF_BOUNDS:.*]] : i1
  // CHECK-NEXT:   %[[STOP_COMBINING_DIMENSIONS:.*]] = or %[[FINAL_DIFFERENT_SIZES]], %[[FINAL_JUST_OUT_OF_BOUNDS]] : i1
  // CHECK-NEXT:   %[[IF_STOP_COMBINING_DIMENSIONS:.*]]:2 = scf.if %[[STOP_COMBINING_DIMENSIONS]] -> (index, index) {
  // CHECK-NEXT:     %[[IS_RUNNING_PRODUCT_NOT_ONE:.*]] = cmpi ne, %[[RUNNING_PRODUCT]], %[[C1_1]] : index
  // CHECK-NEXT:     %[[NEW_OFFSET_1:.*]] = scf.if %[[IS_RUNNING_PRODUCT_NOT_ONE]] -> (index) {
  // CHECK-NEXT:       %[[NEW_OFFSET_0:.*]] = addi %[[OFFSET]], %[[C1_1]] : index
  // CHECK-NEXT:       %[[WAS_IN_BOUNDS:.*]] = cmpi sge, %[[RESULT_DIMENSION]], %[[MINUS_ONE]] : index
  // CHECK-NEXT:       scf.if %[[WAS_IN_BOUNDS]] {
  // CHECK-NEXT:         %[[CURRENT_DIMENSION:.*]] = subi %[[REDUCED_RANK_LHS]], %[[NEW_OFFSET_0]] : index
  // CHECK-NEXT:         store %[[RUNNING_PRODUCT]], %[[RESULT_LHS]][%[[CURRENT_DIMENSION]]] : memref<?xindex>
  // CHECK-NEXT:       }
  //      CHECK:       scf.yield %[[NEW_OFFSET_0]] : index
  // CHECK-NEXT:     } else {
  // CHECK-NEXT:       scf.yield %[[OFFSET]] : index
  // CHECK-NEXT:     }
  // CHECK-NEXT:     %[[IF_DIFFERENT_SIZES:.*]]:2 = scf.if %[[FINAL_DIFFERENT_SIZES]] -> (index, index) {
  // CHECK-NEXT:       %[[NEW_OFFSET_2:.*]] = addi %[[NEW_OFFSET_1]], %[[C1_1]] : index
  // CHECK-NEXT:       %[[IS_IN_BOUNDS:.*]] = cmpi sge, %[[RESULT_DIMENSION]], %[[C0]] : index
  // CHECK-NEXT:       scf.if %[[IS_IN_BOUNDS]] {
  // CHECK-NEXT:         %[[CURRENT_DIMENSION:.*]] = subi %[[REDUCED_RANK_LHS]], %[[NEW_OFFSET_2]] : index
  // CHECK-NEXT:         store %[[CURRENT_SIZE]], %[[RESULT_LHS]][%[[CURRENT_DIMENSION]]] : memref<?xindex>
  // CHECK-NEXT:       }
  //      CHECK:       scf.yield %[[C1_1]], %[[NEW_OFFSET_2]] : index, index
  // CHECK-NEXT:     } else {
  // CHECK-NEXT:       scf.yield %[[FINAL_SAME_SIZE]], %[[NEW_OFFSET_1]] : index, index
  // CHECK-NEXT:     }
  // CHECK-NEXT:     scf.yield %[[IF_DIFFERENT_SIZES]]#0, %[[IF_DIFFERENT_SIZES]]#1 : index, index
  // CHECK-NEXT:   } else {
  // CHECK-NEXT:     %[[NEW_RUNNING_PRODUCT:.*]] = muli %[[RUNNING_PRODUCT]], %[[FINAL_SAME_SIZE]] : index
  // CHECK-NEXT:     scf.yield %[[NEW_RUNNING_PRODUCT]], %[[OFFSET]] : index, index
  // CHECK-NEXT:   }
  // CHECK-NEXT:   scf.yield %[[IF_STOP_COMBINING_DIMENSIONS]]#0, %[[IF_STOP_COMBINING_DIMENSIONS]]#1 : index, index
  // CHECK-NEXT: }
  //      CHECK: return %[[SUBVIEW_LHS:.*]], %[[SUBVIEW_RHS:.*]] : memref<?xindex>, memref<?xindex>
  %0, %1 = chlo.minimum_broadcast_shapes %lhs, %rhs :
      tensor<?xindex>, tensor<?xindex> -> tensor<?xindex>, tensor<?xindex>
  return %0, %1 : tensor<?xindex>, tensor<?xindex>
}
