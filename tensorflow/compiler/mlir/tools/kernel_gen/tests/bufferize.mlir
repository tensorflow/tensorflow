// RUN: kernel-gen-opt %s --computeop-and-func-bufferize --final-bufferize \
// RUN:   --split-input-file | FileCheck %s --check-prefixes=CHECK,ALLOC
// RUN: kernel-gen-opt %s --computeop-and-func-bufferize --final-bufferize \
// RUN:  --promote-buffers-to-stack --split-input-file |\
// RUN:  FileCheck %s  --check-prefixes=CHECK,ALLOCA

// CHECK-LABEL: @tensor.extract
// CHECK-SAME: (%[[ARG:.*]]: memref<?xf32>) -> f32
func @tensor.extract(%arg : tensor<?xf32>) -> f32 {
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %[[RESULT:.*]] = memref.load %[[ARG]][%[[C0]]]
  // CHECK: return %[[RESULT]]
  %c0 = arith.constant 0 : index
  %result = tensor.extract %arg[%c0] : tensor<?xf32>
  return %result : f32
}

// CHECK-LABEL: @tensor.from_elements
// CHECK-SAME: (%[[A:.*]]: f32) -> f32
func @tensor.from_elements(%a : f32) -> f32 {
  // CHECK-DAG: %[[B:.*]] = arith.constant 1.2
  // CHECK-DAG: %[[C:.*]] = arith.constant 2.3
  // ALLOC: %[[MEM:.*]] = memref.alloc() : memref<3xf32>
  // ALLOCA: %[[MEM:.*]] = memref.alloca() : memref<3xf32>
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %[[C1:.*]] = arith.constant 1 : index
  // CHECK: %[[C2:.*]] = arith.constant 2 : index
  // CHECK: store %[[A]], %[[MEM]][%[[C0]]] : memref<3xf32>
  // CHECK: store %[[B]], %[[MEM]][%[[C1]]] : memref<3xf32>
  // CHECK: store %[[C]], %[[MEM]][%[[C2]]] : memref<3xf32>
  %b = arith.constant 1.2 : f32
  %c = arith.constant 2.3 : f32
  %tfe = tensor.from_elements %a, %b, %c : tensor<3xf32>
  %c0 = arith.constant 0 : index
  %result = tensor.extract %tfe[%c0] : tensor<3xf32>
  return %result : f32
}

// CHECK-LABEL: @tensor.generate
// CHECK-SAME: (%[[ARG:.*]]: memref<*xf32>) -> index
func @tensor.generate(%arg : tensor<*xf32>) -> index {
  // CHECK: %[[SIZE:.*]] = memref.rank %[[ARG]] : memref<*xf32>
  // ALLOC: %[[MEM:.*]] = memref.alloc(%[[SIZE]]) : memref<?xindex>
  // ALLOCA: %[[MEM:.*]] = memref.alloca(%[[SIZE]]) : memref<?xindex>
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
  // CHECK: scf.parallel (%[[I:.*]]) = (%[[C0]]) to (%[[SIZE]]) step (%[[C1]]) {
  // CHECK:   %[[ELEM:.*]] = memref.dim %[[ARG]], %[[I]] : memref<*xf32>
  // CHECK:   memref.store %[[ELEM]], %[[MEM]][%[[I]]] : memref<?xindex>
  // CHECK:   scf.yield
  // CHECK: }
  %size = tensor.rank %arg : tensor<*xf32>
  %tfe = tensor.generate %size {
  ^bb0(%i : index):
    %elem = tensor.dim %arg, %i : tensor<*xf32>
    tensor.yield %elem : index
  } : tensor<?xindex>
  %c0 = arith.constant 0 : index
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
    %result = bufferization.to_tensor %arg : memref<?xf32>
    shape.assuming_yield %result : tensor<?xf32>
  }
  return %assuming_result : tensor<?xf32>
}

// CHECK-LABEL: @const
// CHECK-SAME: -> memref<3xf32>
func @const() -> tensor<3xf32> {
  // CHECK: %[[MEM:.*]] = memref.alloca() : memref<3xf32>
  // CHECK-DAG: %[[C4:.*]] = arith.constant 4.000000e+00 : f32
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: store %[[C4]], %[[MEM]][%[[C0]]] : memref<3xf32>
  // CHECK-DAG: %[[C5:.*]] = arith.constant 5.000000e+00 : f32
  // CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
  // CHECK: store %[[C5]], %[[MEM]][%[[C1]]] : memref<3xf32>
  // CHECK-DAG: %[[C6:.*]] = arith.constant 6.000000e+00 : f32
  // CHECK-DAG: %[[C2:.*]] = arith.constant 2 : index
  // CHECK: store %[[C6]], %[[MEM]][%[[C2]]] : memref<3xf32>
  // CHECK-NEXT: return %[[MEM]] : memref<3xf32>
  %result = arith.constant dense<[4.0, 5.0, 6.0]> : tensor<3xf32>
  return %result : tensor<3xf32>
}

// CHECK-LABEL: @const_splat
// CHECK-SAME: -> memref<3xf32>
func @const_splat() -> tensor<3xf32> {
  // CHECK: %[[MEM:.*]] = memref.alloca() : memref<3xf32>
  // CHECK-DAG: %[[C4:.*]] = arith.constant 4.000000e+00 : f32
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: store %[[C4]], %[[MEM]][%[[C0]]] : memref<3xf32>
  // CHECK: %[[C1:.*]] = arith.constant 1 : index
  // CHECK: store %[[C4]], %[[MEM]][%[[C1]]] : memref<3xf32>
  // CHECK: %[[C2:.*]] = arith.constant 2 : index
  // CHECK: store %[[C4]], %[[MEM]][%[[C2]]] : memref<3xf32>
  // CHECK-NEXT: return %[[MEM]] : memref<3xf32>
  %result = arith.constant dense<4.0> : tensor<3xf32>
  return %result : tensor<3xf32>
}

// CHECK-LABEL: @minimum_broadcast_shapes
// CHECK-SAME: (%[[LHS:.*]]: memref<?xindex>, %[[RHS:.*]]: memref<?xindex>)
func @minimum_broadcast_shapes(%lhs: tensor<?xindex>, %rhs: tensor<?xindex>) -> (tensor<?xindex>, tensor<?xindex>) {
  // CHECK-NEXT: %[[C0:.*]] = arith.constant 0 : index
  // CHECK-NEXT: %[[RANK_LHS:.*]] = memref.dim %[[LHS]], %[[C0]] : memref<?xindex>
  // CHECK-NEXT: %[[RANK_RHS:.*]] = memref.dim %[[RHS]], %[[C0]] : memref<?xindex>
  // CHECK-NEXT: %[[IS_GREATER_RANK:.*]] = arith.cmpi ugt, %[[RANK_RHS]], %[[RANK_LHS]] : index
  // CHECK-NEXT: %[[MAX_RANK:.*]] = select %[[IS_GREATER_RANK]], %[[RANK_RHS]], %[[RANK_LHS]] : index
  // CHECK-NEXT: %[[C1_1:.*]] = arith.constant 1 : index
  // CHECK-NEXT: %[[RESULT_LHS:.*]] = memref.alloca(%[[RANK_LHS]]) : memref<?xindex>
  // CHECK-NEXT: scf.for %[[IV_LHS:.*]] = %[[C0]] to %[[RANK_LHS]] step %[[C1_1]] {
  // CHECK-NEXT:   memref.store %[[C1_1]], %[[RESULT_LHS]][%[[IV_LHS]]] : memref<?xindex>
  // CHECK-NEXT: }
  // CHECK-NEXT: %[[RESULT_RHS:.*]] = memref.alloca(%[[RANK_RHS]]) : memref<?xindex>
  // CHECK-NEXT: scf.for %[[IV_RHS:.*]] = %[[C0]] to %[[RANK_RHS]] step %[[C1_1]] {
  // CHECK-NEXT:   memref.store %[[C1_1]], %[[RESULT_RHS]][%[[IV_RHS]]] : memref<?xindex>
  // CHECK-NEXT:  }
  // CHECK-NEXT: %[[C2:.*]] = arith.constant 2 : index
  // CHECK-NEXT: %[[UPPER_BOUND:.*]] = arith.addi %[[MAX_RANK]], %[[C2]] : index
  // CHECK-NEXT: %[[FALSE:.*]] = arith.constant false
  // CHECK-NEXT: %[[MAIN_FOR:.*]]:5 = scf.for %[[IV:.*]] = %[[C1_1]] to %[[UPPER_BOUND]] step %[[C1_1]]
  // CHECK-SAME:     iter_args(%[[BC0:.*]] = %[[FALSE]], %[[BC1:.*]] = %[[FALSE]], %[[RUNNING_PRODUCT:.*]] = %[[C1_1]], %[[OFFSET:.*]] = %[[C0]], %[[INVALID:.*]] = %[[FALSE]]) -> (i1, i1, index, index, i1) {

  // First shape.
  // CHECK-NEXT:   %[[IS_OUT_OF_BOUNDS:.*]] = arith.cmpi ult, %[[RANK_LHS]], %[[IV]] : index
  // CHECK-NEXT:   %[[DIMENSION0:.*]] = arith.subi %[[RANK_LHS]], %[[IV]] : index
  // CHECK-NEXT:   %[[CURRENT_SIZE:.*]] = scf.if %[[IS_OUT_OF_BOUNDS]] -> (index) {
  // CHECK-NEXT:     scf.yield %[[C1_1]] : index
  // CHECK-NEXT:   } else {
  // CHECK-NEXT:     %[[SIZE:.*]] = memref.load %[[LHS]][%[[DIMENSION0]]] : memref<?xindex>
  // CHECK-NEXT:     scf.yield %[[SIZE]] : index
  // CHECK-NEXT:   }
  // CHECK-NEXT:   %[[CURRENT_SIZE_NOT_ONE0:.*]] = arith.cmpi ne, %[[CURRENT_SIZE]], %[[C1_1]] : index
  // CHECK-NEXT:   %[[NEW_SAME_SIZE:.*]] = select %[[CURRENT_SIZE_NOT_ONE0]], %[[CURRENT_SIZE]], %[[C1_1]] : index
  // CHECK-NEXT:   %[[SAME_SIZE_WAS_NOT_ONE:.*]] = arith.cmpi ne, %[[C1_1]], %[[C1_1]] : index
  // CHECK-NEXT:   %[[IS_DIFFERENT_SIZE:.*]] = arith.cmpi ne, %[[C1_1]], %[[NEW_SAME_SIZE]] : index
  // CHECK-NEXT:   %[[IS_INVALID:.*]] = arith.andi %[[SAME_SIZE_WAS_NOT_ONE]], %[[IS_DIFFERENT_SIZE]] : i1
  // CHECK-NEXT:   %[[HAS_INVALID_BROADCAST:.*]] = arith.ori %[[FALSE]], %[[IS_INVALID]] : i1

  // Second shape.
  // CHECK-NEXT:   %[[IS_OUT_OF_BOUNDS:.*]] = arith.cmpi ult, %[[RANK_RHS]], %[[IV]] : index
  // CHECK-NEXT:   %[[DIMENSION1:.*]] = arith.subi %[[RANK_RHS]], %[[IV]] : index
  // CHECK-NEXT:   %[[CURRENT_SIZE:.*]] = scf.if %[[IS_OUT_OF_BOUNDS]] -> (index) {
  // CHECK-NEXT:     scf.yield %[[C1_1]] : index
  // CHECK-NEXT:   } else {
  // CHECK-NEXT:     %[[SIZE:.*]] = memref.load %[[RHS]][%[[DIMENSION1]]] : memref<?xindex>
  // CHECK-NEXT:     scf.yield %[[SIZE]] : index
  // CHECK-NEXT:   }
  // CHECK-NEXT:   %[[CURRENT_SIZE_NOT_ONE1:.*]] = arith.cmpi ne, %[[CURRENT_SIZE]], %[[C1_1]] : index
  // CHECK-NEXT:   %[[NEW_NEW_SAME_SIZE:.*]] = select %[[CURRENT_SIZE_NOT_ONE1]], %[[CURRENT_SIZE]], %[[NEW_SAME_SIZE]] : index
  // CHECK-NEXT:   %[[SAME_SIZE_WAS_NOT_ONE:.*]] = arith.cmpi ne, %[[NEW_SAME_SIZE]], %[[C1_1]] : index
  // CHECK-NEXT:   %[[IS_DIFFERENT_SIZE:.*]] = arith.cmpi ne, %[[NEW_SAME_SIZE]], %[[NEW_NEW_SAME_SIZE]] : index
  // CHECK-NEXT:   %[[IS_INVALID:.*]] = arith.andi %[[SAME_SIZE_WAS_NOT_ONE]], %[[IS_DIFFERENT_SIZE]] : i1
  // CHECK-NEXT:   %[[NEW_HAS_INVALID_BROADCAST:.*]] = arith.ori %[[HAS_INVALID_BROADCAST]], %[[IS_INVALID]] : i1

  // CHECK-NEXT:   %[[SAME_SIZE_IS_ONE:.*]] = arith.cmpi eq, %[[NEW_NEW_SAME_SIZE]], %[[C1_1]] : index
  // CHECK-NEXT:   %[[NO_BROADCASTING_0:.*]] = select %[[SAME_SIZE_IS_ONE]], %[[BC0]], %[[CURRENT_SIZE_NOT_ONE0]] : i1
  // CHECK-NEXT:   %[[BCASTING_IS_DIFFERENT0:.*]] = arith.cmpi ne, %[[BC0]], %[[NO_BROADCASTING_0]] : i1
  // CHECK-NEXT:   %[[DIFFERENT_SET0:.*]] = arith.ori %[[FALSE]], %[[BCASTING_IS_DIFFERENT0]] : i1
  // CHECK-NEXT:   %[[NO_BROADCASTING_1:.*]] = select %[[SAME_SIZE_IS_ONE]], %[[BC1]], %[[CURRENT_SIZE_NOT_ONE1]] : i1
  // CHECK-NEXT:   %[[BCASTING_IS_DIFFERENT1:.*]] = arith.cmpi ne, %[[BC1]], %[[NO_BROADCASTING_1]] : i1
  // CHECK-NEXT:   %[[DIFFERENT_SET1:.*]] = arith.ori %[[DIFFERENT_SET0]], %[[BCASTING_IS_DIFFERENT1]] : i1

  // CHECK-NEXT:   %[[LAST_ITERATION:.*]] = arith.cmpi sgt, %[[IV]], %[[MAX_RANK]] : index
  // CHECK-NEXT:   %[[STOP_COMBINING:.*]] = arith.ori %[[LAST_ITERATION]], %[[DIFFERENT_SET1]] : i1
  // CHECK-NEXT:   %[[IF_STOP_COMBINING:.*]]:2 = scf.if %[[STOP_COMBINING]] -> (index, index) {
  // CHECK-NEXT:     %[[RUNNING_PRODUCT_NOT_ONE:.*]] = arith.cmpi ne, %[[RUNNING_PRODUCT]], %[[C1_1]] : index
  // CHECK-NEXT:     %[[NEW_DIMENSION_OFFSET:.*]] = scf.if %[[RUNNING_PRODUCT_NOT_ONE]] -> (index) {
  // CHECK-NEXT:       %[[NEW_DIM_OFFSET:.*]] = arith.addi %[[OFFSET]], %[[C1_1]] : index
  // CHECK-NEXT:       %[[MINUS_ONE:.*]] = arith.constant -1 : index
  // CHECK-NEXT:       %[[WAS_IN_BOUNDS0:.*]] = arith.cmpi sge, %[[DIMENSION0]], %[[MINUS_ONE]] : index
  // CHECK-NEXT:       %[[SHOULD_STORE_DIM:.*]] = arith.ori %[[WAS_IN_BOUNDS0]], %[[BC0]] : i1
  // CHECK-NEXT:       scf.if %[[SHOULD_STORE_DIM]] {
  // CHECK-NEXT:         %[[OUTPUT_DIM:.*]] = arith.subi %[[RANK_LHS]], %[[NEW_DIM_OFFSET]] : index
  // CHECK-NEXT:         %[[OUTPUT_SIZE:.*]] = select %[[BC0]], %[[RUNNING_PRODUCT]], %[[C1_1]] : index
  // CHECK-NEXT:         memref.store %[[OUTPUT_SIZE]], %[[RESULT_LHS]][%[[OUTPUT_DIM]]] : memref<?xindex>
  // CHECK-NEXT:       }
  // CHECK-NEXT:       %[[WAS_IN_BOUNDS1:.*]] = arith.cmpi sge, %[[DIMENSION1]], %[[MINUS_ONE]] : index
  // CHECK-NEXT:       %[[SHOULD_STORE_DIM:.*]] = arith.ori %[[WAS_IN_BOUNDS1]], %[[BC1]] : i1
  // CHECK-NEXT:       scf.if %[[SHOULD_STORE_DIM]] {
  // CHECK-NEXT:         %[[OUTPUT_DIM:.*]] = arith.subi %[[RANK_RHS]], %[[NEW_DIM_OFFSET]] : index
  // CHECK-NEXT:         %[[OUTPUT_SIZE:.*]] = select %[[BC1]], %[[RUNNING_PRODUCT]], %[[C1_1]] : index
  // CHECK-NEXT:         memref.store %[[OUTPUT_SIZE]], %[[RESULT_RHS]][%[[OUTPUT_DIM]]] : memref<?xindex>
  // CHECK-NEXT:       }
  // CHECK-NEXT:       scf.yield %[[NEW_DIM_OFFSET]] : index
  // CHECK-NEXT:     } else {
  // CHECK-NEXT:       scf.yield %[[OFFSET]] : index
  // CHECK-NEXT:     }
  // CHECK-NEXT:     scf.yield %[[NEW_NEW_SAME_SIZE]], %[[NEW_DIMENSION_OFFSET]] : index, index
  // CHECK-NEXT:   } else {
  // CHECK-NEXT:     %[[NEW_PRODUCT:.*]] = arith.muli %[[RUNNING_PRODUCT]], %[[NEW_NEW_SAME_SIZE]] : index
  // CHECK-NEXT:     scf.yield %[[NEW_PRODUCT]], %[[OFFSET]] : index, index
  // CHECK-NEXT:   }
  // CHECK-NEXT:   %[[NEW_INVALID:.*]] = arith.ori %[[INVALID]], %[[NEW_HAS_INVALID_BROADCAST]] : i1
  // CHECK-NEXT:   scf.yield %[[NO_BROADCASTING_0]], %[[NO_BROADCASTING_1]], %[[IF_STOP_COMBINING]]#0, %[[IF_STOP_COMBINING]]#1, %[[NEW_INVALID]] : i1, i1, index, index, i1
  // CHECK-NEXT: }

  // Count leading ones in first result shape.
  // CHECK-NEXT: %[[TRUE:.*]] = arith.constant true
  // CHECK-NEXT: %[[C0:.*]] = arith.constant 0 : index
  // CHECK-NEXT: %[[C1:.*]] = arith.constant 1 : index
  // CHECK-NEXT: %[[FOR_0:.*]]:2 = scf.for %[[IV:.*]] = %[[C0]] to %[[RANK_LHS]] step %[[C1]] iter_args(%[[ALL_ONES:.*]] = %[[TRUE]], %[[ONE_COUNT:.*]] = %[[C0]]) -> (i1, index) {
  // CHECK-NEXT:   %[[SIZE:.*]] = memref.load %[[RESULT_LHS]][%[[IV]]] : memref<?xindex>
  // CHECK-NEXT:   %[[IS_ONE:.*]] = arith.cmpi eq, %[[SIZE]], %[[C1]] : index
  // CHECK-NEXT:   %[[NEXT_ALL_ONES:.*]] = arith.andi %[[ALL_ONES]], %[[IS_ONE]] : i1
  // CHECK-NEXT:   %[[ONE_COUNT_PLUS_ONE:.*]] = arith.addi %[[ONE_COUNT]], %[[C1]] : index
  // CHECK-NEXT:   %[[NEXT_ONE_COUNT:.*]] = select %[[NEXT_ALL_ONES]], %[[ONE_COUNT_PLUS_ONE]], %[[ONE_COUNT]] : index
  // CHECK-NEXT:   scf.yield %[[NEXT_ALL_ONES]], %[[NEXT_ONE_COUNT]] : i1, index
  // CHECK-NEXT: }

  // Copy the results with leading ones removed.
  // CHECK-NEXT: %[[REDUCED_RANK_LHS:.*]] = arith.subi %[[RANK_LHS]], %[[FOR_0]]#1 : index
  // CHECK-NEXT: %[[REDUCED_RESULT_LHS:.*]] = memref.alloca(%[[REDUCED_RANK_LHS]]) : memref<?xindex>
  // CHECK-NEXT: %[[C0:.*]] = arith.constant 0 : index
  // CHECK-NEXT: %[[C1:.*]] = arith.constant 1 : index
  // CHECK-NEXT: scf.for %[[IV:.*]] = %[[C0]] to %[[REDUCED_RANK_LHS]] step %[[C1]] {
  // CHECK-NEXT:   %[[WITH_OFFSET:.*]] = arith.addi %[[IV]], %[[FOR_0]]#1 : index
  // CHECK-NEXT:   %[[LOAD:.*]] = memref.load %[[RESULT_LHS]][%[[WITH_OFFSET]]] : memref<?xindex>
  // CHECK-NEXT:   memref.store %[[LOAD]], %[[REDUCED_RESULT_LHS]][%[[IV]]] : memref<?xindex>
  // CHECK-NEXT: }

  // Select whether to use the original shapes in case of invalid broadcasts.
  // CHECK-NEXT: %[[FINAL_RESULT_LHS:.*]] = select %[[MAIN_FOR]]#4, %[[LHS]], %[[REDUCED_RESULT_LHS]] : memref<?xindex>

  // (Testing of computing the reduced second shape result is omitted)

  // Select whether to use the original shapes in case of invalid broadcasts.
  // CHECK: %[[FINAL_RESULT_RHS:.*]] = select %[[MAIN_FOR]]#4, %[[RHS]], %[[REDUCED_RESULT_RHS:.*]] : memref<?xindex>
  %0, %1 = chlo.minimum_broadcast_shapes %lhs, %rhs :
      tensor<?xindex>, tensor<?xindex> -> tensor<?xindex>, tensor<?xindex>
  // CHECK-NEXT: return %[[FINAL_RESULT_LHS]], %[[FINAL_RESULT_RHS]] : memref<?xindex>, memref<?xindex>
  return %0, %1 : tensor<?xindex>, tensor<?xindex>
}

// CHECK-LABEL: @tensor_reshape
// CHECK-SAME: (%[[T:.*]]: memref<1x2x2xf32>)
func @tensor_reshape(%t : tensor<1x2x2xf32>) -> tensor<4xf32> {
  // CHECK: memref.collapse_shape %[[T]] {{.*}} : memref<1x2x2xf32> into memref<4xf32>
  %result = tensor.collapse_shape %t [[0, 1, 2]] : tensor<1x2x2xf32> into tensor<4xf32>
  return %result : tensor<4xf32>
}

// CHECK-LABEL: @slice
// CHECK-SAME: (%[[T:.*]]: memref<3xi32>)
func @slice(%t : tensor<3xi32>) -> tensor<1xi32> {
  // CHECK: memref.subview %[[T]][0] [1] [1] : memref<3xi32> to memref<1xi32>
  %result = tensor.extract_slice %t[0] [1] [1] : tensor<3xi32> to tensor<1xi32>
  return %result : tensor<1xi32>
}

// CHECK-LABEL: @jit_execute
// CHECK-SAME: (%[[F:.*]]: !tf_framework.jit_callable, %[[ARG:.*]]: memref<*xf32>) -> memref<*xf32>
func @jit_execute(%f : !tf_framework.jit_callable, %arg : tensor<*xf32>)
    -> tensor<*xf32> {
  // CHECK: %[[RES:.*]] = tf_framework.jit_execute %[[F]](%[[ARG]]) : memref<*xf32> -> memref<*xf32>
  // CHECK: return %[[RES]] : memref<*xf32>
  %0 = tf_framework.jit_execute %f(%arg) : tensor<*xf32> -> tensor<*xf32>
  return %0 : tensor<*xf32>
}

func @dynamic_broadcast_return(%t : tensor<?x?xf32>, %shape : tensor<2xi32>) -> tensor<?x?xf32> {
  // CHECK: memref.copy
  %bcast = "mhlo.dynamic_broadcast_in_dim"(%t, %shape) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<?x?xf32>, tensor<2xi32>) -> tensor<?x?xf32>
  return %bcast : tensor<?x?xf32>
}
