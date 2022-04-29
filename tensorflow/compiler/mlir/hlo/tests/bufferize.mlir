// RUN: mlir-hlo-opt %s --computeop-and-func-bufferize \
// RUN:    --final-bufferize=alignment=128 --split-input-file | FileCheck %s \
// RUN:    --check-prefixes=CHECK,ALLOC
// RUN: mlir-hlo-opt %s --computeop-and-func-bufferize \
// RUN:    --final-bufferize=alignment=128 --promote-buffers-to-stack \
// RUN:    --split-input-file | FileCheck %s  --check-prefixes=CHECK,ALLOCA

// CHECK-LABEL: @tensor.extract
// CHECK-SAME: (%[[ARG:.*]]: memref<?xf32>) -> f32
func.func @tensor.extract(%arg : tensor<?xf32>) -> f32 {
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %[[RESULT:.*]] = memref.load %[[ARG]][%[[C0]]]
  // CHECK: return %[[RESULT]]
  %c0 = arith.constant 0 : index
  %result = tensor.extract %arg[%c0] : tensor<?xf32>
  func.return %result : f32
}

// CHECK-LABEL: @tensor.from_elements
// CHECK-SAME: (%[[A:.*]]: f32) -> f32
func.func @tensor.from_elements(%a : f32) -> f32 {
  // CHECK-DAG: %[[B:.*]] = arith.constant 1.2
  // CHECK-DAG: %[[C:.*]] = arith.constant 2.3
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
  // CHECK-DAG: %[[C2:.*]] = arith.constant 2 : index
  // ALLOC-DAG: %[[MEM:.*]] = memref.alloc() {{.*}} : memref<3xf32>
  // ALLOCA-DAG: %[[MEM:.*]] = memref.alloca() : memref<3xf32>
  // CHECK: store %[[A]], %[[MEM]][%[[C0]]] : memref<3xf32>
  // CHECK: store %[[B]], %[[MEM]][%[[C1]]] : memref<3xf32>
  // CHECK: store %[[C]], %[[MEM]][%[[C2]]] : memref<3xf32>
  %b = arith.constant 1.2 : f32
  %c = arith.constant 2.3 : f32
  %tfe = tensor.from_elements %a, %b, %c : tensor<3xf32>
  %c0 = arith.constant 0 : index
  %result = tensor.extract %tfe[%c0] : tensor<3xf32>
  func.return %result : f32
}

// CHECK-LABEL: @tensor.generate
// CHECK-SAME: (%[[ARG:.*]]: memref<*xf32>) -> index
func.func @tensor.generate(%arg : tensor<*xf32>) -> index {
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
  // CHECK-DAG: %[[SIZE:.*]] = memref.rank %[[ARG]] : memref<*xf32>
  // ALLOC-DAG: %[[MEM:.*]] = memref.alloc(%[[SIZE]]) {{.*}} : memref<?xindex>
  // ALLOCA-DAG: %[[MEM:.*]] = memref.alloca(%[[SIZE]]) : memref<?xindex>
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
  func.return %result : index
}

// CHECK-LABEL: @assuming
// CHECK-SAME: (%[[WITNESS:.*]]: !shape.witness, %[[ARG:.*]]: memref<?xf32>)
// CHECK-SAME: -> memref<?xf32>
func.func @assuming(%witness: !shape.witness, %arg : memref<?xf32>)
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
  func.return %assuming_result : tensor<?xf32>
}

// -----

// CHECK: memref.global "private" constant @[[BUFFER:.*]] : memref<3xf32> = dense<[4.000000e+00, 5.000000e+00, 6.000000e+00]>
// CHECK-SAME: alignment = 128
// CHECK: @const
// CHECK-SAME: -> memref<3xf32>
func.func @const() -> tensor<3xf32> {
  // CHECK:  %[[RESULT:.*]] = memref.get_global @[[BUFFER]] : memref<3xf32>
  // CHECK:  return %[[RESULT]] : memref<3xf32>
  %result = arith.constant dense<[4.0, 5.0, 6.0]> : tensor<3xf32>
  func.return %result : tensor<3xf32>
}

// -----

// CHECK: memref.global "private" constant @[[BUFFER:.*]] : memref<3xf32> = dense<4.000000e+00>
// CHECK-SAME: alignment = 128
// CHECK: @const_splat
// CHECK-SAME: -> memref<3xf32>
func.func @const_splat() -> tensor<3xf32> {
  // CHECK:  %[[RESULT:.*]] = memref.get_global @[[BUFFER]] : memref<3xf32>
  // CHECK:  return %[[RESULT]] : memref<3xf32>
  %result = arith.constant dense<4.0> : tensor<3xf32>
  func.return %result : tensor<3xf32>
}

// -----

// CHECK-LABEL: @minimum_broadcast_shapes
// CHECK-SAME: (%[[LHS:.*]]: memref<?xindex>, %[[RHS:.*]]: memref<?xindex>)
func.func @minimum_broadcast_shapes(%lhs: tensor<?xindex>, %rhs: tensor<?xindex>) -> (tensor<?xindex>, tensor<?xindex>) {
  // CHECK-NEXT: %[[C0:.*]] = arith.constant 0 : index
  // CHECK-NEXT: %[[RANK_LHS:.*]] = memref.dim %[[LHS]], %[[C0]] : memref<?xindex>
  // CHECK-NEXT: %[[RANK_RHS:.*]] = memref.dim %[[RHS]], %[[C0]] : memref<?xindex>
  // CHECK-NEXT: %[[IS_GREATER_RANK:.*]] = arith.cmpi ugt, %[[RANK_RHS]], %[[RANK_LHS]] : index
  // CHECK-NEXT: %[[MAX_RANK:.*]] = arith.select %[[IS_GREATER_RANK]], %[[RANK_RHS]], %[[RANK_LHS]] : index
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
  // CHECK-NEXT:   %[[NEW_SAME_SIZE:.*]] = arith.select %[[CURRENT_SIZE_NOT_ONE0]], %[[CURRENT_SIZE]], %[[C1_1]] : index
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
  // CHECK-NEXT:   %[[NEW_NEW_SAME_SIZE:.*]] = arith.select %[[CURRENT_SIZE_NOT_ONE1]], %[[CURRENT_SIZE]], %[[NEW_SAME_SIZE]] : index
  // CHECK-NEXT:   %[[SAME_SIZE_WAS_NOT_ONE:.*]] = arith.cmpi ne, %[[NEW_SAME_SIZE]], %[[C1_1]] : index
  // CHECK-NEXT:   %[[IS_DIFFERENT_SIZE:.*]] = arith.cmpi ne, %[[NEW_SAME_SIZE]], %[[NEW_NEW_SAME_SIZE]] : index
  // CHECK-NEXT:   %[[IS_INVALID:.*]] = arith.andi %[[SAME_SIZE_WAS_NOT_ONE]], %[[IS_DIFFERENT_SIZE]] : i1
  // CHECK-NEXT:   %[[NEW_HAS_INVALID_BROADCAST:.*]] = arith.ori %[[HAS_INVALID_BROADCAST]], %[[IS_INVALID]] : i1

  // CHECK-NEXT:   %[[SAME_SIZE_IS_ONE:.*]] = arith.cmpi eq, %[[NEW_NEW_SAME_SIZE]], %[[C1_1]] : index
  // CHECK-NEXT:   %[[NO_BROADCASTING_0:.*]] = arith.select %[[SAME_SIZE_IS_ONE]], %[[BC0]], %[[CURRENT_SIZE_NOT_ONE0]] : i1
  // CHECK-NEXT:   %[[BCASTING_IS_DIFFERENT0:.*]] = arith.cmpi ne, %[[BC0]], %[[NO_BROADCASTING_0]] : i1
  // CHECK-NEXT:   %[[DIFFERENT_SET0:.*]] = arith.ori %[[FALSE]], %[[BCASTING_IS_DIFFERENT0]] : i1
  // CHECK-NEXT:   %[[NO_BROADCASTING_1:.*]] = arith.select %[[SAME_SIZE_IS_ONE]], %[[BC1]], %[[CURRENT_SIZE_NOT_ONE1]] : i1
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
  // CHECK-NEXT:         %[[OUTPUT_SIZE:.*]] = arith.select %[[BC0]], %[[RUNNING_PRODUCT]], %[[C1_1]] : index
  // CHECK-NEXT:         memref.store %[[OUTPUT_SIZE]], %[[RESULT_LHS]][%[[OUTPUT_DIM]]] : memref<?xindex>
  // CHECK-NEXT:       }
  // CHECK-NEXT:       %[[WAS_IN_BOUNDS1:.*]] = arith.cmpi sge, %[[DIMENSION1]], %[[MINUS_ONE]] : index
  // CHECK-NEXT:       %[[SHOULD_STORE_DIM:.*]] = arith.ori %[[WAS_IN_BOUNDS1]], %[[BC1]] : i1
  // CHECK-NEXT:       scf.if %[[SHOULD_STORE_DIM]] {
  // CHECK-NEXT:         %[[OUTPUT_DIM:.*]] = arith.subi %[[RANK_RHS]], %[[NEW_DIM_OFFSET]] : index
  // CHECK-NEXT:         %[[OUTPUT_SIZE:.*]] = arith.select %[[BC1]], %[[RUNNING_PRODUCT]], %[[C1_1]] : index
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
  // CHECK-NEXT:   %[[NEXT_ONE_COUNT:.*]] = arith.select %[[NEXT_ALL_ONES]], %[[ONE_COUNT_PLUS_ONE]], %[[ONE_COUNT]] : index
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
  // CHECK-NEXT: %[[FINAL_RESULT_LHS:.*]] = arith.select %[[MAIN_FOR]]#4, %[[LHS]], %[[REDUCED_RESULT_LHS]] : memref<?xindex>

  // (Testing of computing the reduced second shape result is omitted)

  // Select whether to use the original shapes in case of invalid broadcasts.
  // CHECK: %[[FINAL_RESULT_RHS:.*]] = arith.select %[[MAIN_FOR]]#4, %[[RHS]], %[[REDUCED_RESULT_RHS:.*]] : memref<?xindex>
  %0, %1 = chlo.minimum_broadcast_shapes %lhs, %rhs :
      tensor<?xindex>, tensor<?xindex> -> tensor<?xindex>, tensor<?xindex>
  // CHECK-NEXT: return %[[FINAL_RESULT_LHS]], %[[FINAL_RESULT_RHS]] : memref<?xindex>, memref<?xindex>
  func.return %0, %1 : tensor<?xindex>, tensor<?xindex>
}

// CHECK-LABEL: @tensor_reshape
// CHECK-SAME: (%[[T:.*]]: memref<1x2x2xf32>)
func.func @tensor_reshape(%t : tensor<1x2x2xf32>) -> tensor<4xf32> {
  // CHECK: memref.collapse_shape %[[T]] {{.*}} : memref<1x2x2xf32> into memref<4xf32>
  %result = tensor.collapse_shape %t [[0, 1, 2]] : tensor<1x2x2xf32> into tensor<4xf32>
  func.return %result : tensor<4xf32>
}

// CHECK-LABEL: @slice
// CHECK-SAME: (%[[T:.*]]: memref<3xi32>)
func.func @slice(%t : tensor<3xi32>) -> tensor<1xi32> {
  // CHECK: memref.subview %[[T]][0] [1] [1] : memref<3xi32> to memref<1xi32>
  %result = tensor.extract_slice %t[0] [1] [1] : tensor<3xi32> to tensor<1xi32>
  func.return %result : tensor<1xi32>
}

func.func @dynamic_broadcast_return(%t : tensor<?x?xf32>, %shape : tensor<2xi32>) -> tensor<?x?xf32> {
  // CHECK: memref.copy
  %bcast = "mhlo.dynamic_broadcast_in_dim"(%t, %shape) {broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>} : (tensor<?x?xf32>, tensor<2xi32>) -> tensor<?x?xf32>
  func.return %bcast : tensor<?x?xf32>
}


// CHECK-LABEL: @arith_select
// CHECK-SAME: %[[C:.*]]: memref<i1>,
// CHECK-SAME: %[[LHS:.*]]: memref<1xf32>,
// CHECK-SAME: %[[RHS:.*]]: memref<1xf32>
func.func @arith_select(%c : tensor<i1>, %lhs: tensor<1xf32>, %rhs: tensor<1xf32>)
                  -> tensor<1xf32> {
  // CHECK: %[[COND:.*]] = memref.load %[[C]][]
  // CHECK: %[[RESULT:.*]] = arith.select %[[COND]], %[[LHS]], %[[RHS]]
  // CHECK-SAME:             : memref<1xf32>
  %cond = tensor.extract %c[] : tensor<i1>
  %result = arith.select %cond, %lhs, %rhs : tensor<1xf32>
  func.return %result : tensor<1xf32>
}


#map = affine_map<(d0) -> (d0)>
func.func @init_tensor_multiple_users(%lhs: tensor<10xf32>,
    %rhs: tensor<10xf32>) -> (tensor<10xf32>, tensor<10xf32>) {
  %init = linalg.init_tensor [10] : tensor<10xf32>
  %add = linalg.generic {
    indexing_maps = [#map, #map, #map],
    iterator_types = ["parallel"]}
    ins(%lhs, %rhs : tensor<10xf32>, tensor<10xf32>)
    outs(%init : tensor<10xf32>) {
  ^bb0(%l: f32, %r: f32, %o: f32):
    %a = arith.addf %l, %r : f32
    linalg.yield %a : f32
  } -> tensor<10xf32>
  %sub = linalg.generic {
    indexing_maps = [#map, #map, #map],
    iterator_types = ["parallel"]}
    ins(%lhs, %rhs : tensor<10xf32>, tensor<10xf32>)
    outs(%init : tensor<10xf32>) {
  ^bb0(%l: f32, %r: f32, %o: f32):
    %s = arith.subf %l, %r : f32
    linalg.yield %s : f32
  } -> tensor<10xf32>
  func.return %add, %sub : tensor<10xf32>, tensor<10xf32>
}
