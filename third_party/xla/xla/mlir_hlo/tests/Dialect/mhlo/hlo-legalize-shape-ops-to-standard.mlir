// RUN: mlir-hlo-opt %s -hlo-legalize-shapeops-to-standard -split-input-file | FileCheck %s

// CHECK-LABEL: compute_reshape_shape
// CHECK-SAME: %[[NUM_ELS:.*]]: index
// CHECK-SAME: %[[TARGET_SHAPE:.*]]: tensor<2xi32>
func.func @compute_reshape_shape(%arg0: index, %arg1: tensor<2xi32>) -> tensor<2xi32> {
  // CHECK: %[[N1:.*]] = arith.constant -1 : index
  // CHECK: %[[IT:.*]] = arith.index_cast %[[TARGET_SHAPE]] : tensor<2xi32> to tensor<2xindex>
  // CHECK: %[[RANK:.*]] = shape.rank %[[IT]] : tensor<2xindex> -> index
  // CHECK: %[[TOTAL:.*]] = shape.reduce(%[[IT]], %[[N1]]) : tensor<2xindex> -> index {
  // CHECK:   ^bb0(%[[IDX:.*]]: index, %[[VAL:.*]]: index, %[[REDUCTION:.*]]: index):
  // CHECK:   %[[NEW_RED:.*]] = arith.muli %[[VAL]], %[[REDUCTION]] : index
  // CHECK:   shape.yield %[[NEW_RED]] : index
  // CHECK: }
  // CHECK: %[[DYNAMIC_EXTENT:.*]] = arith.divui %[[NUM_ELS]], %[[TOTAL]] : index
  // CHECK: %[[COMPUTED_SHAPE:.*]] = tensor.generate   {
  // CHECK:   ^bb0(%[[ARG:.*]]: index):
  // CHECK:   %[[EXT1:.*]] = shape.get_extent %[[IT]], %[[ARG]] : tensor<2xindex>, index -> index
  // CHECK:   %[[IS_DYNAMIC:.*]] = arith.cmpi eq, %[[EXT1]], %[[N1]] : index
  // CHECK:   %[[EXTENT:.*]] = arith.select %[[IS_DYNAMIC]], %[[DYNAMIC_EXTENT]], %[[EXT1]] : index
  // CHECK:   %[[EXTENT_INT:.*]] = arith.index_cast %[[EXTENT]] : index to i32
  // CHECK:   tensor.yield %[[EXTENT_INT]] : i32
  // CHECK: } : tensor<2xi32>
  %0 = "mhlo.compute_reshape_shape"(%arg0, %arg1) : (index, tensor<2xi32>) -> tensor<2xi32>
  func.return %0 : tensor<2xi32>
}

// CHECK-LABEL: cstr_reshapable_op
// CHECK-SAME: %[[NUM_ELS:.*]]: index
// CHECK-SAME: %[[TARGET_SHAPE:.*]]: tensor<2xi32>
func.func @cstr_reshapable_op(%arg0: index, %arg1: tensor<2xi32>) -> !shape.witness {
  // CHECK-DAG: %[[N1:.*]] = arith.constant -1 : index
  // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
  // CHECK: %[[IT0:.*]] = arith.index_cast %[[TARGET_SHAPE]] : tensor<2xi32> to tensor<2xindex>
  // CHECK: %[[VALID:.*]]:3 = shape.reduce(%[[IT0]], %[[C1]], %[[C0]], %[[C0]]) : tensor<2xindex> -> (index, index, index) {
  // CHECK:   ^bb0(%[[IDX:.*]]: index, %[[VAL:.*]]: index, %[[PROD:.*]]: index, %[[DYN_DIMS:.*]]: index, %[[ILLEGAL_DIMS:.*]]: index):
  // CHECK:   %[[V1:.*]] = arith.cmpi eq, %[[N1]], %[[VAL]] : index
  // CHECK:   %[[V2:.*]] = arith.cmpi slt, %[[VAL]], %[[N1]] : index
  // CHECK:   %[[V3:.*]] = arith.select %[[V1]], %[[C1]], %[[C0]] : index
  // CHECK:   %[[V4:.*]] = arith.addi %[[V3]], %[[DYN_DIMS]] : index
  // CHECK:   %[[V5:.*]] = arith.select %[[V2]], %[[C1]], %[[C0]] : index
  // CHECK:   %[[V6:.*]] = arith.addi %[[V5]], %[[ILLEGAL_DIMS]] : index
  // CHECK:   %[[V7:.*]] = arith.select %[[V1]], %[[C1]], %[[VAL]] : index
  // CHECK:   %[[V8:.*]] = arith.muli %[[V7]], %[[PROD]] : index
  // CHECK:   shape.yield %[[V8]], %[[V4]], %[[V6]] : index, index, index
  // CHECK: }
  // CHECK: %[[IS_ZERO_ELS:.*]] = arith.cmpi eq, %[[VALID]]#0, %[[C0]] : index
  // CHECK: %[[DIV:.*]] = arith.select %[[IS_ZERO_ELS]], %[[C1]], %[[VALID]]#0 : index
  // CHECK: %[[REM:.*]] = arith.remsi %[[NUM_ELS]], %[[DIV]] : index
  // CHECK: %[[DIVISIBLE:.*]] = arith.cmpi eq, %[[C0]], %[[REM]] : index
  // CHECK: %[[NOT_TOO_DYNAMIC:.*]] = arith.cmpi ule,  %[[VALID]]#1, %[[C1]] : index
  // CHECK: %[[ALL_VALID_DIMS:.*]] = arith.cmpi eq, %[[VALID]]#2, %[[C0]] : index
  // CHECK: %[[ONE_DYNAMIC:.*]] = arith.cmpi eq, %[[VALID]]#1, %[[C1]] : index
  // CHECK: %[[IS_ALL_EQUAL:.*]] = arith.cmpi eq, %[[NUM_ELS]], %[[VALID]]#0 : index
  // CHECK: %[[EQUAL_IF_NOT_DYNAMIC:.*]] = arith.ori %[[ONE_DYNAMIC]], %[[IS_ALL_EQUAL]] : i1
  // CHECK: %[[PARTIAL_AND2:.*]] = arith.andi %[[ALL_VALID_DIMS]], %[[EQUAL_IF_NOT_DYNAMIC]] : i1
  // CHECK: %[[PARTIAL_AND:.*]] = arith.andi %[[NOT_TOO_DYNAMIC]], %[[PARTIAL_AND2]] : i1
  // CHECK: %[[ALL_CSTRS:.*]] = arith.andi %[[DIVISIBLE]], %[[PARTIAL_AND]] : i1
  // CHECK: %[[W:.*]] = shape.cstr_require %[[ALL_CSTRS]], "Required valid reshape shape input"
  %0 = "mhlo.cstr_reshapable"(%arg0, %arg1) : (index, tensor<2xi32>) -> !shape.witness
  func.return %0 : !shape.witness
}
