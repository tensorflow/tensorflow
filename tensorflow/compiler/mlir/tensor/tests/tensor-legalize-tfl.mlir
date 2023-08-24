// RUN: tf-opt --split-input-file --tensor-legalize-tfl --verify-each %s | FileCheck %s

// -----

// CHECK-LABEL: test_reshape_constant
// CHECK-SAME: %[[INPUT:.*]]: tensor<13x21x3xf32>
// CHECK: %[[SHAPE:.*]] = arith.constant dense<[1, 819]> : tensor<2xi32>
// CHECK: %[[RESULT:.*]] = tensor.reshape %[[INPUT]](%[[SHAPE]]) : (tensor<13x21x3xf32>, tensor<2xi32>) -> tensor<*xf32>
// CHECK: return %[[RESULT]] : tensor<*xf32>
func.func @test_reshape_constant(%arg0: tensor<13x21x3xf32>) -> tensor<*xf32> {
  %cst = arith.constant dense<[1, 819]> : tensor<2xi32>
  %0 = "tfl.reshape"(%arg0, %cst) : (tensor<13x21x3xf32>, tensor<2xi32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// -----

// CHECK-LABEL: test_reshape_constant_with_placeholder
// CHECK-SAME: %[[INPUT:.*]]: tensor<13x21x3xf32>
// CHECK: %[[SHAPE:.*]] = arith.constant dense<[1, 819]> : tensor<2xi32>
// CHECK: %[[RESULT:.*]] = tensor.reshape %[[INPUT]](%[[SHAPE]]) : (tensor<13x21x3xf32>, tensor<2xi32>) -> tensor<*xf32>
// CHECK: return %[[RESULT]] : tensor<*xf32>
func.func @test_reshape_constant_with_placeholder(%arg0: tensor<13x21x3xf32>) -> tensor<*xf32> {
  %cst = arith.constant dense<[1, -1]> : tensor<2xi32>
  %0 = "tfl.reshape"(%arg0, %cst) : (tensor<13x21x3xf32>, tensor<2xi32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// -----

// CHECK-LABEL: test_reshape_variable
// CHECK-SAME: %[[INPUT:.*]]: tensor<?xf32>
// CHECK-SAME: %[[SHAPE:.*]]: tensor<2xi32>
// CHECK: %[[CONST_MINUS_1_SPLAT:.*]] = arith.constant dense<-1> : tensor<2xi32>
// CHECK: %[[CONST_1:.*]] = arith.constant 1 : index
// CHECK: %[[CONST_0:.*]] = arith.constant 0 : index
// CHECK: %[[CONST_1_TENSOR:.*]] = arith.constant dense<1> : tensor<i32>
// CHECK: %[[CONST_0_I32:.*]] = arith.constant 0 : i32
// CHECK: %[[SHAPE_PRODUCT_TENSOR:.*]] = linalg.reduce ins(%[[SHAPE]] : tensor<2xi32>) outs(%[[CONST_1_TENSOR]] : tensor<i32>) dimensions = [0]
// CHECK:   (%[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: i32) {
// CHECK:     %[[VAL_2:.*]] = arith.muli %[[VAL_0]], %[[VAL_1]] : i32
// CHECK:     linalg.yield %[[VAL_2]] : i32
// CHECK:   }
// CHECK: %[[SHAPE_PRODUCT:.*]] = tensor.extract %[[SHAPE_PRODUCT_TENSOR]][] : tensor<i32>
// CHECK: %[[SHAPE_IS_NEGATIVE:.*]] = arith.cmpi slt, %[[SHAPE_PRODUCT]], %[[CONST_0_I32]] : i32
// CHECK: %[[SHAPE_RESOLVED:.*]] = scf.if %[[SHAPE_IS_NEGATIVE]] -> (tensor<2xi32>) {
// CHECK:   %[[INPUT_SIZE:.*]] = scf.for %[[I:.*]] = %[[CONST_0]] to %[[CONST_1]] step %[[CONST_1]] iter_args(%[[ACC_SIZE:.*]] = %[[CONST_1]]) -> (index) {
// CHECK:     %[[VAL_3:.*]] = tensor.dim %[[INPUT]], %[[I]] : tensor<?xf32>
// CHECK:     %[[TEMP_SIZE:.*]] = arith.muli %[[ACC_SIZE]], %[[VAL_3]] : index
// CHECK:     scf.yield %[[TEMP_SIZE]] : index
// CHECK:   }
// CHECK:   %[[INPUT_SIZE_I32:.*]] = arith.index_cast %[[INPUT_SIZE]] : index to i32
// CHECK:   %[[SHAPE_PRODUCT_ABS:.*]] = math.absi %[[SHAPE_PRODUCT]] : i32
// CHECK:   %[[WILCARD_SUBSTITUTION:.*]] = arith.divsi %[[INPUT_SIZE_I32]], %[[SHAPE_PRODUCT_ABS]] : i32
// CHECK:   %[[PLACEHOLDER_SUBSTITUTION_SPLAT:.*]] = tensor.splat %[[WILCARD_SUBSTITUTION]] : tensor<2xi32>
// CHECK:   %[[PLACEHOLDER_MASK:.*]] = arith.cmpi eq, %[[SHAPE]], %[[CONST_MINUS_1_SPLAT]] : tensor<2xi32>
// CHECK:   %[[SHAPE_RESOLVED_TEMP:.*]] = arith.select %[[PLACEHOLDER_MASK]], %[[PLACEHOLDER_SUBSTITUTION_SPLAT]], %[[SHAPE]] : tensor<2xi1>, tensor<2xi32>
// CHECK:   scf.yield %[[SHAPE_RESOLVED_TEMP]] : tensor<2xi32>
// CHECK: } else {
// CHECK:   scf.yield %[[SHAPE]] : tensor<2xi32>
// CHECK: }
// CHECK: %[[RESULT:.*]] = tensor.reshape %[[INPUT]](%[[SHAPE_RESOLVED]]) : (tensor<?xf32>, tensor<2xi32>) -> tensor<?x?xf32>
// CHECK: return %[[RESULT]] : tensor<?x?xf32>
func.func @test_reshape_variable(%arg0: tensor<?xf32>, %arg1: tensor<2xi32>) -> tensor<?x?xf32> {
  %0 = "tfl.reshape"(%arg0, %arg1) : (tensor<?xf32>, tensor<2xi32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// -----

// CHECK-LABEL: test_reshape_unranked
// CHECK-SAME: %[[INPUT:.*]]: tensor<*xf32>
// CHECK-SAME: %[[SHAPE:.*]]: tensor<*xi32>
// CHECK: %[[CONST_MINUS_1:.*]] = arith.constant dense<-1> : tensor<i32>
// CHECK: %[[CONST_1:.*]] = arith.constant 1 : index
// CHECK: %[[CONST_0:.*]] = arith.constant 0 : index
// CHECK: %[[CONST_1_TENSOR:.*]] = arith.constant dense<1> : tensor<i32>
// CHECK: %[[CONST_0_I32:.*]] = arith.constant 0 : i32
// CHECK: %[[SHAPE_CAST:.*]] = tensor.cast %[[SHAPE]] : tensor<*xi32> to tensor<?xi32>
// CHECK: %[[SHAPE_PRODUCT_TENSOR:.*]] = linalg.reduce ins(%[[SHAPE_CAST]] : tensor<?xi32>) outs(%[[CONST_1_TENSOR]] : tensor<i32>) dimensions = [0]
// CHECK:   (%[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: i32) {
// CHECK:     %[[VAL_2:.*]] = arith.muli %[[VAL_0]], %[[VAL_1]] : i32
// CHECK:     linalg.yield %[[VAL_2]] : i32
// CHECK:   }
// CHECK: %[[SHAPE_PRODUCT:.*]] = tensor.extract %[[SHAPE_PRODUCT_TENSOR]][] : tensor<i32>
// CHECK: %[[SHAPE_PRODUCT_IS_NEGATIVE:.*]] = arith.cmpi slt, %[[SHAPE_PRODUCT]], %[[CONST_0_I32]] : i32
// CHECK: %[[SHAPE_RESOLVED:.*]] = scf.if %[[SHAPE_PRODUCT_IS_NEGATIVE]] -> (tensor<?xi32>) {
// CHECK:   %[[INPUT_RANK:.*]] = tensor.rank %[[INPUT]] : tensor<*xf32>
// CHECK:   %[[INPUT_SIZE:.*]] = scf.for %[[I:.*]] = %[[CONST_0]] to %[[INPUT_RANK]] step %[[CONST_1]] iter_args(%[[ACC_SIZE:.*]] = %[[CONST_1]]) -> (index) {
// CHECK:     %[[VAL_3:.*]] = tensor.dim %[[INPUT]], %[[I]] : tensor<*xf32>
// CHECK:     %[[TEMP_SIZE:.*]] = arith.muli %[[ACC_SIZE]], %[[VAL_3]] : index
// CHECK:     scf.yield %[[TEMP_SIZE]] : index
// CHECK:   }
// CHECK:   %[[INPUT_SIZE_I32:.*]] = arith.index_cast %[[INPUT_SIZE]] : index to i32
// CHECK:   %[[SHAPE_PRODUCT_ABS:.*]] = math.absi %[[SHAPE_PRODUCT]] : i32
// CHECK:   %[[SHAPE_SIZE:.*]] = tensor.dim %[[SHAPE_CAST]], %[[CONST_0]] : tensor<?xi32>
// CHECK:   %[[PLACEHOLDER_SUBSTITUTION:.*]] = arith.divsi %[[INPUT_SIZE_I32]], %[[SHAPE_PRODUCT_ABS]] : i32
// CHECK:   %[[PLACEHOLDER_SUBSTITUTION_TENSOR:.*]] = tensor.from_elements %[[PLACEHOLDER_SUBSTITUTION]] : tensor<i32>
// CHECK:   %[[PLACEHOLDER_SUBSTITUTION_EMPTY:.*]] = tensor.empty(%[[SHAPE_SIZE]]) : tensor<?xi32>
// CHECK:   %[[PLACEHOLDER_SUBSTITUTION_SPLAT:.*]] = linalg.broadcast ins(%[[PLACEHOLDER_SUBSTITUTION_TENSOR]] : tensor<i32>) outs(%[[PLACEHOLDER_SUBSTITUTION_EMPTY]] : tensor<?xi32>) dimensions = [0]
// CHECK:   %[[MINUS_1_EMPTY:.*]] = tensor.empty(%[[SHAPE_SIZE]]) : tensor<?xi32>
// CHECK:   %[[MINUS_1_SPLAT:.*]] = linalg.broadcast ins(%[[CONST_MINUS_1]] : tensor<i32>) outs(%[[MINUS_1_EMPTY]] : tensor<?xi32>) dimensions = [0]
// CHECK:   %[[SHAPE_MASK:.*]] = arith.cmpi eq, %[[SHAPE_CAST]], %[[MINUS_1_SPLAT]] : tensor<?xi32>
// CHECK:   %[[RESOLVED_SHAPE_TEMP:.*]] = arith.select %[[SHAPE_MASK]], %[[PLACEHOLDER_SUBSTITUTION_SPLAT]], %[[SHAPE_CAST]] : tensor<?xi1>, tensor<?xi32>
// CHECK:   scf.yield %[[RESOLVED_SHAPE_TEMP]] : tensor<?xi32>
// CHECK: } else {
// CHECK:   scf.yield %[[SHAPE_CAST]] : tensor<?xi32>
// CHECK: }
// CHECK: %[[RESULT:.*]] = tensor.reshape %[[INPUT]](%[[SHAPE_RESOLVED]]) : (tensor<*xf32>, tensor<?xi32>) -> tensor<*xf32>
// CHECK: return %[[RESULT]] : tensor<*xf32>
func.func @test_reshape_unranked(%arg0: tensor<*xf32>, %arg1: tensor<*xi32>) -> tensor<*xf32> {
  %0 = "tfl.reshape"(%arg0, %arg1) : (tensor<*xf32>, tensor<*xi32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}

