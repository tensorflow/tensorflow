// RUN: tf-opt -tfl-lower-static-tensor-list %s | FileCheck %s --dump-input-on-failure

func @tensorlistGetItem(%arg0: tensor<3x10xf32>, %arg1: tensor<1xi32>, %arg2: tensor<i32>) -> (tensor<10xf32>, tensor<3x10xf32>) {
  %0 = "tf.TensorListFromTensor"(%arg0, %arg1) : (tensor<3x10xf32>, tensor<1xi32>) -> tensor<!tf.variant<tensor<10xf32>>>
  %1 = "tf.TensorListGetItem"(%0, %arg2, %arg1) : (tensor<!tf.variant<tensor<10xf32>>>, tensor<i32>, tensor<1xi32>) -> tensor<10xf32>
  %2 = "tf.TensorListStack"(%0, %arg1) : (tensor<!tf.variant<tensor<10xf32>>>, tensor<1xi32>) -> tensor<3x10xf32>
  return %1, %2 : tensor<10xf32>, tensor<3x10xf32>

// CHECK-LABEL: tensorlistGetItem
// CHECK:  %0 = "tf.Gather"(%arg0, %arg2) {validate_indices = true} : (tensor<3x10xf32>, tensor<i32>) -> tensor<10xf32>
// CHECK: return %0, %arg0 : tensor<10xf32>, tensor<3x10xf32>
}

func @tensorlistGetItemWithUnknownRank(%arg0: tensor<*xf32>, %arg1: tensor<1xi32>, %arg2: tensor<i32>) -> (tensor<*xf32>, tensor<*xf32>) {
  %0 = "tf.TensorListFromTensor"(%arg0, %arg1) : (tensor<*xf32>, tensor<1xi32>) -> tensor<!tf.variant<tensor<*xf32>>>
  %1 = "tf.TensorListGetItem"(%0, %arg2, %arg1) : (tensor<!tf.variant<tensor<*xf32>>>, tensor<i32>, tensor<1xi32>) -> tensor<*xf32>
  %2 = "tf.TensorListStack"(%0, %arg1) : (tensor<!tf.variant<tensor<*xf32>>>, tensor<1xi32>) -> tensor<*xf32>
  return %1, %2 : tensor<*xf32>, tensor<*xf32>

// CHECK-LABEL: tensorlistGetItemWithUnknownRank
// CHECK:  %0 = "tf.Gather"(%arg0, %arg2) {validate_indices = true} : (tensor<*xf32>, tensor<i32>) -> tensor<*xf32>
// CHECK: return %0, %arg0 : tensor<*xf32>, tensor<*xf32>
}

func @tensorlistStackWithConstantElementShape(%arg0: tensor<?x3xf32>) -> (tensor<2x3xf32>) {
  %cst = constant dense<3> : tensor<1xi32>
  %0 = "tf.TensorListFromTensor"(%arg0, %cst) : (tensor<?x3xf32>, tensor<1xi32>) -> tensor<!tf.variant<tensor<3xf32>>>
  %1 = "tf.TensorListStack"(%0, %cst) {num_elements = 2 : i64} : (tensor<!tf.variant<tensor<3xf32>>>, tensor<1xi32>) -> tensor<2x3xf32>
  return %1 : tensor<2x3xf32>

// CHECK-LABEL: tensorlistStackWithConstantElementShape
// CHECK:  [[ELEM_SHAPE:%cst.*]] = constant dense<3> : tensor<1xi32>
// CHECK-NEXT:  [[SHAPE:%.*]] = "tf.Shape"(%arg0) : (tensor<?x3xf32>) -> tensor<?xi32>
// CHECK-NEXT:  [[RESHAPE:%.*]] = "tf.Reshape"(%arg0, [[SHAPE]]) : (tensor<?x3xf32>, tensor<?xi32>) -> tensor<2x3xf32>
// CHECK-NEXT:  return [[RESHAPE]] : tensor<2x3xf32>
}

func @tensorlistSetItem(%arg0: tensor<3x10xf32>, %arg1: tensor<1xi32>, %arg2: tensor<i32>, %arg3: tensor<10xf32>) -> tensor<3x10xf32> {
  %0 = "tf.TensorListFromTensor"(%arg0, %arg1) : (tensor<3x10xf32>, tensor<1xi32>) -> tensor<!tf.variant<tensor<10xf32>>>
  %1 = "tf.TensorListSetItem"(%0, %arg2, %arg3) : (tensor<!tf.variant<tensor<10xf32>>>, tensor<i32>, tensor<10xf32>) -> tensor<!tf.variant<tensor<10xf32>>>
  %2 = "tf.TensorListStack"(%1, %arg1) : (tensor<!tf.variant<tensor<10xf32>>>, tensor<1xi32>) -> tensor<3x10xf32>
  return %2 : tensor<3x10xf32>

// CHECK-LABEL: tensorlistSetItem
// CHECK-SAME:  ([[INPUT:%.*]]: tensor<3x10xf32>, [[ELEM_SHAPE:%.*]]: tensor<1xi32>, [[INDEX:%.*]]: tensor<i32>, [[ITEM:%.*]]: tensor<10xf32>)
// CHECK-DAG:  [[RANK:%.*]] = "tf.Rank"([[ITEM]]) : (tensor<10xf32>) -> tensor<i32>
// CHECK-DAG:  [[ZERO:%cst.*]] = constant dense<0> : tensor<i32>
// CHECK-DAG:  [[ONE:%cst.*]] = constant dense<1> : tensor<i32>
// CHECK:  [[SUFFIX_START:%.*]] = "tf.Add"([[INDEX]], [[ONE]]) : (tensor<i32>, tensor<i32>) -> tensor<i32>
// CHECK:  [[VECTOR_RANK:%.*]] = "tf.ExpandDims"([[RANK]], [[ZERO]]) : (tensor<i32>, tensor<i32>) -> tensor<1xi32>
// CHECK-DAG:  [[NEG_ONE:%cst.*]] = constant dense<-1> : tensor<i32>
// CHECK-DAG:  [[ZERO_1:%cst.*]] = constant dense<0> : tensor<i32>


// CHECK:  [[PARTIAL_START_POS:%.*]] = "tf.Fill"([[VECTOR_RANK]], [[ZERO_1]]) : (tensor<1xi32>, tensor<i32>) -> tensor<?xi32>
// CHECK:  [[ZERO_2:%cst.*]] = constant dense<0> : tensor<i32>
// CHECK:  [[EXPANDED_START_INDEX:%.*]] = "tf.ExpandDims"([[ZERO]], [[ZERO_2]]) : (tensor<i32>, tensor<i32>) -> tensor<1xi32>
// CHECK:  [[START_POS:%.*]] = "tf.Concat"([[ZERO_2]], [[EXPANDED_START_INDEX]], [[PARTIAL_START_POS]]) {N = 2 : i64} : (tensor<i32>, tensor<1xi32>, tensor<?xi32>) -> tensor<?xi32>
// CHECK:  [[LEADING_DIM:%.*]] = "tf.ExpandDims"([[INDEX]], [[ZERO_2]]) : (tensor<i32>, tensor<i32>) -> tensor<1xi32>
// CHECK:  [[NEG_ONE_1:%cst.*]] = constant dense<-1> : tensor<i32>
// CHECK:  [[PARTIAL_SIZE:%.*]] = "tf.Fill"([[VECTOR_RANK]], [[NEG_ONE_1]]) : (tensor<1xi32>, tensor<i32>) -> tensor<?xi32>
// CHECK:  [[SLICE_SIZE:%.*]] = "tf.Concat"([[ZERO_2]], [[LEADING_DIM]], [[PARTIAL_SIZE]]) {N = 2 : i64} : (tensor<i32>, tensor<1xi32>, tensor<?xi32>) -> tensor<?xi32>
// CHECK:  [[SLICE:%.*]] = "tf.Slice"([[INPUT]], [[START_POS]], [[SLICE_SIZE]]) : (tensor<3x10xf32>, tensor<?xi32>, tensor<?xi32>) -> tensor<*xf32>


// CHECK:  [[ZERO_3:%cst.*]] = constant dense<0> : tensor<i32>
// CHECK:  [[PARTIAL_START_POS_1:%.*]] = "tf.Fill"([[VECTOR_RANK]], [[ZERO_3]]) : (tensor<1xi32>, tensor<i32>) -> tensor<?xi32>
// CHECK:  [[ZERO_4:%cst.*]] = constant dense<0> : tensor<i32>
// CHECK:  [[EXPANDED_START_INDEX_1:%.*]] = "tf.ExpandDims"([[SUFFIX_START]], [[ZERO_4]]) : (tensor<i32>, tensor<i32>) -> tensor<1xi32>
// CHECK:  [[START_POS_1:%.*]] = "tf.Concat"([[ZERO_4]], [[EXPANDED_START_INDEX_1]], [[PARTIAL_START_POS_1]]) {N = 2 : i64} : (tensor<i32>, tensor<1xi32>, tensor<?xi32>) -> tensor<?xi32>
// CHECK:  [[LEADING_DIM_1:%.*]] = "tf.ExpandDims"([[NEG_ONE]], [[ZERO_4]]) : (tensor<i32>, tensor<i32>) -> tensor<1xi32>
// CHECK:  [[NEG_ONE_2:%cst.*]] = constant dense<-1> : tensor<i32>
// CHECK:  [[PARTIAL_SIZE_1:%.*]] = "tf.Fill"([[VECTOR_RANK]], [[NEG_ONE_2]]) : (tensor<1xi32>, tensor<i32>) -> tensor<?xi32>
// CHECK:  [[SLICE_SIZE_1:%.*]] = "tf.Concat"([[ZERO_4]], [[LEADING_DIM_1]], [[PARTIAL_SIZE_1]]) {N = 2 : i64} : (tensor<i32>, tensor<1xi32>, tensor<?xi32>) -> tensor<?xi32>
// CHECK:  [[SLICE_1:%.*]] = "tf.Slice"([[INPUT]], [[START_POS_1]], [[SLICE_SIZE_1]]) : (tensor<3x10xf32>, tensor<?xi32>, tensor<?xi32>) -> tensor<*xf32>


// CHECK:  [[EXPANDED_ITEM:%.*]] = "tf.ExpandDims"([[ITEM]], [[ZERO]]) : (tensor<10xf32>, tensor<i32>) -> tensor<*xf32>
// CHECK:  [[RESULT:%.*]] = "tf.Concat"([[ZERO]], [[SLICE]], [[EXPANDED_ITEM]], [[SLICE_1]]) {N = 3 : i64} : (tensor<i32>, tensor<*xf32>, tensor<*xf32>, tensor<*xf32>) -> tensor<3x10xf32>
// CHECK:  return [[RESULT]] : tensor<3x10xf32>
}

func @tensorlistSetItemWithScalarElements(%arg0: tensor<5xf32>, %arg1: tensor<0xi32>, %arg2: tensor<i32>, %arg3: tensor<f32>) -> tensor<5xf32> {
  %0 = "tf.TensorListFromTensor"(%arg0, %arg1) : (tensor<5xf32>, tensor<0xi32>) -> tensor<!tf.variant<tensor<f32>>>
  %1 = "tf.TensorListSetItem"(%0, %arg2, %arg3) : (tensor<!tf.variant<tensor<f32>>>, tensor<i32>, tensor<f32>) -> tensor<!tf.variant<tensor<f32>>>
  %2 = "tf.TensorListStack"(%1, %arg1) : (tensor<!tf.variant<tensor<f32>>>, tensor<0xi32>) -> tensor<5xf32>
  return %2 : tensor<5xf32>

// CHECK-LABEL: tensorlistSetItemWithScalarElements
// CHECK-SAME:  ([[INPUT:%.*]]: tensor<5xf32>, [[ELEM_SHAPE:%.*]]: tensor<0xi32>, [[INDEX:%.*]]: tensor<i32>, [[ITEM:%.*]]: tensor<f32>)
// CHECK-DAG:  [[RANK:%.*]] = "tf.Rank"([[ITEM]]) : (tensor<f32>) -> tensor<i32>
// CHECK-DAG:  [[ZERO:%cst.*]] = constant dense<0> : tensor<i32>
// CHECK-DAG:  [[ONE:%cst.*]] = constant dense<1> : tensor<i32>
// CHECK:  [[SUFFIX_START:%.*]] = "tf.Add"([[INDEX]], [[ONE]]) : (tensor<i32>, tensor<i32>) -> tensor<i32>
// CHECK:  [[VECTOR_RANK:%.*]] = "tf.ExpandDims"([[RANK]], [[ZERO]]) : (tensor<i32>, tensor<i32>) -> tensor<1xi32>
// CHECK-DAG:  [[NEG_ONE:%cst.*]] = constant dense<-1> : tensor<i32>
// CHECK-DAG:  [[ZERO_1:%cst.*]] = constant dense<0> : tensor<i32>


// CHECK:  [[PARTIAL_START_POS:%.*]] = "tf.Fill"([[VECTOR_RANK]], [[ZERO_1]]) : (tensor<1xi32>, tensor<i32>) -> tensor<?xi32>
// CHECK:  [[ZERO_2:%cst.*]] = constant dense<0> : tensor<i32>
// CHECK:  [[EXPANDED_START_INDEX:%.*]] = "tf.ExpandDims"([[ZERO]], [[ZERO_2]]) : (tensor<i32>, tensor<i32>) -> tensor<1xi32>
// CHECK:  [[START_POS:%.*]] = "tf.Concat"([[ZERO_2]], [[EXPANDED_START_INDEX]], [[PARTIAL_START_POS]]) {N = 2 : i64} : (tensor<i32>, tensor<1xi32>, tensor<?xi32>) -> tensor<?xi32>
// CHECK:  [[LEADING_DIM:%.*]] = "tf.ExpandDims"([[INDEX]], [[ZERO_2]]) : (tensor<i32>, tensor<i32>) -> tensor<1xi32>
// CHECK:  [[NEG_ONE_1:%cst.*]] = constant dense<-1> : tensor<i32>
// CHECK:  [[PARTIAL_SIZE:%.*]] = "tf.Fill"([[VECTOR_RANK]], [[NEG_ONE_1]]) : (tensor<1xi32>, tensor<i32>) -> tensor<?xi32>
// CHECK:  [[SLICE_SIZE:%.*]] = "tf.Concat"([[ZERO_2]], [[LEADING_DIM]], [[PARTIAL_SIZE]]) {N = 2 : i64} : (tensor<i32>, tensor<1xi32>, tensor<?xi32>) -> tensor<?xi32>
// CHECK:  [[SLICE:%.*]] = "tf.Slice"([[INPUT]], [[START_POS]], [[SLICE_SIZE]]) : (tensor<5xf32>, tensor<?xi32>, tensor<?xi32>) -> tensor<*xf32>


// CHECK:  [[ZERO_3:%cst.*]] = constant dense<0> : tensor<i32>
// CHECK:  [[PARTIAL_START_POS_1:%.*]] = "tf.Fill"([[VECTOR_RANK]], [[ZERO_3]]) : (tensor<1xi32>, tensor<i32>) -> tensor<?xi32>
// CHECK:  [[ZERO_4:%cst.*]] = constant dense<0> : tensor<i32>
// CHECK:  [[EXPANDED_START_INDEX_1:%.*]] = "tf.ExpandDims"([[SUFFIX_START]], [[ZERO_4]]) : (tensor<i32>, tensor<i32>) -> tensor<1xi32>
// CHECK:  [[START_POS_1:%.*]] = "tf.Concat"([[ZERO_4]], [[EXPANDED_START_INDEX_1]], [[PARTIAL_START_POS_1]]) {N = 2 : i64} : (tensor<i32>, tensor<1xi32>, tensor<?xi32>) -> tensor<?xi32>
// CHECK:  [[LEADING_DIM_1:%.*]] = "tf.ExpandDims"([[NEG_ONE]], [[ZERO_4]]) : (tensor<i32>, tensor<i32>) -> tensor<1xi32>
// CHECK:  [[NEG_ONE_2:%cst.*]] = constant dense<-1> : tensor<i32>
// CHECK:  [[PARTIAL_SIZE_1:%.*]] = "tf.Fill"([[VECTOR_RANK]], [[NEG_ONE_2]]) : (tensor<1xi32>, tensor<i32>) -> tensor<?xi32>
// CHECK:  [[SLICE_SIZE_1:%.*]] = "tf.Concat"([[ZERO_4]], [[LEADING_DIM_1]], [[PARTIAL_SIZE_1]]) {N = 2 : i64} : (tensor<i32>, tensor<1xi32>, tensor<?xi32>) -> tensor<?xi32>
// CHECK:  [[SLICE_1:%.*]] = "tf.Slice"([[INPUT]], [[START_POS_1]], [[SLICE_SIZE_1]]) : (tensor<5xf32>, tensor<?xi32>, tensor<?xi32>) -> tensor<*xf32>


// CHECK:  [[EXPANDED_ITEM:%.*]] = "tf.ExpandDims"([[ITEM]], [[ZERO]]) : (tensor<f32>, tensor<i32>) -> tensor<*xf32>
// CHECK:  [[RESULT:%.*]] = "tf.Concat"([[ZERO]], [[SLICE]], [[EXPANDED_ITEM]], [[SLICE_1]]) {N = 3 : i64} : (tensor<i32>, tensor<*xf32>, tensor<*xf32>, tensor<*xf32>) -> tensor<5xf32>
// CHECK:  return [[RESULT]] : tensor<5xf32>
}

func @tensorlistReserve(%arg0: tensor<3xi32>, %arg1: tensor<i32>, %arg2: tensor<i32>) -> tensor<?x?x?xf32> {
  %0 = "tf.TensorListReserve"(%arg0, %arg1) : (tensor<3xi32>, tensor<i32>) -> tensor<!tf.variant<tensor<?x?x?xf32>>>
  %1 = "tf.TensorListGetItem"(%0, %arg2, %arg0) : (tensor<!tf.variant<tensor<?x?x?xf32>>>, tensor<i32>, tensor<3xi32>) -> tensor<?x?x?xf32>
  return %1 : tensor<?x?x?xf32>

// CHECK-LABEL: tensorlistReserve
// CHECK-DAG:  [[ZERO1:%cst.*]] = constant dense<0> : tensor<i32>
// CHECK-DAG:  [[ZERO2:%cst.*]] = constant dense<0> : tensor<i32>
// CHECK-DAG:  [[DIM0:%.*]] = "tf.ExpandDims"(%arg1, [[ZERO1]]) : (tensor<i32>, tensor<i32>) -> tensor<1xi32>
// CHECK-DAG:  [[SHAPE:%.*]] = "tf.Concat"([[ZERO2]], [[DIM0]], %arg0) {N = 2 : i64} : (tensor<i32>, tensor<1xi32>, tensor<3xi32>) -> tensor<4xi32>
// CHECK-DAG:  [[VALUES:%.*]] = constant dense<0.000000e+00> : tensor<f32>
// CHECK:      [[LIST:%.*]] = "tf.Fill"([[SHAPE]], [[VALUES]]) : (tensor<4xi32>, tensor<f32>) -> tensor<?x?x?x?xf32>
// CHECK:      [[RESULT:%.*]] = "tf.Gather"([[LIST]], %arg2) {validate_indices = true} : (tensor<?x?x?x?xf32>, tensor<i32>) -> tensor<?x?x?xf32>
// CHECK:      return [[RESULT]] : tensor<?x?x?xf32>
}

func @tensorlistReserveUnrankedElements(%arg0: tensor<?xi32>, %arg1: tensor<i32>, %arg2: tensor<i32>) -> tensor<*xf32> {
  %0 = "tf.TensorListReserve"(%arg0, %arg1) : (tensor<?xi32>, tensor<i32>) -> tensor<!tf.variant<tensor<*xf32>>>
  %1 = "tf.TensorListGetItem"(%0, %arg2, %arg0) : (tensor<!tf.variant<tensor<*xf32>>>, tensor<i32>, tensor<?xi32>) -> tensor<*xf32>
  return %1 : tensor<*xf32>

// CHECK-LABEL: tensorlistReserveUnrankedElements
// CHECK:  [[RESULT:%[0-9]+]] = "tf.Fill"{{.*}}(tensor<?xi32>, tensor<f32>) -> tensor<*xf32>
// CHECK:  [[RESULT2:%[0-9]+]] = "tf.Gather"{{.*}}{validate_indices = true} : (tensor<*xf32>, tensor<i32>) -> tensor<*xf32>
// CHECK:  return [[RESULT2]] : tensor<*xf32>
}

func @tensorlistReserveConstantUnknownElementShapeDim(%arg0: tensor<i32>, %arg1: tensor<i32>) -> tensor<?x7xf32> {
  %cst = constant dense<[-1, 7]> : tensor<2xi32>
  %0 = "tf.TensorListReserve"(%cst, %arg0) : (tensor<2xi32>, tensor<i32>) -> tensor<!tf.variant<tensor<?x7xf32>>>
  %1 = "tf.TensorListGetItem"(%0, %arg1, %cst) : (tensor<!tf.variant<tensor<?x7xf32>>>, tensor<i32>, tensor<2xi32>) -> tensor<?x7xf32>
  return %1 : tensor<?x7xf32>

// CHECK-LABEL: tensorlistReserveConstantUnknownElementShapeDim
// CHECK-DAG:  [[ZERO1:%cst.*]] = constant dense<0> : tensor<i32>
// CHECK-DAG:  [[ZERO2:%cst.*]] = constant dense<0> : tensor<i32>
// CHECK-DAG:  [[ELEMENT_SHAPE:%cst.*]] = constant dense<[1, 7]> : tensor<2xi32>
// CHECK-DAG:  [[DIM0:%.*]] = "tf.ExpandDims"(%arg0, [[ZERO1]]) : (tensor<i32>, tensor<i32>) -> tensor<1xi32>
// CHECK-DAG:  [[SHAPE:%.*]] = "tf.Concat"([[ZERO2]], [[DIM0]], [[ELEMENT_SHAPE]]) {N = 2 : i64} : (tensor<i32>, tensor<1xi32>, tensor<2xi32>) -> tensor<3xi32>
// CHECK-DAG:  [[VALUES:%.*]] = constant dense<0.000000e+00> : tensor<f32>
// CHECK:      [[LIST:%.*]] = "tf.Fill"([[SHAPE]], [[VALUES]]) : (tensor<3xi32>, tensor<f32>) -> tensor<?x?x7xf32>
// CHECK:      [[RESULT:%.*]] = "tf.Gather"([[LIST]], %arg1) {validate_indices = true} : (tensor<?x?x7xf32>, tensor<i32>) -> tensor<?x7xf32>
// CHECK:      return [[RESULT]] : tensor<?x7xf32>
}

func @EmptyTensorList(%arg0: tensor<3xi32>, %arg1: tensor<i32>, %arg2: tensor<i32>) -> tensor<?x?x?xf32> {
  %0 = "tf.EmptyTensorList"(%arg0, %arg1) : (tensor<3xi32>, tensor<i32>) -> tensor<!tf.variant<tensor<?x?x?xf32>>>
  %1 = "tf.TensorListGetItem"(%0, %arg2, %arg0) : (tensor<!tf.variant<tensor<?x?x?xf32>>>, tensor<i32>, tensor<3xi32>) -> tensor<?x?x?xf32>
  return %1 : tensor<?x?x?xf32>

// CHECK-LABEL: EmptyTensorList
// CHECK-SAME:  ([[ELEM_SHAPE:%.*]]: tensor<3xi32>, [[MAX_ELEMS:%.*]]: tensor<i32>, [[IDX:%.*]]: tensor<i32>)
// CHECK-DAG:  [[DIM0:%cst.*]] = constant dense<0> : tensor<1xi32>
// CHECK-DAG:  [[ZERO:%cst.*]] = constant dense<0> : tensor<i32>
// CHECK-DAG:  [[SHAPE:%.*]] = "tf.Concat"([[ZERO]], [[DIM0]], [[ELEM_SHAPE]]) {N = 2 : i64} : (tensor<i32>, tensor<1xi32>, tensor<3xi32>) -> tensor<4xi32>
// CHECK-DAG:  [[VALUES:%.*]] = constant dense<0.000000e+00> : tensor<f32>
// CHECK:      [[LIST:%.*]] = "tf.Fill"([[SHAPE]], [[VALUES]]) : (tensor<4xi32>, tensor<f32>) -> tensor<?x?x?x?xf32>
// CHECK:      [[RESULT:%.*]] = "tf.Gather"([[LIST]], [[IDX]]) {validate_indices = true} : (tensor<?x?x?x?xf32>, tensor<i32>) -> tensor<?x?x?xf32>
// CHECK:      return [[RESULT]] : tensor<?x?x?xf32>
}

func @tensorlistPushBack(%arg0: tensor<3x10xf32>, %arg1: tensor<1xi32>, %arg2: tensor<10xf32>) -> tensor<?x10xf32> {
  %0 = "tf.TensorListFromTensor"(%arg0, %arg1) : (tensor<3x10xf32>, tensor<1xi32>) -> tensor<!tf.variant<tensor<10xf32>>>
  %1 = "tf.TensorListPushBack"(%0, %arg2) : (tensor<!tf.variant<tensor<10xf32>>>, tensor<10xf32>) -> tensor<!tf.variant<tensor<10xf32>>>
  %2 = "tf.TensorListStack"(%1, %arg1) : (tensor<!tf.variant<tensor<10xf32>>>, tensor<1xi32>) -> tensor<?x10xf32>
  return %2 : tensor<?x10xf32>

// CHECK-LABEL: tensorlistPushBack
// CHECK-SAME:  ([[INPUT:%.*]]: tensor<3x10xf32>, [[ELEM_SHAPE:%.*]]: tensor<1xi32>, [[ITEM:%.*]]: tensor<10xf32>)
// CHECK:   [[ZERO:%.*]] = constant dense<0> : tensor<i32>
// CHECK:   [[EXP_ITEM:%.*]] = "tf.ExpandDims"([[ITEM]], [[ZERO]]) {{.*}} -> tensor<1x10xf32>
// CHECK:   [[RESULT:%.*]] = "tf.Concat"(%cst, [[INPUT]], [[EXP_ITEM]]) {N = 2 : i64} : {{.*}} -> tensor<?x10xf32>
// CHECK:   return [[RESULT]] : tensor<?x10xf32>
}

func @tensorlistLength(%arg0: tensor<3x10xf32>, %arg1: tensor<1xi32>) -> (tensor<i32>) {
  %0 = "tf.TensorListFromTensor"(%arg0, %arg1) : (tensor<3x10xf32>, tensor<1xi32>) -> tensor<!tf.variant<tensor<10xf32>>>
  %1 = "tf.TensorListLength"(%0) : (tensor<!tf.variant<tensor<10xf32>>>) -> tensor<i32>
  return %1: tensor<i32>

// CHECK-LABEL: tensorlistLength
// CHECK-SAME: ([[INPUT:%.*]]: tensor<3x10xf32>, [[ELEM_SHAPE:%.*]]: tensor<1xi32>)
// CHECK-DAG: [[SHAPE:%.*]] = "tf.Shape"([[INPUT]]) {{.*}} -> tensor<2xi32>
// CHECK-DAG: [[ZERO:%cst.*]] = constant dense<0> : tensor<i32>
// CHECK: [[RESULT:%.*]] = "tf.Gather"([[SHAPE]], [[ZERO]]) {validate_indices = true} : (tensor<2xi32>, tensor<i32>) -> tensor<i32>
// CHECK: return [[RESULT]] : tensor<i32>
}

func @tensorlistWhileLoop(%arg0: tensor<2x3xf32>) -> tensor<*xf32> {
  %cst = constant dense<3> : tensor<1xi32>
  %cst_0 = constant dense<0> : tensor<i32>
  %cst_1 = constant dense<-1> : tensor<i32>
  %0 = "tf.TensorListFromTensor"(%arg0, %cst) : (tensor<2x3xf32>, tensor<1xi32>) -> tensor<!tf.variant<tensor<3xf32>>>
  %1:2 = "tf.While"(%cst_0, %0) {T = ["tfdtype$DT_INT32", "tfdtype$DT_VARIANT"], body = @tensorlistWhileBody, cond = @tensorlistWhileCond, is_stateless = false} : (tensor<i32>, tensor<!tf.variant<tensor<3xf32>>>) -> (tensor<i32>, tensor<!tf.variant<tensor<*xf32>>>)
  %2 = "tf.TensorListStack"(%1#1, %cst_1) : (tensor<!tf.variant<tensor<*xf32>>>, tensor<i32>) -> tensor<*xf32>
  return %2 : tensor<*xf32>

// make sure the variant types in input/output have been updated, and `T` attribute
// is removed.
// CHECK-LABEL: func @tensorlistWhileLoop
// CHECK-NOT: "tf.While"{{.*}}T =
// CHECK: "tf.While"
// CHECK-SAME: (tensor<i32>, tensor<2x3xf32>) -> (tensor<i32>, tensor<*xf32>)
// CHECK:  return %0#1 : tensor<*xf32>
}

func @tensorlistWhileBody(%arg0: tensor<i32>, %arg1: tensor<!tf.variant>) -> (tensor<i32>, tensor<!tf.variant>) {
  %0 = "tf.TensorListLength"(%arg1) : (tensor<!tf.variant>) -> tensor<i32>
  %1 = "tf.Identity"(%arg1) : (tensor<!tf.variant>) -> tensor<!tf.variant>
  return %0, %1 : tensor<i32>, tensor<!tf.variant>

// verify `body` function's signature.
// CHECK: func @tensorlistWhileBody(%[[ARG0:.*]]: tensor<i32>, %[[ARG:.*]]: tensor<*xf32>) -> (tensor<i32>, tensor<*xf32>)
// CHECK-NOT: tensor<!tf.variant>
// CHECK:  %[[LEN:.*]] = "tf.Gather"
// CHECK-NOT: tensor<!tf.variant>
// CHECK:  %[[LIST:.*]] = "tf.Identity"(%arg1) : (tensor<*xf32>) -> tensor<*xf32>
// CHECK:  return %[[LEN]], %[[LIST]] : tensor<i32>, tensor<*xf32>
}

func @tensorlistWhileCond(%arg0: tensor<i32>, %arg1: tensor<!tf.variant>) -> tensor<i1> {
  %cst = constant dense<2> : tensor<i32>
  %0 = "tf.Less"(%arg0, %cst) : (tensor<i32>, tensor<i32>) -> tensor<i1>
  return %0 : tensor<i1>

// verify `cond` function's signature.
// CHECK: func @tensorlistWhileCond(%[[ARG0:.*]]: tensor<i32>, %[[ARG1:.*]]: tensor<*xf32>) -> tensor<i1>
// CHECK:  %[[RESULT:.*]] = "tf.Less"(%[[ARG0]], {{.*}}) : (tensor<i32>, tensor<i32>) -> tensor<i1>
// CHECK:  return %[[RESULT]] : tensor<i1>
}

func @tensorlistResize(%arg0: tensor<3x10xf32>, %arg1: tensor<1xi32>, %arg2: tensor<i32>) -> tensor<?x10xf32> {
  %0 = "tf.TensorListFromTensor"(%arg0, %arg1) : (tensor<3x10xf32>, tensor<1xi32>) -> tensor<!tf.variant<tensor<10xf32>>>
  %1 = "tf.TensorListResize"(%0, %arg2) : (tensor<!tf.variant<tensor<10xf32>>>, tensor<i32>) -> tensor<!tf.variant<tensor<10xf32>>>
  %2 = "tf.TensorListStack"(%1, %arg1) : (tensor<!tf.variant<tensor<10xf32>>>, tensor<1xi32>) -> tensor<?x10xf32>
  return %2: tensor<?x10xf32>

// CHECK-LABEL: tensorlistResize
// CHECK-SAME:  ([[INPUT:%.*]]: tensor<3x10xf32>, [[ELEM_SHAPE:%.*]]: tensor<1xi32>, [[SIZE:%.*]]: tensor<i32>)
// CHECK:  [[ZERO:%.*]] = constant dense<0> : tensor<i32>
// CHECK:  [[SHAPE:%.*]] = "tf.Shape"([[INPUT]]) : (tensor<3x10xf32>) -> tensor<2xi32>
// CHECK:  [[ZERO_1:%.*]] = constant dense<0> : tensor<i32>
// CHECK:  [[INPUT_SIZE:%.*]] = "tf.Gather"([[SHAPE]], [[ZERO_1]]) {validate_indices = true} : (tensor<2xi32>, tensor<i32>) -> tensor<i32>
// CHECK:  [[SIZE_DIFF:%.*]] = "tf.Sub"([[SIZE]], [[INPUT_SIZE]]) : (tensor<i32>, tensor<i32>) -> tensor<i32>
// CHECK:  [[DIFF_RES:%.*]] = "tf.Greater"([[SIZE_DIFF]], [[ZERO]]) : (tensor<i32>, tensor<i32>) -> tensor<i1>
// CHECK:  [[SHAPE_1:%.*]] = "tf.Shape"([[INPUT]]) : (tensor<3x10xf32>) -> tensor<?xi32>
// CHECK:  [[RESULT:%.*]] = "tf.If"([[DIFF_RES]], [[INPUT]], [[SHAPE_1]], [[SIZE_DIFF]], [[SIZE]]) {else_branch = @cond_false, is_stateless = true, output_shapes = ["{}"], then_branch = @cond_true} : (tensor<i1>, tensor<3x10xf32>, tensor<?xi32>, tensor<i32>, tensor<i32>) -> tensor<?x10xf32>
// CHECK:  return [[RESULT]] : tensor<?x10xf32>
}

// CHECK-LABEL:  func @cond_true
// CHECK-SAME:  ([[INPUT:%.*]]: tensor<3x10xf32>, [[SHAPE:%.*]]: tensor<?xi32>, [[SIZE_DIFF:%.*]]: tensor<i32>, [[SIZE:%.*]]: tensor<i32>)
// CHECK-NEXT:  [[NEG_ONE:%.*]] = constant dense<-1> : tensor<1xi32>
// CHECK-NEXT:  [[ZERO:%.*]] = constant dense<0> : tensor<i32>
// CHECK-NEXT:  [[ONE:%.*]] = constant dense<1> : tensor<1xi32>
// CHECK-NEXT:  [[ELEM_SHAPE:%.*]] = "tf.Slice"([[SHAPE]], [[ONE]], [[NEG_ONE]]) : (tensor<?xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<?xi32>
// CHECK-NEXT:  [[ZERO_1:%.*]] = constant dense<0> : tensor<i32>
// CHECK-NEXT:  [[EXPANDED_SIZE_DIFF:%.*]] = "tf.ExpandDims"([[SIZE_DIFF]], [[ZERO_1]]) : (tensor<i32>, tensor<i32>) -> tensor<1xi32>
// CHECK-NEXT:  [[ZERO_2:%.*]] = constant dense<0> : tensor<i32>
// CHECK-NEXT:  [[EXTENDED_SHAPE:%.*]] = "tf.Concat"([[ZERO_2]], [[EXPANDED_SIZE_DIFF]], [[ELEM_SHAPE]]) {N = 2 : i64} : (tensor<i32>, tensor<1xi32>, tensor<?xi32>) -> tensor<2xi32>
// CHECK-NEXT:  [[ZERO_FLOAT:%.*]] = constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:  [[EXTENDED_PART:%.*]] = "tf.Fill"([[EXTENDED_SHAPE]], [[ZERO_FLOAT]]) : (tensor<2xi32>, tensor<f32>) -> tensor<?x10xf32>
// CHECK-NEXT:  [[NEG_ONE_1:%.*]] = constant dense<-1> : tensor<i32>
// CHECK-NEXT:  [[RESULT:%.*]] = "tf.Concat"([[ZERO]], [[INPUT]], [[EXTENDED_PART]]) {N = 2 : i64} : (tensor<i32>, tensor<3x10xf32>, tensor<?x10xf32>) -> tensor<?x10xf32>
// CHECK-NEXT:  return [[RESULT]] : tensor<?x10xf32>


// CHECK-LABEL:  func @cond_false
// CHECK-SAME:  ([[INPUT:%.*]]: tensor<3x10xf32>, [[SHAPE:%.*]]: tensor<?xi32>, [[SIZE_DIFF:%.*]]: tensor<i32>, [[SIZE:%.*]]: tensor<i32>)
// CHECK:  [[ZERO:%.*]] = constant dense<0> : tensor<i32>
// CHECK:  [[ONE:%.*]] = constant dense<1> : tensor<1xi32>
// CHECK:  [[RANK:%.*]] = "tf.Rank"([[INPUT]]) : (tensor<3x10xf32>) -> tensor<i32>
// CHECK:  [[ELEM_RANK:%.*]] = "tf.Sub"([[RANK]], [[ONE]]) : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
// CHECK:  [[ZERO_1:%.*]] = constant dense<0> : tensor<i32>
// CHECK:  [[PARTIAL_POS:%.*]] = "tf.Fill"([[ELEM_RANK]], [[ZERO_1]]) : (tensor<1xi32>, tensor<i32>) -> tensor<?xi32>
// CHECK:  [[ZERO_2:%.*]] = constant dense<0> : tensor<i32>
// CHECK:  [[START:%.*]] = "tf.ExpandDims"([[ZERO]], [[ZERO_2]]) : (tensor<i32>, tensor<i32>) -> tensor<1xi32>
// CHECK:  [[SLICE_BEGIN:%.*]] = "tf.Concat"([[ZERO_2]], [[START]], [[PARTIAL_POS]]) {N = 2 : i64} : (tensor<i32>, tensor<1xi32>, tensor<?xi32>) -> tensor<?xi32>
// CHECK:  [[SLICE_SIZE_HEAD:%.*]] = "tf.ExpandDims"([[SIZE]], [[ZERO_2]]) : (tensor<i32>, tensor<i32>) -> tensor<1xi32>
// CHECK:  [[NEG_ONE:%.*]] = constant dense<-1> : tensor<i32>
// CHECK:  [[SLICE_SIZE_TAIL:%.*]] = "tf.Fill"([[ELEM_RANK]], [[NEG_ONE]]) : (tensor<1xi32>, tensor<i32>) -> tensor<?xi32>
// CHECK:  [[SLICE_SIZE:%.*]] = "tf.Concat"([[ZERO_2]], [[SLICE_SIZE_HEAD]], [[SLICE_SIZE_TAIL]]) {N = 2 : i64} : (tensor<i32>, tensor<1xi32>, tensor<?xi32>) -> tensor<?xi32>
// CHECK:  [[RESULT:%.*]] = "tf.Slice"([[INPUT]], [[SLICE_BEGIN]], [[SLICE_SIZE]]) : (tensor<3x10xf32>, tensor<?xi32>, tensor<?xi32>) -> tensor<?x10xf32>
// CHECK:  return [[RESULT]] : tensor<?x10xf32>
// CHECK:  }
