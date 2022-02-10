// RUN: tf-opt "-tfl-lower-static-tensor-list=allow-tensorlist-pass-through default-to-single-batch" -split-input-file %s | FileCheck %s

// -----

// CHECK-LABEL: tensorlistConst
func @tensorlistConst(%arg0 : tensor<1xi32>) -> tensor<2x3xi32> {
  // CHECK-DAG: %[[ELEMENT0:.*]] = "tf.Const"() {value = dense<[0, 1, 2]> : tensor<3xi32>} : () -> tensor<3xi32>
  // CHECK-DAG: %[[ELEMENT1:.*]] = "tf.Const"() {value = dense<[3, 4, 5]> : tensor<3xi32>} : () -> tensor<3xi32>
  // CHECK: %[[LIST:.*]] = "tf.Pack"(%[[ELEMENT0]], %[[ELEMENT1]]) {axis = 0 : i64} : (tensor<3xi32>, tensor<3xi32>) -> tensor<2x3xi32>
  %0 = "tf.Const"() {value = opaque<"tf", "0x746674656E736F722464747970653A2044545F56415249414E542074656E736F725F7368617065207B207D2074656E736F725F636F6E74656E743A2022485C6E5C30323674656E736F72666C6F773A3A54656E736F724C6973745C3032325C3032305C3030305C3030335C3337375C3337375C3337375C3337375C3337375C3337375C3337375C3337375C3337375C3030315C3032325C3030325C3031305C3030335C3033325C725C3031305C3030335C3032325C3030345C3032325C3030325C3031305C3030333A5C3030335C3030305C3030315C3030325C3033325C725C3031305C3030335C3032325C3030345C3032325C3030325C3031305C3030333A5C3030335C3030335C3030345C30303522"> : tensor<!tf_type.variant>} : () -> tensor<!tf_type.variant<tensor<3xi32>>>

  // CHECK: return %[[LIST]]
  %1 = "tf.TensorListStack"(%0, %arg0) : (tensor<!tf_type.variant<tensor<3xi32>>>, tensor<1xi32>) -> tensor<2x3xi32>
  return %1 : tensor<2x3xi32>
}

// -----

// CHECK-LABEL: func @emptyTensorlistConst
func @emptyTensorlistConst(%arg0 : tensor<1xi32>) -> tensor<0x3xi32> {
  %0 = "tf.Const"() {value = opaque<"tf", "0x746674656E736F722464747970653A2044545F56415249414E542074656E736F725F7368617065207B207D2074656E736F725F636F6E74656E743A20222A5C6E5C30323674656E736F72666C6F773A3A54656E736F724C6973745C3032325C3032305C3030305C3030335C3337375C3337375C3337375C3337375C3337375C3337375C3337375C3337375C3337375C3030315C3032325C3030325C3031305C30303322"> : tensor<!tf_type.variant>} : () -> tensor<!tf_type.variant<tensor<3xi32>>>

  // CHECK: "tf.Const"() {value = dense<> : tensor<0x3xi32>} : () -> tensor<0x3xi32>
  // CHECK-NOT: tf.TensorListStack
  %1 = "tf.TensorListStack"(%0, %arg0) : (tensor<!tf_type.variant<tensor<3xi32>>>, tensor<1xi32>) -> tensor<0x3xi32>
  return %1 : tensor<0x3xi32>
}

// -----

// CHECK-LABEL: func @tensorlistGetItem
func @tensorlistGetItem(%arg0: tensor<3x10xf32>, %arg1: tensor<1xi32>, %arg2: tensor<i32>) -> (tensor<10xf32>, tensor<3x10xf32>) {
  %0 = "tf.TensorListFromTensor"(%arg0, %arg1) : (tensor<3x10xf32>, tensor<1xi32>) -> tensor<!tf_type.variant<tensor<10xf32>>>
  %1 = "tf.TensorListGetItem"(%0, %arg2, %arg1) : (tensor<!tf_type.variant<tensor<10xf32>>>, tensor<i32>, tensor<1xi32>) -> tensor<10xf32>
  %2 = "tf.TensorListStack"(%0, %arg1) : (tensor<!tf_type.variant<tensor<10xf32>>>, tensor<1xi32>) -> tensor<3x10xf32>
  return %1, %2 : tensor<10xf32>, tensor<3x10xf32>

// CHECK:  %0 = "tf.Gather"(%arg0, %arg2) {validate_indices = true} : (tensor<3x10xf32>, tensor<i32>) -> tensor<10xf32>
// CHECK: return %0, %arg0 : tensor<10xf32>, tensor<3x10xf32>
}

// -----

// CHECK-LABEL: func @tensorlistGetItemWithUnknownRank
func @tensorlistGetItemWithUnknownRank(%arg0: tensor<*xf32>, %arg1: tensor<1xi32>, %arg2: tensor<i32>) -> (tensor<*xf32>, tensor<*xf32>) {
  %0 = "tf.TensorListFromTensor"(%arg0, %arg1) : (tensor<*xf32>, tensor<1xi32>) -> tensor<!tf_type.variant<tensor<*xf32>>>
  %1 = "tf.TensorListGetItem"(%0, %arg2, %arg1) : (tensor<!tf_type.variant<tensor<*xf32>>>, tensor<i32>, tensor<1xi32>) -> tensor<*xf32>
  %2 = "tf.TensorListStack"(%0, %arg1) : (tensor<!tf_type.variant<tensor<*xf32>>>, tensor<1xi32>) -> tensor<*xf32>
  return %1, %2 : tensor<*xf32>, tensor<*xf32>

// CHECK:  %0 = "tf.Gather"(%arg0, %arg2) {validate_indices = true} : (tensor<*xf32>, tensor<i32>) -> tensor<*xf32>
// CHECK: return %0, %arg0 : tensor<*xf32>, tensor<*xf32>
}

// -----

// CHECK-LABEL: tensorlistStackWithConstantElementShape
func @tensorlistStackWithConstantElementShape(%arg0: tensor<?x3xf32>) -> (tensor<2x3xf32>) {
  %cst = arith.constant dense<3> : tensor<1xi32>
  %0 = "tf.TensorListFromTensor"(%arg0, %cst) : (tensor<?x3xf32>, tensor<1xi32>) -> tensor<!tf_type.variant<tensor<3xf32>>>
  %1 = "tf.TensorListStack"(%0, %cst) {num_elements = 2 : i64} : (tensor<!tf_type.variant<tensor<3xf32>>>, tensor<1xi32>) -> tensor<2x3xf32>
  return %1 : tensor<2x3xf32>

// CHECK:  [[ELEM_SHAPE:%cst.*]] = arith.constant dense<3> : tensor<1xi32>
// CHECK-NEXT:  [[SHAPE:%.*]] = "tf.Shape"(%arg0) : (tensor<?x3xf32>) -> tensor<?xi32>
// CHECK-NEXT:  [[RESHAPE:%.*]] = "tf.Reshape"(%arg0, [[SHAPE]]) : (tensor<?x3xf32>, tensor<?xi32>) -> tensor<2x3xf32>
// CHECK-NEXT:  return [[RESHAPE]] : tensor<2x3xf32>
}

// -----

// CHECK-LABEL: tensorlistSetItem
func @tensorlistSetItem(%arg0: tensor<3x10xf32>, %arg1: tensor<1xi32>, %arg2: tensor<i32>, %arg3: tensor<10xf32>) -> tensor<3x10xf32> {
  %0 = "tf.TensorListFromTensor"(%arg0, %arg1) : (tensor<3x10xf32>, tensor<1xi32>) -> tensor<!tf_type.variant<tensor<10xf32>>>
  %1 = "tf.TensorListSetItem"(%0, %arg2, %arg3) : (tensor<!tf_type.variant<tensor<10xf32>>>, tensor<i32>, tensor<10xf32>) -> tensor<!tf_type.variant<tensor<10xf32>>>
  %2 = "tf.TensorListStack"(%1, %arg1) : (tensor<!tf_type.variant<tensor<10xf32>>>, tensor<1xi32>) -> tensor<3x10xf32>
  return %2 : tensor<3x10xf32>

// CHECK-SAME:  ([[INPUT:%.*]]: tensor<3x10xf32>, [[ELEM_SHAPE:%.*]]: tensor<1xi32>, [[INDEX:%.*]]: tensor<i32>, [[ITEM:%.*]]: tensor<10xf32>)
// CHECK-DAG:  [[RANK:%.*]] = "tf.Rank"([[ITEM]]) : (tensor<10xf32>) -> tensor<i32>
// CHECK-DAG:  [[ZERO:%cst.*]] = arith.constant dense<0> : tensor<i32>
// CHECK-DAG:  [[ONE:%cst.*]] = arith.constant dense<1> : tensor<i32>
// CHECK:  [[SUFFIX_START:%.*]] = "tf.Add"([[INDEX]], [[ONE]]) : (tensor<i32>, tensor<i32>) -> tensor<i32>
// CHECK:  [[VECTOR_RANK:%.*]] = "tf.ExpandDims"([[RANK]], [[ZERO]]) : (tensor<i32>, tensor<i32>) -> tensor<1xi32>
// CHECK-DAG:  [[NEG_ONE:%cst.*]] = arith.constant dense<-1> : tensor<i32>
// CHECK-DAG:  [[ZERO_1:%cst.*]] = arith.constant dense<0> : tensor<i32>


// CHECK:  [[PARTIAL_START_POS:%.*]] = "tf.Fill"([[VECTOR_RANK]], [[ZERO_1]]) : (tensor<1xi32>, tensor<i32>) -> tensor<?xi32>
// CHECK:  [[ZERO_2:%cst.*]] = arith.constant dense<0> : tensor<i32>
// CHECK:  [[EXPANDED_START_INDEX:%.*]] = "tf.ExpandDims"([[ZERO]], [[ZERO_2]]) : (tensor<i32>, tensor<i32>) -> tensor<1xi32>
// CHECK:  [[START_POS:%.*]] = "tf.Concat"([[ZERO_2]], [[EXPANDED_START_INDEX]], [[PARTIAL_START_POS]]) : (tensor<i32>, tensor<1xi32>, tensor<?xi32>) -> tensor<?xi32>
// CHECK:  [[LEADING_DIM:%.*]] = "tf.ExpandDims"([[INDEX]], [[ZERO_2]]) : (tensor<i32>, tensor<i32>) -> tensor<1xi32>
// CHECK:  [[NEG_ONE_1:%cst.*]] = arith.constant dense<-1> : tensor<i32>
// CHECK:  [[PARTIAL_SIZE:%.*]] = "tf.Fill"([[VECTOR_RANK]], [[NEG_ONE_1]]) : (tensor<1xi32>, tensor<i32>) -> tensor<?xi32>
// CHECK:  [[SLICE_SIZE:%.*]] = "tf.Concat"([[ZERO_2]], [[LEADING_DIM]], [[PARTIAL_SIZE]]) : (tensor<i32>, tensor<1xi32>, tensor<?xi32>) -> tensor<?xi32>
// CHECK:  [[SLICE:%.*]] = "tf.Slice"([[INPUT]], [[START_POS]], [[SLICE_SIZE]]) : (tensor<3x10xf32>, tensor<?xi32>, tensor<?xi32>) -> tensor<*xf32>


// CHECK:  [[ZERO_3:%cst.*]] = arith.constant dense<0> : tensor<i32>
// CHECK:  [[PARTIAL_START_POS_1:%.*]] = "tf.Fill"([[VECTOR_RANK]], [[ZERO_3]]) : (tensor<1xi32>, tensor<i32>) -> tensor<?xi32>
// CHECK:  [[ZERO_4:%cst.*]] = arith.constant dense<0> : tensor<i32>
// CHECK:  [[EXPANDED_START_INDEX_1:%.*]] = "tf.ExpandDims"([[SUFFIX_START]], [[ZERO_4]]) : (tensor<i32>, tensor<i32>) -> tensor<1xi32>
// CHECK:  [[START_POS_1:%.*]] = "tf.Concat"([[ZERO_4]], [[EXPANDED_START_INDEX_1]], [[PARTIAL_START_POS_1]]) : (tensor<i32>, tensor<1xi32>, tensor<?xi32>) -> tensor<?xi32>
// CHECK:  [[LEADING_DIM_1:%.*]] = "tf.ExpandDims"([[NEG_ONE]], [[ZERO_4]]) : (tensor<i32>, tensor<i32>) -> tensor<1xi32>
// CHECK:  [[NEG_ONE_2:%cst.*]] = arith.constant dense<-1> : tensor<i32>
// CHECK:  [[PARTIAL_SIZE_1:%.*]] = "tf.Fill"([[VECTOR_RANK]], [[NEG_ONE_2]]) : (tensor<1xi32>, tensor<i32>) -> tensor<?xi32>
// CHECK:  [[SLICE_SIZE_1:%.*]] = "tf.Concat"([[ZERO_4]], [[LEADING_DIM_1]], [[PARTIAL_SIZE_1]]) : (tensor<i32>, tensor<1xi32>, tensor<?xi32>) -> tensor<?xi32>
// CHECK:  [[SLICE_1:%.*]] = "tf.Slice"([[INPUT]], [[START_POS_1]], [[SLICE_SIZE_1]]) : (tensor<3x10xf32>, tensor<?xi32>, tensor<?xi32>) -> tensor<*xf32>


// CHECK:  [[EXPANDED_ITEM:%.*]] = "tf.ExpandDims"([[ITEM]], [[ZERO]]) : (tensor<10xf32>, tensor<i32>) -> tensor<*xf32>
// CHECK:  [[RESULT:%.*]] = "tf.Concat"([[ZERO]], [[SLICE]], [[EXPANDED_ITEM]], [[SLICE_1]]) : (tensor<i32>, tensor<*xf32>, tensor<*xf32>, tensor<*xf32>) -> tensor<3x10xf32>
// CHECK:  return [[RESULT]] : tensor<3x10xf32>
}

// -----

// CHECK-LABEL: tensorlistSetItemWithScalarElements
func @tensorlistSetItemWithScalarElements(%arg0: tensor<5xf32>, %arg1: tensor<0xi32>, %arg2: tensor<i32>, %arg3: tensor<f32>) -> tensor<5xf32> {
  %0 = "tf.TensorListFromTensor"(%arg0, %arg1) : (tensor<5xf32>, tensor<0xi32>) -> tensor<!tf_type.variant<tensor<f32>>>
  %1 = "tf.TensorListSetItem"(%0, %arg2, %arg3) : (tensor<!tf_type.variant<tensor<f32>>>, tensor<i32>, tensor<f32>) -> tensor<!tf_type.variant<tensor<f32>>>
  %2 = "tf.TensorListStack"(%1, %arg1) : (tensor<!tf_type.variant<tensor<f32>>>, tensor<0xi32>) -> tensor<5xf32>
  return %2 : tensor<5xf32>

// CHECK-SAME:  ([[INPUT:%.*]]: tensor<5xf32>, [[ELEM_SHAPE:%.*]]: tensor<0xi32>, [[INDEX:%.*]]: tensor<i32>, [[ITEM:%.*]]: tensor<f32>)
// CHECK-DAG:  [[RANK:%.*]] = "tf.Rank"([[ITEM]]) : (tensor<f32>) -> tensor<i32>
// CHECK-DAG:  [[ZERO:%cst.*]] = arith.constant dense<0> : tensor<i32>
// CHECK-DAG:  [[ONE:%cst.*]] = arith.constant dense<1> : tensor<i32>
// CHECK:  [[SUFFIX_START:%.*]] = "tf.Add"([[INDEX]], [[ONE]]) : (tensor<i32>, tensor<i32>) -> tensor<i32>
// CHECK:  [[VECTOR_RANK:%.*]] = "tf.ExpandDims"([[RANK]], [[ZERO]]) : (tensor<i32>, tensor<i32>) -> tensor<1xi32>
// CHECK-DAG:  [[NEG_ONE:%cst.*]] = arith.constant dense<-1> : tensor<i32>
// CHECK-DAG:  [[ZERO_1:%cst.*]] = arith.constant dense<0> : tensor<i32>


// CHECK:  [[PARTIAL_START_POS:%.*]] = "tf.Fill"([[VECTOR_RANK]], [[ZERO_1]]) : (tensor<1xi32>, tensor<i32>) -> tensor<?xi32>
// CHECK:  [[ZERO_2:%cst.*]] = arith.constant dense<0> : tensor<i32>
// CHECK:  [[EXPANDED_START_INDEX:%.*]] = "tf.ExpandDims"([[ZERO]], [[ZERO_2]]) : (tensor<i32>, tensor<i32>) -> tensor<1xi32>
// CHECK:  [[START_POS:%.*]] = "tf.Concat"([[ZERO_2]], [[EXPANDED_START_INDEX]], [[PARTIAL_START_POS]]) : (tensor<i32>, tensor<1xi32>, tensor<?xi32>) -> tensor<?xi32>
// CHECK:  [[LEADING_DIM:%.*]] = "tf.ExpandDims"([[INDEX]], [[ZERO_2]]) : (tensor<i32>, tensor<i32>) -> tensor<1xi32>
// CHECK:  [[NEG_ONE_1:%cst.*]] = arith.constant dense<-1> : tensor<i32>
// CHECK:  [[PARTIAL_SIZE:%.*]] = "tf.Fill"([[VECTOR_RANK]], [[NEG_ONE_1]]) : (tensor<1xi32>, tensor<i32>) -> tensor<?xi32>
// CHECK:  [[SLICE_SIZE:%.*]] = "tf.Concat"([[ZERO_2]], [[LEADING_DIM]], [[PARTIAL_SIZE]]) : (tensor<i32>, tensor<1xi32>, tensor<?xi32>) -> tensor<?xi32>
// CHECK:  [[SLICE:%.*]] = "tf.Slice"([[INPUT]], [[START_POS]], [[SLICE_SIZE]]) : (tensor<5xf32>, tensor<?xi32>, tensor<?xi32>) -> tensor<*xf32>


// CHECK:  [[ZERO_3:%cst.*]] = arith.constant dense<0> : tensor<i32>
// CHECK:  [[PARTIAL_START_POS_1:%.*]] = "tf.Fill"([[VECTOR_RANK]], [[ZERO_3]]) : (tensor<1xi32>, tensor<i32>) -> tensor<?xi32>
// CHECK:  [[ZERO_4:%cst.*]] = arith.constant dense<0> : tensor<i32>
// CHECK:  [[EXPANDED_START_INDEX_1:%.*]] = "tf.ExpandDims"([[SUFFIX_START]], [[ZERO_4]]) : (tensor<i32>, tensor<i32>) -> tensor<1xi32>
// CHECK:  [[START_POS_1:%.*]] = "tf.Concat"([[ZERO_4]], [[EXPANDED_START_INDEX_1]], [[PARTIAL_START_POS_1]]) : (tensor<i32>, tensor<1xi32>, tensor<?xi32>) -> tensor<?xi32>
// CHECK:  [[LEADING_DIM_1:%.*]] = "tf.ExpandDims"([[NEG_ONE]], [[ZERO_4]]) : (tensor<i32>, tensor<i32>) -> tensor<1xi32>
// CHECK:  [[NEG_ONE_2:%cst.*]] = arith.constant dense<-1> : tensor<i32>
// CHECK:  [[PARTIAL_SIZE_1:%.*]] = "tf.Fill"([[VECTOR_RANK]], [[NEG_ONE_2]]) : (tensor<1xi32>, tensor<i32>) -> tensor<?xi32>
// CHECK:  [[SLICE_SIZE_1:%.*]] = "tf.Concat"([[ZERO_4]], [[LEADING_DIM_1]], [[PARTIAL_SIZE_1]])  : (tensor<i32>, tensor<1xi32>, tensor<?xi32>) -> tensor<?xi32>
// CHECK:  [[SLICE_1:%.*]] = "tf.Slice"([[INPUT]], [[START_POS_1]], [[SLICE_SIZE_1]]) : (tensor<5xf32>, tensor<?xi32>, tensor<?xi32>) -> tensor<*xf32>


// CHECK:  [[EXPANDED_ITEM:%.*]] = "tf.ExpandDims"([[ITEM]], [[ZERO]]) : (tensor<f32>, tensor<i32>) -> tensor<*xf32>
// CHECK:  [[RESULT:%.*]] = "tf.Concat"([[ZERO]], [[SLICE]], [[EXPANDED_ITEM]], [[SLICE_1]]) : (tensor<i32>, tensor<*xf32>, tensor<*xf32>, tensor<*xf32>) -> tensor<5xf32>
// CHECK:  return [[RESULT]] : tensor<5xf32>
}

// -----

// CHECK-LABEL: tensorlistReserve
func @tensorlistReserve(%arg0: tensor<3xi32>, %arg1: tensor<i32>, %arg2: tensor<i32>) -> tensor<?x?x?xf32> {
  %0 = "tf.TensorListReserve"(%arg0, %arg1) : (tensor<3xi32>, tensor<i32>) -> tensor<!tf_type.variant<tensor<?x?x?xf32>>>
  %1 = "tf.TensorListGetItem"(%0, %arg2, %arg0) : (tensor<!tf_type.variant<tensor<?x?x?xf32>>>, tensor<i32>, tensor<3xi32>) -> tensor<?x?x?xf32>
  return %1 : tensor<?x?x?xf32>

// CHECK-DAG:  [[ZERO1:%cst.*]] = arith.constant dense<0> : tensor<i32>
// CHECK-DAG:  [[ZERO2:%cst.*]] = arith.constant dense<0> : tensor<i32>
// CHECK-DAG:  [[DIM0:%.*]] = "tf.ExpandDims"(%arg1, [[ZERO1]]) : (tensor<i32>, tensor<i32>) -> tensor<1xi32>
// CHECK-DAG:  [[SHAPE:%.*]] = "tf.Concat"([[ZERO2]], [[DIM0]], %arg0) : (tensor<i32>, tensor<1xi32>, tensor<3xi32>) -> tensor<4xi32>
// CHECK-DAG:  [[VALUES:%.*]] = arith.constant dense<0.000000e+00> : tensor<f32>
// CHECK:      [[LIST:%.*]] = "tf.Fill"([[SHAPE]], [[VALUES]]) : (tensor<4xi32>, tensor<f32>) -> tensor<?x?x?x?xf32>
// CHECK:      [[RESULT:%.*]] = "tf.Gather"([[LIST]], %arg2) {validate_indices = true} : (tensor<?x?x?x?xf32>, tensor<i32>) -> tensor<?x?x?xf32>
// CHECK:      return [[RESULT]] : tensor<?x?x?xf32>
}

// -----

// CHECK-LABEL: tensorlistReserveUnrankedElements
func @tensorlistReserveUnrankedElements(%arg0: tensor<?xi32>, %arg1: tensor<i32>, %arg2: tensor<i32>) -> tensor<*xf32> {
  %0 = "tf.TensorListReserve"(%arg0, %arg1) : (tensor<?xi32>, tensor<i32>) -> tensor<!tf_type.variant<tensor<*xf32>>>
  %1 = "tf.TensorListGetItem"(%0, %arg2, %arg0) : (tensor<!tf_type.variant<tensor<*xf32>>>, tensor<i32>, tensor<?xi32>) -> tensor<*xf32>
  return %1 : tensor<*xf32>

// CHECK:  [[RESULT:%[0-9]+]] = "tf.Fill"{{.*}}(tensor<?xi32>, tensor<f32>) -> tensor<*xf32>
// CHECK:  [[RESULT2:%[0-9]+]] = "tf.Gather"{{.*}}{validate_indices = true} : (tensor<*xf32>, tensor<i32>) -> tensor<*xf32>
// CHECK:  return [[RESULT2]] : tensor<*xf32>
}

// -----

// CHECK-LABEL: tensorlistReserveConstantUnknownElementShapeDim
func @tensorlistReserveConstantUnknownElementShapeDim(%arg0: tensor<i32>, %arg1: tensor<i32>) -> tensor<?x7xf32> {
  %cst = arith.constant dense<[-1, 7]> : tensor<2xi32>
  %0 = "tf.TensorListReserve"(%cst, %arg0) : (tensor<2xi32>, tensor<i32>) -> tensor<!tf_type.variant<tensor<?x7xf32>>>
  %1 = "tf.TensorListGetItem"(%0, %arg1, %cst) : (tensor<!tf_type.variant<tensor<?x7xf32>>>, tensor<i32>, tensor<2xi32>) -> tensor<?x7xf32>
  return %1 : tensor<?x7xf32>

// CHECK-DAG:  [[ZERO1:%cst.*]] = arith.constant dense<0> : tensor<i32>
// CHECK-DAG:  [[ZERO2:%cst.*]] = arith.constant dense<0> : tensor<i32>
// CHECK-DAG:  [[ELEMENT_SHAPE:%cst.*]] = arith.constant dense<[1, 7]> : tensor<2xi32>
// CHECK-DAG:  [[DIM0:%.*]] = "tf.ExpandDims"(%arg0, [[ZERO1]]) : (tensor<i32>, tensor<i32>) -> tensor<1xi32>
// CHECK-DAG:  [[SHAPE:%.*]] = "tf.Concat"([[ZERO2]], [[DIM0]], [[ELEMENT_SHAPE]]) : (tensor<i32>, tensor<1xi32>, tensor<2xi32>) -> tensor<3xi32>
// CHECK-DAG:  [[VALUES:%.*]] = arith.constant dense<0.000000e+00> : tensor<f32>
// CHECK:      [[LIST:%.*]] = "tf.Fill"([[SHAPE]], [[VALUES]]) : (tensor<3xi32>, tensor<f32>) -> tensor<?x?x7xf32>
// CHECK:      [[RESULT:%.*]] = "tf.Gather"([[LIST]], %arg1) {validate_indices = true} : (tensor<?x?x7xf32>, tensor<i32>) -> tensor<?x7xf32>
// CHECK:      return [[RESULT]] : tensor<?x7xf32>
}

// -----

// CHECK-LABEL: tensorlistReserveUnknownElementShape
func @tensorlistReserveUnknownElementShape(%arg0: tensor<i32>, %arg1: tensor<i32>, %arg2: tensor<i32>, %arg3: tensor<2xf32>) -> tensor<*xf32> {
  %0 = "tf.TensorListReserve"(%arg0, %arg1) : (tensor<i32>, tensor<i32>) -> tensor<!tf_type.variant<tensor<*xf32>>>
  %1 = "tf.TensorListSetItem"(%0, %arg2, %arg3) : (tensor<!tf_type.variant<tensor<*xf32>>>, tensor<i32>, tensor<2xf32>) -> tensor<!tf_type.variant<tensor<2xf32>>>
  %cst = arith.constant dense<-1> : tensor<i32>
  %2 = "tf.TensorListStack"(%1, %cst) : (tensor<!tf_type.variant<tensor<2xf32>>>, tensor<i32>) -> tensor<*xf32>
  return %2 : tensor<*xf32>

// CHECK-DAG:  [[SHAPE:%[0-9]+]] = "tf.Shape"(%arg3) : (tensor<2xf32>) -> tensor<?xi32>
// CHECK-DAG:  [[CST:%.*]] = arith.constant dense<0> : tensor<i32>
// CHECK-DAG:  [[EXPAND_DIM:%[0-9]+]] = "tf.ExpandDims"(%arg1, [[CST]]) : (tensor<i32>, tensor<i32>) -> tensor<1xi32>
// CHECK-DAG:  [[CST0:%.*]] = arith.constant dense<0> : tensor<i32>
// CHECK-DAG:  [[FINAL_SHAPE:%[0-9]+]] = "tf.Concat"([[CST0]], [[EXPAND_DIM]], [[SHAPE]]) : (tensor<i32>, tensor<1xi32>, tensor<?xi32>) -> tensor<?xi32>
// CHECK-DAG:  [[CST1:%.*]] = arith.constant dense<0.000000e+00> : tensor<f32>
// CHECK:  [[FILL:%[0-9]+]] = "tf.Fill"([[FINAL_SHAPE]], [[CST1]]) : (tensor<?xi32>, tensor<f32>) -> tensor<*xf32>
}

// -----

// CHECK-LABEL: tensorlistReserveUnrankedElementShape
func @tensorlistReserveUnrankedElementShape(%arg0: tensor<*xi32>, %arg1: tensor<i32>, %arg2: tensor<i32>) -> tensor<*xf32> {
  %0 = "tf.TensorListReserve"(%arg0, %arg1) : (tensor<*xi32>, tensor<i32>) -> tensor<!tf_type.variant<tensor<*xf32>>>
  %1 = "tf.TensorListGetItem"(%0, %arg2, %arg0) : (tensor<!tf_type.variant<tensor<*xf32>>>, tensor<i32>, tensor<*xi32>) -> tensor<*xf32>
  return %1 : tensor<*xf32>

// CHECK-DAG:  [[AXIS:%.*]] = arith.constant dense<0> : tensor<i32>
// CHECK-DAG:  [[EXPAND_DIM:%[0-9]+]] = "tf.ExpandDims"(%arg1, [[AXIS]]) : (tensor<i32>, tensor<i32>) -> tensor<1xi32>
// CHECK-DAG:  [[AXIS_1:%.*]] = arith.constant dense<0> : tensor<i32>
// CHECK-DAG:  [[CONCAT:%.*]] = "tf.Concat"([[AXIS_1]], [[EXPAND_DIM]], %arg0) : (tensor<i32>, tensor<1xi32>, tensor<*xi32>) -> tensor<?xi32>
// CHECK:  [[CST:%.*]] = arith.constant dense<0.000000e+00> : tensor<f32>
// CHECK:  [[FILL:%.*]] = "tf.Fill"([[CONCAT]], [[CST]]) : (tensor<?xi32>, tensor<f32>) -> tensor<*xf32>
// CHECK:  [[GATHER:%.*]] = "tf.Gather"([[FILL]], %arg2) {validate_indices = true} : (tensor<*xf32>, tensor<i32>) -> tensor<*xf32>
// CHECK:  return [[GATHER]] : tensor<*xf32>
}

// -----

func @EmptyTensorList(%arg0: tensor<3xi32>, %arg1: tensor<i32>, %arg2: tensor<i32>) -> tensor<?x?x?xf32> {
  %0 = "tf.EmptyTensorList"(%arg0, %arg1) : (tensor<3xi32>, tensor<i32>) -> tensor<!tf_type.variant<tensor<?x?x?xf32>>>
  %1 = "tf.TensorListGetItem"(%0, %arg2, %arg0) : (tensor<!tf_type.variant<tensor<?x?x?xf32>>>, tensor<i32>, tensor<3xi32>) -> tensor<?x?x?xf32>
  return %1 : tensor<?x?x?xf32>

// CHECK-LABEL: EmptyTensorList
// CHECK-SAME:  ([[ELEM_SHAPE:%.*]]: tensor<3xi32>, [[MAX_ELEMS:%.*]]: tensor<i32>, [[IDX:%.*]]: tensor<i32>)
// CHECK-DAG:  [[DIM0:%cst.*]] = arith.constant dense<0> : tensor<1xi32>
// CHECK-DAG:  [[ZERO:%cst.*]] = arith.constant dense<0> : tensor<i32>
// CHECK-DAG:  [[SHAPE:%.*]] = "tf.Concat"([[ZERO]], [[DIM0]], [[ELEM_SHAPE]]) : (tensor<i32>, tensor<1xi32>, tensor<3xi32>) -> tensor<4xi32>
// CHECK-DAG:  [[VALUES:%.*]] = arith.constant dense<0.000000e+00> : tensor<f32>
// CHECK:      [[LIST:%.*]] = "tf.Fill"([[SHAPE]], [[VALUES]]) : (tensor<4xi32>, tensor<f32>) -> tensor<0x?x?x?xf32>
// CHECK:      [[RESULT:%.*]] = "tf.Gather"([[LIST]], [[IDX]]) {validate_indices = true} : (tensor<0x?x?x?xf32>, tensor<i32>) -> tensor<?x?x?xf32>
// CHECK:      return [[RESULT]] : tensor<?x?x?xf32>
}

// -----

func @tensorlistPushBack(%arg0: tensor<3x10xf32>, %arg1: tensor<1xi32>, %arg2: tensor<10xf32>) -> tensor<?x10xf32> {
  %0 = "tf.TensorListFromTensor"(%arg0, %arg1) : (tensor<3x10xf32>, tensor<1xi32>) -> tensor<!tf_type.variant<tensor<10xf32>>>
  %1 = "tf.TensorListPushBack"(%0, %arg2) : (tensor<!tf_type.variant<tensor<10xf32>>>, tensor<10xf32>) -> tensor<!tf_type.variant<tensor<10xf32>>>
  %2 = "tf.TensorListStack"(%1, %arg1) : (tensor<!tf_type.variant<tensor<10xf32>>>, tensor<1xi32>) -> tensor<?x10xf32>
  return %2 : tensor<?x10xf32>

// CHECK-LABEL: tensorlistPushBack
// CHECK-SAME:  ([[INPUT:%.*]]: tensor<3x10xf32>, [[ELEM_SHAPE:%.*]]: tensor<1xi32>, [[ITEM:%.*]]: tensor<10xf32>)
// CHECK:   [[ZERO:%.*]] = arith.constant dense<0> : tensor<i32>
// CHECK:   [[EXP_ITEM:%.*]] = "tf.ExpandDims"([[ITEM]], [[ZERO]]) {{.*}} -> tensor<1x10xf32>
// CHECK:   [[RESULT:%.*]] = "tf.Concat"(%cst, [[INPUT]], [[EXP_ITEM]]) : {{.*}} -> tensor<?x10xf32>
// CHECK:   return [[RESULT]] : tensor<?x10xf32>
}

// -----

func @tensorlistLength(%arg0: tensor<3x10xf32>, %arg1: tensor<1xi32>) -> (tensor<i32>) {
  %0 = "tf.TensorListFromTensor"(%arg0, %arg1) : (tensor<3x10xf32>, tensor<1xi32>) -> tensor<!tf_type.variant<tensor<10xf32>>>
  %1 = "tf.TensorListLength"(%0) : (tensor<!tf_type.variant<tensor<10xf32>>>) -> tensor<i32>
  return %1: tensor<i32>

// CHECK-LABEL: tensorlistLength
// CHECK-SAME: ([[INPUT:%.*]]: tensor<3x10xf32>, [[ELEM_SHAPE:%.*]]: tensor<1xi32>)
// CHECK-DAG: [[SHAPE:%.*]] = "tf.Shape"([[INPUT]]) {{.*}} -> tensor<2xi32>
// CHECK-DAG: [[ZERO:%cst.*]] = arith.constant dense<0> : tensor<i32>
// CHECK: [[RESULT:%.*]] = "tf.Gather"([[SHAPE]], [[ZERO]]) {validate_indices = true} : (tensor<2xi32>, tensor<i32>) -> tensor<i32>
// CHECK: return [[RESULT]] : tensor<i32>
}

// -----

func @tensorlistWhileLoop(%arg0: tensor<2x3xf32>) -> tensor<2x3xf32> {
  %cst = arith.constant dense<3> : tensor<1xi32>
  %cst_0 = arith.constant dense<0> : tensor<i32>
  %cst_1 = arith.constant dense<-1> : tensor<i32>
  %0 = "tf.TensorListFromTensor"(%arg0, %cst) : (tensor<2x3xf32>, tensor<1xi32>) -> tensor<!tf_type.variant<tensor<3xf32>>>
  %1:2 = "tf.While"(%cst_0, %0) {T = ["tfdtype$DT_INT32", "tfdtype$DT_VARIANT"], body = @tensorlistWhileBody, cond = @tensorlistWhileCond, is_stateless = false} : (tensor<i32>, tensor<!tf_type.variant<tensor<3xf32>>>) -> (tensor<i32>, tensor<!tf_type.variant<tensor<*xf32>>>)
  %2 = "tf.TensorListStack"(%1#1, %cst_1) : (tensor<!tf_type.variant<tensor<*xf32>>>, tensor<i32>) -> tensor<2x3xf32>
  return %2 : tensor<2x3xf32>

// make sure the variant types in input/output have been updated, and `T` attribute
// is removed.
// CHECK-LABEL: func @tensorlistWhileLoop
// CHECK-NOT: "tf.While"{{.*}}T =
// CHECK: "tf.While"
// CHECK-SAME: (tensor<i32>, tensor<2x3xf32>) -> (tensor<i32>, tensor<2x3xf32>)
// CHECK:  return %0#1 : tensor<2x3xf32>
}

func @tensorlistWhileBody(%arg0: tensor<i32>, %arg1: tensor<!tf_type.variant>) -> (tensor<i32>, tensor<!tf_type.variant>) {
  %0 = "tf.TensorListLength"(%arg1) : (tensor<!tf_type.variant>) -> tensor<i32>
  %1 = "tf.Identity"(%arg1) : (tensor<!tf_type.variant>) -> tensor<!tf_type.variant>
  return %0, %1 : tensor<i32>, tensor<!tf_type.variant>

// verify `body` function's signature.
// CHECK: func @tensorlistWhileBody(%[[ARG0:.*]]: tensor<i32>, %[[ARG:.*]]: tensor<2x3xf32>) -> (tensor<i32>, tensor<2x3xf32>)
// CHECK-NOT: tensor<!tf_type.variant>
// CHECK:  %[[LEN:.*]] = "tf.Gather"
// CHECK-NOT: tensor<!tf_type.variant>
// CHECK:  %[[LIST:.*]] = "tf.Identity"(%arg1) : (tensor<2x3xf32>) -> tensor<2x3xf32>
// CHECK:  return %[[LEN]], %[[LIST]] : tensor<i32>, tensor<2x3xf32>
}

func @tensorlistWhileCond(%arg0: tensor<i32>, %arg1: tensor<!tf_type.variant>) -> tensor<i1> {
  %cst = arith.constant dense<2> : tensor<i32>
  %0 = "tf.Less"(%arg0, %cst) : (tensor<i32>, tensor<i32>) -> tensor<i1>
  return %0 : tensor<i1>

// verify `cond` function's signature.
// CHECK: func @tensorlistWhileCond(%[[ARG0:.*]]: tensor<i32>, %[[ARG1:.*]]: tensor<2x3xf32>) -> tensor<i1>
// CHECK:  %[[RESULT:.*]] = "tf.Less"(%[[ARG0]], {{.*}}) : (tensor<i32>, tensor<i32>) -> tensor<i1>
// CHECK:  return %[[RESULT]] : tensor<i1>
}

// -----

// CHECK-LABEL: func @tensorlistWhileRegion
func @tensorlistWhileRegion(%arg0: tensor<2x3xf32>) -> tensor<*xf32> {
  %cst = arith.constant dense<3> : tensor<1xi32>
  %cst_0 = arith.constant dense<0> : tensor<i32>
  %cst_1 = arith.constant dense<-1> : tensor<i32>
  %0 = "tf.TensorListFromTensor"(%arg0, %cst) : (tensor<2x3xf32>, tensor<1xi32>) -> tensor<!tf_type.variant<tensor<3xf32>>>
  // CHECK: "tf.WhileRegion"
  %1:2 = "tf.WhileRegion"(%cst_0, %0) ({
      ^bb0(%carg0: tensor<i32>, %carg1: tensor<!tf_type.variant>):
       %cst_2 = arith.constant dense<2> : tensor<i32>
       %1 = "tf.Less"(%carg0, %cst_2) : (tensor<i32>, tensor<i32>) -> tensor<i1>
       "tf.Yield"(%1) : (tensor<i1>) -> ()

      // verify condition types
      // CHECK: ^bb0(%[[CARG0:.*]]: tensor<i32>, %[[CARG1:.*]]: tensor<*xf32>):
      // CHECK:  %[[COND:.*]] = "tf.Less"(%[[CARG0]], {{.*}}) : (tensor<i32>, tensor<i32>) -> tensor<i1>
      // CHECK:  "tf.Yield"(%[[COND]]) : (tensor<i1>) -> ()

    },
    {
      ^bb0(%barg0: tensor<i32>, %barg1: tensor<!tf_type.variant>):
       %1 = "tf.TensorListLength"(%barg1) : (tensor<!tf_type.variant>) -> tensor<i32>
       "tf.Yield"(%1, %barg1) : (tensor<i32>, tensor<!tf_type.variant>) -> ()

      // verify body types
      // CHECK: ^bb0(%[[BARG0:.*]]: tensor<i32>, %[[BARG1:.*]]: tensor<*xf32>):
      // CHECK-NOT: tensor<!tf_type.variant>
      // CHECK:  %[[LEN:.*]] = "tf.Gather"
      // CHECK-NOT: tensor<!tf_type.variant>
      // CHECK:  "tf.Yield"(%[[LEN]], %[[BARG1]]) : (tensor<i32>, tensor<*xf32>) -> ()

  }) {is_stateless = false} : (tensor<i32>, tensor<!tf_type.variant<tensor<3xf32>>>) -> (tensor<i32>, tensor<!tf_type.variant<tensor<*xf32>>>)
  // make sure the variant types in input/output have been updated
  // CHECK: {is_stateless = false} : (tensor<i32>, tensor<2x3xf32>) -> (tensor<i32>, tensor<*xf32>)
  %2 = "tf.TensorListStack"(%1#1, %cst_1) : (tensor<!tf_type.variant<tensor<*xf32>>>, tensor<i32>) -> tensor<*xf32>
  // CHECK:  return %0#1 : tensor<*xf32>
  return %2 : tensor<*xf32>
}

// -----

func @otherVariantWhileLoop(%arg0: tensor<1xi32>) -> tensor<1xi32> {
  %0 = "tf.Const"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
  %1 = "tf.Const"() {value = dense<-1> : tensor<i32>} : () -> tensor<i32>
  %2 = "tf.EmptyTensorMap"() {device = ""} : () -> tensor<!tf_type.variant>
  %3:4 = "tf.While"(%0, %1, %0, %2) {_lower_using_switch_merge = true, _num_original_outputs = 4 : i64, _read_only_resource_inputs = [], body = @otherVariantWhileBody, cond = @otherVariantWhileCond, device = "", is_stateless = true, parallel_iterations = 10 : i64, shape_invariant} : (tensor<i32>, tensor<i32>, tensor<i32>, tensor<!tf_type.variant>) -> (tensor<i32>, tensor<i32>, tensor<i32>, tensor<!tf_type.variant>)
  %4 = "tf.Identity"(%3#3) {device = ""} : (tensor<!tf_type.variant>) -> tensor<!tf_type.variant>
  %5 = "tf.TensorMapSize"(%4) {device = ""} : (tensor<!tf_type.variant>) -> tensor<i32>
  %6 = "tf.AddV2"(%arg0, %5) {device = ""} : (tensor<1xi32>, tensor<i32>) -> tensor<1xi32>
  return %6 : tensor<1xi32>
}

// Make sure the non TensorList variant types in input/output have remained.
// CHECK-LABEL: otherVariantWhileLoop
// CHECK: "tf.While"
// CHECK-SAME: (tensor<i32>, tensor<i32>, tensor<i32>, tensor<!tf_type.variant>) -> (tensor<i32>, tensor<i32>, tensor<i32>, tensor<!tf_type.variant>)

func @otherVariantWhileBody(%arg0: tensor<i32>, %arg1: tensor<i32>, %arg2: tensor<i32>, %arg3: tensor<!tf_type.variant>) -> (tensor<i32>, tensor<i32>, tensor<i32>, tensor<!tf_type.variant>) {
  %0 = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
  %1 = "tf.AddV2"(%arg2, %0) {device = ""} : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %2 = "tf.TensorMapInsert"(%arg3, %arg2, %arg2) {device = "", key_dtype = i32, value_dtype = i32} : (tensor<!tf_type.variant>, tensor<i32>, tensor<i32>) -> tensor<!tf_type.variant>
  %3 = "tf.AddV2"(%arg0, %0) {device = ""} : (tensor<i32>, tensor<i32>) -> tensor<i32>
  return %3, %arg1, %1, %2 : tensor<i32>, tensor<i32>, tensor<i32>, tensor<!tf_type.variant>
}

// Verify `body` function's signature.
// CHECK-LABEL: func @otherVariantWhileBody
// CHECK:       [[CST:%.*]] = "tf.Const"()
// CHECK-NEXT:  [[ADD:%.*]] = "tf.AddV2"(%arg2, [[CST]])
// CHECK-NEXT:  [[TENSOR_MAP_INSERT_RESULT:%.*]] = "tf.TensorMapInsert"(%arg3, %arg2, %arg2)
// CHECK-NEXT:  [[ADD_2:%.*]] = "tf.AddV2"(%arg0, [[CST]])
// CHECK-NEXT:  return [[ADD_2]], %arg1, [[ADD]], [[TENSOR_MAP_INSERT_RESULT]]

func @otherVariantWhileCond(%arg0: tensor<i32>, %arg1: tensor<i32>, %arg2: tensor<i32>, %arg3: tensor<!tf_type.variant>) -> tensor<i1> {
  %0 = "tf.Const"() {value = dense<10> : tensor<i32>} : () -> tensor<i32>
  %1 = "tf.Less"(%arg2, %0) {device = ""} : (tensor<i32>, tensor<i32>) -> tensor<i1>
  return %1 : tensor<i1>
}

// Verify `cond` function's signature.
// CHECK-LABEL: func @otherVariantWhileCond
// CHECK:       [[CST:%.*]] = "tf.Const"()
// CHECK-NEXT:  [[LESS:%.*]] = "tf.Less"(%arg2, [[CST]])
// CHECK-NEXT:  return [[LESS]]

// -----

func @tensorlistResize(%arg0: tensor<3x10xf32>, %arg1: tensor<1xi32>, %arg2: tensor<i32>) -> tensor<?x10xf32> {
  %0 = "tf.TensorListFromTensor"(%arg0, %arg1) : (tensor<3x10xf32>, tensor<1xi32>) -> tensor<!tf_type.variant<tensor<10xf32>>>
  %1 = "tf.TensorListResize"(%0, %arg2) : (tensor<!tf_type.variant<tensor<10xf32>>>, tensor<i32>) -> tensor<!tf_type.variant<tensor<10xf32>>>
  %2 = "tf.TensorListStack"(%1, %arg1) : (tensor<!tf_type.variant<tensor<10xf32>>>, tensor<1xi32>) -> tensor<?x10xf32>
  return %2: tensor<?x10xf32>

// CHECK-LABEL: tensorlistResize
// CHECK-SAME:  ([[INPUT:%.*]]: tensor<3x10xf32>, [[ELEM_SHAPE:%.*]]: tensor<1xi32>, [[SIZE:%.*]]: tensor<i32>)
// CHECK:  [[ZERO:%.*]] = arith.constant dense<0> : tensor<i32>
// CHECK:  [[SHAPE:%.*]] = "tf.Shape"([[INPUT]]) : (tensor<3x10xf32>) -> tensor<2xi32>
// CHECK:  [[ZERO_1:%.*]] = arith.constant dense<0> : tensor<i32>
// CHECK:  [[INPUT_SIZE:%.*]] = "tf.Gather"([[SHAPE]], [[ZERO_1]]) {validate_indices = true} : (tensor<2xi32>, tensor<i32>) -> tensor<i32>
// CHECK:  [[SIZE_DIFF:%.*]] = "tf.Sub"([[SIZE]], [[INPUT_SIZE]]) : (tensor<i32>, tensor<i32>) -> tensor<i32>
// CHECK:  [[DIFF_RES:%.*]] = "tf.Greater"([[SIZE_DIFF]], [[ZERO]]) : (tensor<i32>, tensor<i32>) -> tensor<i1>
// CHECK:  [[SHAPE_1:%.*]] = "tf.Shape"([[INPUT]]) : (tensor<3x10xf32>) -> tensor<?xi32>
// CHECK:  [[RESULT:%.*]] = "tf.If"([[DIFF_RES]], [[INPUT]], [[SHAPE_1]], [[SIZE_DIFF]], [[SIZE]]) {else_branch = @cond_false, is_stateless = true, then_branch = @cond_true} : (tensor<i1>, tensor<3x10xf32>, tensor<?xi32>, tensor<i32>, tensor<i32>) -> tensor<?x10xf32>
// CHECK:  return [[RESULT]] : tensor<?x10xf32>
}

// CHECK-LABEL:  func private @cond_true
// CHECK-SAME:  ([[INPUT:%.*]]: tensor<3x10xf32>, [[SHAPE:%.*]]: tensor<?xi32>, [[SIZE_DIFF:%.*]]: tensor<i32>, [[SIZE:%.*]]: tensor<i32>)
// CHECK-NEXT:  [[NEG_ONE:%.*]] = arith.constant dense<-1> : tensor<1xi32>
// CHECK-NEXT:  [[ZERO:%.*]] = arith.constant dense<0> : tensor<i32>
// CHECK-NEXT:  [[ONE:%.*]] = arith.constant dense<1> : tensor<1xi32>
// CHECK-NEXT:  [[ELEM_SHAPE:%.*]] = "tf.Slice"([[SHAPE]], [[ONE]], [[NEG_ONE]]) : (tensor<?xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<?xi32>
// CHECK-NEXT:  [[ZERO_1:%.*]] = arith.constant dense<0> : tensor<i32>
// CHECK-NEXT:  [[EXPANDED_SIZE_DIFF:%.*]] = "tf.ExpandDims"([[SIZE_DIFF]], [[ZERO_1]]) : (tensor<i32>, tensor<i32>) -> tensor<1xi32>
// CHECK-NEXT:  [[ZERO_2:%.*]] = arith.constant dense<0> : tensor<i32>
// CHECK-NEXT:  [[EXTENDED_SHAPE:%.*]] = "tf.Concat"([[ZERO_2]], [[EXPANDED_SIZE_DIFF]], [[ELEM_SHAPE]]) : (tensor<i32>, tensor<1xi32>, tensor<?xi32>) -> tensor<2xi32>
// CHECK-NEXT:  [[ZERO_FLOAT:%.*]] = arith.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:  [[EXTENDED_PART:%.*]] = "tf.Fill"([[EXTENDED_SHAPE]], [[ZERO_FLOAT]]) : (tensor<2xi32>, tensor<f32>) -> tensor<?x10xf32>
// CHECK-NEXT:  [[NEG_ONE_1:%.*]] = arith.constant dense<-1> : tensor<i32>
// CHECK-NEXT:  [[RESULT:%.*]] = "tf.Concat"([[ZERO]], [[INPUT]], [[EXTENDED_PART]]) : (tensor<i32>, tensor<3x10xf32>, tensor<?x10xf32>) -> tensor<?x10xf32>
// CHECK-NEXT:  return [[RESULT]] : tensor<?x10xf32>


// CHECK-LABEL:  func private @cond_false
// CHECK-SAME:  ([[INPUT:%.*]]: tensor<3x10xf32>, [[SHAPE:%.*]]: tensor<?xi32>, [[SIZE_DIFF:%.*]]: tensor<i32>, [[SIZE:%.*]]: tensor<i32>)
// CHECK-DAG:  [[ZERO:%.*]] = arith.constant dense<0> : tensor<i32>
// CHECK-DAG:  [[ONE:%.*]] = arith.constant dense<1> : tensor<1xi32>
// CHECK:  [[RANK:%.*]] = "tf.Rank"([[INPUT]]) : (tensor<3x10xf32>) -> tensor<i32>
// CHECK:  [[ELEM_RANK:%.*]] = "tf.Sub"([[RANK]], [[ONE]]) : (tensor<i32>, tensor<1xi32>) -> tensor<1xi32>
// CHECK:  [[ZERO_1:%.*]] = arith.constant dense<0> : tensor<i32>
// CHECK:  [[PARTIAL_POS:%.*]] = "tf.Fill"([[ELEM_RANK]], [[ZERO_1]]) : (tensor<1xi32>, tensor<i32>) -> tensor<?xi32>
// CHECK:  [[ZERO_2:%.*]] = arith.constant dense<0> : tensor<i32>
// CHECK:  [[START:%.*]] = "tf.ExpandDims"([[ZERO]], [[ZERO_2]]) : (tensor<i32>, tensor<i32>) -> tensor<1xi32>
// CHECK:  [[SLICE_BEGIN:%.*]] = "tf.Concat"([[ZERO_2]], [[START]], [[PARTIAL_POS]]) : (tensor<i32>, tensor<1xi32>, tensor<?xi32>) -> tensor<?xi32>
// CHECK:  [[SLICE_SIZE_HEAD:%.*]] = "tf.ExpandDims"([[SIZE]], [[ZERO_2]]) : (tensor<i32>, tensor<i32>) -> tensor<1xi32>
// CHECK:  [[NEG_ONE:%.*]] = arith.constant dense<-1> : tensor<i32>
// CHECK:  [[SLICE_SIZE_TAIL:%.*]] = "tf.Fill"([[ELEM_RANK]], [[NEG_ONE]]) : (tensor<1xi32>, tensor<i32>) -> tensor<?xi32>
// CHECK:  [[SLICE_SIZE:%.*]] = "tf.Concat"([[ZERO_2]], [[SLICE_SIZE_HEAD]], [[SLICE_SIZE_TAIL]]) : (tensor<i32>, tensor<1xi32>, tensor<?xi32>) -> tensor<?xi32>
// CHECK:  [[RESULT:%.*]] = "tf.Slice"([[INPUT]], [[SLICE_BEGIN]], [[SLICE_SIZE]]) : (tensor<3x10xf32>, tensor<?xi32>, tensor<?xi32>) -> tensor<?x10xf32>
// CHECK:  return [[RESULT]] : tensor<?x10xf32>
// CHECK:  }

// -----

// CHECK-LABEL: tensorlistReserveWithDynamicShape
func @tensorlistReserveWithDynamicShape(%arg0: tensor<i32>, %arg1: tensor<i32>, %arg2: tensor<i32>) -> tensor<?x?x?xf32> {
  %0 = "tf.TensorListReserve"(%arg0, %arg1) : (tensor<i32>, tensor<i32>) -> tensor<!tf_type.variant<tensor<?x?x?xf32>>>
  %1 = "tf.TensorListGetItem"(%0, %arg2, %arg0) : (tensor<!tf_type.variant<tensor<?x?x?xf32>>>, tensor<i32>, tensor<i32>) -> tensor<?x?x?xf32>
  return %1 : tensor<?x?x?xf32>

// CHECK: %0 = "tf.TensorListReserve"(%arg0, %arg1) : (tensor<i32>, tensor<i32>) -> tensor<!tf_type.variant<tensor<?x?x?xf32>>>
// CHECK: %1 = "tf.TensorListGetItem"(%0, %arg2, %arg0) : (tensor<!tf_type.variant<tensor<?x?x?xf32>>>, tensor<i32>, tensor<i32>) -> tensor<?x?x?xf32>
// CHECK: return %1 : tensor<?x?x?xf32>
}

// -----

// CHECK-LABEL: tensorlistConcat
func @tensorlistConcat(%arg0: tensor<3x2x2xf32>, %lead: tensor<i64>) -> (tensor<?x2xf32>, tensor<0xi64>) {
  %cst = arith.constant dense<[2, 2]> : tensor<2xi32>
  %list = "tf.TensorListFromTensor"(%arg0, %cst) : (tensor<3x2x2xf32>, tensor<2xi32>) -> tensor<!tf_type.variant<tensor<f32>>>
  %t:2 = "tf.TensorListConcatV2"(%list, %cst, %lead) : (tensor<!tf_type.variant<tensor<f32>>>, tensor<2xi32>, tensor<i64>) -> (tensor<?x2xf32>, tensor<0xi64>)
  return %t#0, %t#1 : tensor<?x2xf32>, tensor<0xi64>

// CHECK: [[ELEMENT_SHAPE:%.*]] = arith.constant dense<2> : tensor<2xi32>
// CHECK: [[UNPACK:%.*]]:3 = "tf.Unpack"(%arg0) {axis = 0 : i64} : (tensor<3x2x2xf32>) -> (tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>)
// CHECK: [[SCALAR_ZERO:%.*]] = arith.constant dense<0> : tensor<i32>
// CHECK: [[CONCAT:%.*]] = "tf.Concat"([[SCALAR_ZERO]], [[UNPACK]]#0, [[UNPACK]]#1, [[UNPACK]]#2) : (tensor<i32>, tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<?x2xf32>
// CHECK: [[LENGTHS:%.*]] = arith.constant dense<0> : tensor<0xi64>
// CHECK: return [[CONCAT]], [[LENGTHS]] : tensor<?x2xf32>, tensor<0xi64>
}

// -----

func @whileLoopWithDynamicTensorList(%arg0: tensor<i32>, %arg1: tensor<i32>) -> tensor<*xf32> {
  %cst = arith.constant dense<3> : tensor<1xi32>
  %cst_0 = arith.constant dense<0> : tensor<i32>
  %cst_1 = arith.constant dense<-1> : tensor<i32>
  %0 = "tf.TensorListReserve"(%arg0, %arg1) : (tensor<i32>, tensor<i32>) -> tensor<!tf_type.variant<tensor<?x?xf32>>>
  %1:2 = "tf.While"(%cst_0, %0) {T = ["tfdtype$DT_INT32", "tfdtype$DT_VARIANT"], body = @tensorlistWhileBody, cond = @tensorlistWhileCond, is_stateless = false} : (tensor<i32>, tensor<!tf_type.variant<tensor<?x?xf32>>>) -> (tensor<i32>, tensor<!tf_type.variant<tensor<*xf32>>>)
  %2 = "tf.TensorListStack"(%1#1, %cst_1) : (tensor<!tf_type.variant<tensor<*xf32>>>, tensor<i32>) -> tensor<*xf32>
  return %2 : tensor<*xf32>

// verify tensorlist ops pass through.
// CHECK-LABEL: func @whileLoopWithDynamicTensorList
// CHECK: "tf.TensorListReserve"
// CHECK: "tf.While"
// CHECK-SAME: (tensor<i32>, tensor<!tf_type.variant<tensor<?x?xf32>>>) -> (tensor<i32>, tensor<!tf_type.variant<tensor<*xf32>>>)
// CHECK: "tf.TensorListStack"
}

func @tensorlistWhileBody(%arg0: tensor<i32>, %arg1: tensor<!tf_type.variant>) -> (tensor<i32>, tensor<!tf_type.variant>) {
  %0 = "tf.TensorListLength"(%arg1) : (tensor<!tf_type.variant>) -> tensor<i32>
  %1 = "tf.Identity"(%arg1) : (tensor<!tf_type.variant>) -> tensor<!tf_type.variant>
  return %0, %1 : tensor<i32>, tensor<!tf_type.variant>

// verify `body` function's signature stays unchanged.
// CHECK: func @tensorlistWhileBody(%[[ARG0:.*]]: tensor<i32>, %[[ARG:.*]]: tensor<!tf_type.variant>) -> (tensor<i32>, tensor<!tf_type.variant>)
}

func @tensorlistWhileCond(%arg0: tensor<i32>, %arg1: tensor<!tf_type.variant>) -> tensor<i1> {
  %cst = arith.constant dense<2> : tensor<i32>
  %0 = "tf.Less"(%arg0, %cst) : (tensor<i32>, tensor<i32>) -> tensor<i1>
  return %0 : tensor<i1>

// verify `cond` function's signature stays unchanged.
// CHECK: func @tensorlistWhileCond(%[[ARG0:.*]]: tensor<i32>, %[[ARG1:.*]]: tensor<!tf_type.variant>) -> tensor<i1>
}

// -----

func @tensorListIf(%arg0: tensor<3x10xf32>, %arg1: tensor<1xi32>, %arg2: tensor<i32>) -> tensor<3x10xf32> {
  %cst = arith.constant dense<2> : tensor<i32>
  %0 = "tf.TensorListFromTensor"(%arg0, %arg1) : (tensor<3x10xf32>, tensor<1xi32>) -> tensor<!tf_type.variant<tensor<10xf32>>>
  %1 = "tf.Less"(%arg2, %cst) : (tensor<i32>, tensor<i32>) -> tensor<i1>
  %res = "tf.If"(%1, %0) {else_branch = @tensorListIfCondFalse, is_stateless = true, then_branch = @tensorListIfCondTrue} : (tensor<i1>, tensor<!tf_type.variant<tensor<10xf32>>>) -> tensor<!tf_type.variant<tensor<10xf32>>>
  %2 = "tf.TensorListStack"(%res, %arg1) : (tensor<!tf_type.variant<tensor<10xf32>>>, tensor<1xi32>) -> tensor<3x10xf32>
  return %2 : tensor<3x10xf32>

// CHECK: func @tensorListIf
// CHECK-NEXT:  %cst = arith.constant dense<2> : tensor<i32>
// CHECK-NEXT:  %0 = "tf.Less"(%arg2, %cst) : (tensor<i32>, tensor<i32>) -> tensor<i1>
// CHECK-NEXT:  %1 = "tf.If"(%0, %arg0) {else_branch = @tensorListIfCondFalse, is_stateless = true, then_branch = @tensorListIfCondTrue} : (tensor<i1>, tensor<3x10xf32>) -> tensor<3x10xf32>
// CHECK-NEXT:  return %1 : tensor<3x10xf32>
}

func @tensorListIfCondTrue(%arg0: tensor<!tf_type.variant<tensor<10xf32>>>) -> (tensor<!tf_type.variant<tensor<10xf32>>>) {
  %cst = arith.constant dense<0> : tensor<i32>
  %cst_0 = arith.constant dense<0.0> : tensor<10xf32>
  %0 = "tf.TensorListSetItem"(%arg0, %cst, %cst_0) : (tensor<!tf_type.variant<tensor<10xf32>>>, tensor<i32>, tensor<10xf32>) -> tensor<!tf_type.variant<tensor<10xf32>>>
  return %0 : tensor<!tf_type.variant<tensor<10xf32>>>

// CHECK: func @tensorListIfCondTrue(%arg0: tensor<3x10xf32>) -> tensor<3x10xf32>
// CHECK-NOT: tf.TensorListSetItem
}

func @tensorListIfCondFalse(%arg0: tensor<!tf_type.variant<tensor<10xf32>>>) -> (tensor<!tf_type.variant<tensor<10xf32>>>) {
  %cst = arith.constant dense<0> : tensor<i32>
  %cst_0 = arith.constant dense<1.0> : tensor<10xf32>
  %0 = "tf.TensorListSetItem"(%arg0, %cst, %cst_0) : (tensor<!tf_type.variant<tensor<10xf32>>>, tensor<i32>, tensor<10xf32>) -> tensor<!tf_type.variant<tensor<10xf32>>>
  return %0 : tensor<!tf_type.variant<tensor<10xf32>>>

// CHECK: func @tensorListIfCondFalse(%arg0: tensor<3x10xf32>) -> tensor<3x10xf32>
// CHECK-NOT: tf.TensorListSetItem
}
