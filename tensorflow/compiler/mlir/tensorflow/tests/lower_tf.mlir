// RUN: tf-opt %s -test-tf-lower-tf | FileCheck %s

// CHECK-LABEL: invert_permutation
func.func @invert_permutation(%arg0: tensor<5xi32>) -> tensor<5xi32> {
  // CHECK-DAG: %[[UPDATES:.*]] = "tf.Const"() {value = dense<[0, 1, 2, 3, 4]> : tensor<5xi32>} : () -> tensor<5xi32>
  // CHECK-DAG: %[[SHAPE:.*]] = "tf.Const"() {value = dense<[5, 1]> : tensor<2xi32>} : () -> tensor<2xi32>
  // CHECK-DAG: %[[cst_1:.*]] = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
  // CHECK-DAG: %[[cst_2:.*]] = "tf.Const"() {value = dense<1> : tensor<5xi32>} : () -> tensor<5xi32>
  // CHECK-DAG: %[[cst_3:.*]] = "tf.Const"() {value = dense<0> : tensor<5xi32>} : () -> tensor<5xi32>

  // CHECK-DAG: %[[INDICES:.*]] = "tf.Reshape"(%arg0, %[[SHAPE]]) : (tensor<5xi32>, tensor<2xi32>) -> tensor<5x1xi32>
  // CHECK-DAG: %[[INDICES_1:.*]] = "tf.TensorScatterAdd"(%[[cst_3]], %[[INDICES]], %[[cst_2]]) : (tensor<5xi32>, tensor<5x1xi32>, tensor<5xi32>) -> tensor<5xi32>
  // CHECK-DAG: %[[INDICES_2:.*]] = "tf.Sub"(%[[cst_1]], %[[INDICES_1]]) : (tensor<i32>, tensor<5xi32>) -> tensor<5xi32>
  // CHECK-DAG: %[[INDICES_3:.*]] = "tf.Mul"(%[[INDICES_2]], %arg0) : (tensor<5xi32>, tensor<5xi32>) -> tensor<5xi32>
  // CHECK-DAG: %[[INDICES_4:.*]] = "tf.TensorScatterAdd"(%[[cst_3]], %0, %[[UPDATES]]) : (tensor<5xi32>, tensor<5x1xi32>, tensor<5xi32>) -> tensor<5xi32>
  // CHECK-DAG: %[[INDICES_5:.*]] = "tf.AddV2"(%[[INDICES_3]], %[[INDICES_4]]) : (tensor<5xi32>, tensor<5xi32>) -> tensor<5xi32>
  %0 = "tf.InvertPermutation"(%arg0) : (tensor<5xi32>) -> tensor<5xi32>
  func.return %0 : tensor<5xi32>
}

// CHECK-LABEL: invert_permutation_dynamic
func.func @invert_permutation_dynamic(%arg0: tensor<?xi32>) -> tensor<?xi32> {
  // CHECK: tf.InvertPermutation
  %0 = "tf.InvertPermutation"(%arg0) : (tensor<?xi32>) -> tensor<?xi32>
  func.return %0 : tensor<?xi32>
}

// CHECK-LABEL: invert_permutation_unranked
func.func @invert_permutation_unranked(%arg0: tensor<*xi32>) -> tensor<*xi32> {
  // CHECK: tf.InvertPermutation
  %0 = "tf.InvertPermutation"(%arg0) : (tensor<*xi32>) -> tensor<*xi32>
  func.return %0 : tensor<*xi32>
}

// CHECK-LABEL: simple_pack
// CHECK-SAME: %[[ARG0:.*]]: tensor<3x5xf32>, %[[ARG1:.*]]: tensor<3x5xf32>
func.func @simple_pack(%arg0: tensor<3x5xf32>, %arg1: tensor<3x5xf32>) -> tensor<2x3x5xf32> {
  // CHECK: %[[AXIS:.*]] = "tf.Const"() {value = dense<0> : tensor<i64>}
  // CHECK: %[[INP0:.*]] = "tf.ExpandDims"(%[[ARG0]], %[[AXIS]]) : (tensor<3x5xf32>, tensor<i64>) -> tensor<1x3x5xf32>
  // CHECK: %[[INP1:.*]] = "tf.ExpandDims"(%[[ARG1]], %[[AXIS]]) : (tensor<3x5xf32>, tensor<i64>) -> tensor<1x3x5xf32>
  // CHECK: "tf.ConcatV2"(%[[INP0]], %[[INP1]], %[[AXIS]]) : (tensor<1x3x5xf32>, tensor<1x3x5xf32>, tensor<i64>) -> tensor<2x3x5xf32>

  %0 = "tf.Pack"(%arg0, %arg1) : (tensor<3x5xf32>, tensor<3x5xf32>) -> tensor<2x3x5xf32>
  func.return %0 : tensor<2x3x5xf32>
}

// CHECK-LABEL: func @square
func.func @square(%arg0: tensor<3xf32>) -> tensor<3xf32> {
  // CHECK: "tf.Mul"(%arg0, %arg0)
  %1 = "tf.Square"(%arg0) : (tensor<3xf32>) -> tensor<3xf32>
  func.return %1 : tensor<3xf32>
}

// CHECK-LABEL: func @squared_difference_real
func.func @squared_difference_real(%arg0: tensor<3xf32>, %arg1: tensor<3xf32>) -> tensor<3xf32> {
  // CHECK: [[R1:%.+]] = "tf.Sub"(%arg0, %arg1)
  // CHECK: "tf.Mul"([[R1]], [[R1]])
  %1 = "tf.SquaredDifference"(%arg0, %arg1) : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xf32>
  func.return %1 : tensor<3xf32>
}

// CHECK-LABEL: func @squared_difference_complex
func.func @squared_difference_complex(%arg0: tensor<3xcomplex<f32>>, %arg1: tensor<3xcomplex<f32>>) -> tensor<3xcomplex<f32>> {
  // CHECK-DAG: [[R1:%.+]] = "tf.Sub"(%arg0, %arg1)
  // CHECK-DAG: [[R2:%.+]] = "tf.Conj"([[R1]])
  // CHECK-DAG: "tf.Mul"([[R1]], [[R2]])
  %1 = "tf.SquaredDifference"(%arg0, %arg1) : (tensor<3xcomplex<f32>>, tensor<3xcomplex<f32>>) -> tensor<3xcomplex<f32>>
  func.return %1 : tensor<3xcomplex<f32>>
}

// CHECK-LABEL: func @div_no_nan
// CHECK-SAME: (%[[X:.*]]: tensor<*xf32>, %[[Y:.*]]: tensor<*xf32>)
func.func @div_no_nan(%arg0: tensor<*xf32>, %arg1: tensor<*xf32>) -> tensor<*xf32> {
  // CHECK:  %[[ZERO:.*]] = "tf.Const"() {value = dense<0.000000e+00> : tensor<f32>} : () -> tensor<f32>
  // CHECK:  %[[IS_ZERO:.*]] = "tf.Equal"(%[[Y]], %[[ZERO]]) {incompatible_shape_error = true} : (tensor<*xf32>, tensor<f32>) -> tensor<*xi1>
  // CHECK:  %[[DIV:.*]] = "tf.Div"(%[[X]], %[[Y]]) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
  // CHECK:  %[[RESULT:.*]] = "tf.SelectV2"(%[[IS_ZERO]], %[[ZERO]], %[[DIV]]) : (tensor<*xi1>, tensor<f32>, tensor<*xf32>) -> tensor<*xf32>
  %0 = "tf.DivNoNan"(%arg0, %arg1) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>

  // CHECK: return %[[RESULT]]
  func.return %0 : tensor<*xf32>
}

// CHECK-LABEL: @truncate_div_int
// CHECK-SAME: (%[[LHS:.*]]: tensor<*xi32>, %[[RHS:.*]]: tensor<*xi32>)
func.func @truncate_div_int(%arg0: tensor<*xi32>, %arg1: tensor<*xi32>)
    -> tensor<*xi32> {
  // CHECK: %[[RESULT:.*]] = "tf.Div"(%[[LHS]], %[[RHS]])
  // CHECK: return %[[RESULT]]
  %0 = "tf.TruncateDiv"(%arg0, %arg1)
      : (tensor<*xi32>, tensor<*xi32>) -> tensor<*xi32>
  func.return %0 : tensor<*xi32>
}

// CHECK-LABEL: @truncate_div_float
// CHECK-SAME: (%[[LHS:.*]]: tensor<*xf32>, %[[RHS:.*]]: tensor<*xf32>)
func.func @truncate_div_float(%arg0: tensor<*xf32>, %arg1: tensor<*xf32>)
    -> tensor<*xf32> {
  // CHECK:  %[[ZERO:.*]] = "tf.Const"() {value = dense<0.000000e+00> : tensor<f32>} : () -> tensor<f32>
  // CHECK:  %[[XDIVY:.*]] = "tf.Div"(%[[LHS]], %[[RHS]])
  // CHECK:  %[[MASK:.*]] = "tf.Less"(%[[XDIVY]], %[[ZERO]])
  // CHECK:  %[[CEIL:.*]] = "tf.Ceil"(%[[XDIVY]])
  // CHECK:  %[[FLOOR:.*]] = "tf.Floor"(%[[XDIVY]])
  // CHECK:  %[[RESULT:.*]] = "tf.SelectV2"(%[[MASK]], %[[CEIL]], %[[FLOOR]])
  %0 = "tf.TruncateDiv"(%arg0, %arg1)
      : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>

  // CHECK: return %[[RESULT]]
  func.return %0 : tensor<*xf32>
}

// CHECK-LABEL: func @mul_no_nan
// CHECK-SAME: (%[[X:.*]]: tensor<2x3xf32>, %[[Y:.*]]: tensor<3xf32>)
func.func @mul_no_nan(%arg0: tensor<2x3xf32>, %arg1: tensor<3xf32>) -> tensor<2x3xf32> {
  // CHECK:  %[[ZERO:.*]] = "tf.Const"() {value = dense<0.000000e+00> : tensor<f32>} : () -> tensor<f32>
  // CHECK:  %[[IS_ZERO:.*]] = "tf.Equal"(%[[Y]], %[[ZERO]]) {incompatible_shape_error = true} : (tensor<3xf32>, tensor<f32>) -> tensor<3xi1>
  // CHECK:  %[[MUL:.*]] = "tf.Mul"(%[[X]], %[[Y]]) : (tensor<2x3xf32>, tensor<3xf32>) -> tensor<2x3xf32>
  // CHECK:  %[[RESULT:.*]] = "tf.SelectV2"(%[[IS_ZERO]], %[[ZERO]], %[[MUL]]) : (tensor<3xi1>, tensor<f32>, tensor<2x3xf32>) -> tensor<2x3xf32>
  %0 = "tf.MulNoNan"(%arg0, %arg1) : (tensor<2x3xf32>, tensor<3xf32>) -> tensor<2x3xf32>

  // CHECK: return %[[RESULT]]
  func.return %0 : tensor<2x3xf32>
}

// CHECK-LABEL: @is_inf
func.func @is_inf(%arg0: tensor<3x4xf32>) -> tensor<3x4xi1> {
  // CHECK: %[[INF:.*]] = "tf.Const"() {value = dense<0x7F800000> : tensor<f32>} : () -> tensor<f32>
  // CHECK: %[[ABS:.*]] = "tf.Abs"(%arg0) : (tensor<3x4xf32>) -> tensor<3x4xf32>
  // CHECK: %[[RESULT:.*]] = "tf.Equal"(%[[ABS]], %[[INF]]) {incompatible_shape_error = true} : (tensor<3x4xf32>, tensor<f32>) -> tensor<3x4xi1>
  %0 = "tf.IsInf"(%arg0) : (tensor<3x4xf32>) -> tensor<3x4xi1>
  // CHECK: return %[[RESULT]]
  func.return %0 : tensor<3x4xi1>
}

// CHECK-LABEL: @is_nan
func.func @is_nan(%arg0: tensor<3x4xf32>) -> tensor<3x4xi1> {
  // CHECK: %[[RESULT:.*]] = "tf.NotEqual"(%arg0, %arg0) {incompatible_shape_error = true} : (tensor<3x4xf32>, tensor<3x4xf32>) -> tensor<3x4xi1>
  %0 = "tf.IsNan"(%arg0) : (tensor<3x4xf32>) -> tensor<3x4xi1>
  // CHECK: return %[[RESULT]]
  func.return %0 : tensor<3x4xi1>
}

// CHECK-LABEL: func @fill
// CHECK-SAME: (%[[ARG0:.*]]: tensor<*xi64>, %[[ARG1:.*]]: tensor<*xf32>)
func.func @fill(%arg0: tensor<*xi64>, %arg1: tensor<*xf32>) -> tensor<*xf32> {
  // CHECK: "tf.BroadcastTo"(%[[ARG1]], %[[ARG0]])
  %0 = "tf.Fill"(%arg0, %arg1) : (tensor<*xi64>, tensor<*xf32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

func.func @empty(%arg0: tensor<?xi32>) -> tensor<*xf32> {
  // CHECK-DAG: [[CST:%.+]] = "tf.Const"() {value = dense<0.000000e+00> : tensor<f32>}
  // CHECK-DAG: [[RES:%.+]] = "tf.BroadcastTo"([[CST]], %arg0)
  %0 = "tf.Empty"(%arg0) {init = true} : (tensor<?xi32>) -> (tensor<*xf32>)

  // CHECK: return [[RES]]
  func.return %0 : tensor<*xf32>
}

// CHECK-LABEL: func @l2_loss
// CHECK-SAME: (%[[INPUT:.*]]: tensor<?x?xf32>)
func.func @l2_loss(%arg0: tensor<?x?xf32>) -> tensor<f32> {

  // CHECK-DAG: %[[SQUARE:.*]] = "tf.Mul"(%[[INPUT]], %[[INPUT]]) : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  // CHECK-DAG: %[[REDUCE_AXES:.*]] = "tf.Const"() {value = dense<[0, 1]> : tensor<2xi64>}
  // CHECK-DAG: %[[SUM:.*]] = "tf.Sum"(%[[SQUARE]], %[[REDUCE_AXES]]) {keep_dims = false} : (tensor<?x?xf32>, tensor<2xi64>) -> tensor<f32>
  // CHECK-DAG: %[[TWO:.*]] = "tf.Const"() {value = dense<2.000000e+00> : tensor<f32>}
  // CHECK-DAG: %[[LOSS:.*]] = "tf.Div"(%[[SUM]], %[[TWO]]) : (tensor<f32>, tensor<f32>) -> tensor<f32>

  %0 = "tf.L2Loss"(%arg0) : (tensor<?x?xf32>) -> tensor<f32>

  // CHECK: return %[[LOSS]] : tensor<f32>
  func.return %0 : tensor<f32>
}

// CHECK-LABEL: func @l2_loss_unranked
func.func @l2_loss_unranked(%arg0: tensor<*xf32>) -> tensor<f32> {
  // CHECK: tf.L2Loss
  %0 = "tf.L2Loss"(%arg0) : (tensor<*xf32>) -> tensor<f32>
  func.return %0 : tensor<f32>
}

// CHECK-LABEL: pack_with_unranked
// CHECK-SAME: %[[ARG0:.*]]: tensor<?x5xf32>, %[[ARG1:.*]]: tensor<*xf32>
func.func @pack_with_unranked(%arg0: tensor<?x5xf32>, %arg1: tensor<*xf32>) -> tensor<*xf32> {
  // CHECK: %[[AXIS:.*]] = "tf.Const"() {value = dense<-2> : tensor<i64>}
  // CHECK: %[[INP0:.*]] = "tf.ExpandDims"(%[[ARG0]], %[[AXIS]]) : (tensor<?x5xf32>, tensor<i64>) -> tensor<?x1x5xf32>
  // CHECK: %[[INP1:.*]] = "tf.ExpandDims"(%[[ARG1]], %[[AXIS]]) : (tensor<*xf32>, tensor<i64>) -> tensor<*xf32>
  // CHECK: "tf.ConcatV2"(%[[INP0]], %[[INP1]], %[[AXIS]]) : (tensor<?x1x5xf32>, tensor<*xf32>, tensor<i64>) -> tensor<*xf32>

  %0 = "tf.Pack"(%arg0, %arg1) {axis = -2 : i64} : (tensor<?x5xf32>, tensor<*xf32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// CHECK-LABEL: func @pad
func.func @pad(%arg0: tensor<3xf32>) -> tensor<6xf32> {
  %padding = "tf.Const"() { value = dense<[[1, 2]]> : tensor<1x2xi64> } : () -> tensor<1x2xi64>
  // CHECK-DAG: [[PAD:%.+]] = "tf.Const"() {{.+}} -> tensor<1x2xi64>
  // CHECK-DAG: [[CST:%.+]] = "tf.Const"() {value = dense<0.000000e+00> : tensor<f32>}
  // CHECK: "tf.PadV2"(%arg0, [[PAD]], [[CST]])
  %0 = "tf.Pad"(%arg0, %padding) : (tensor<3xf32>, tensor<1x2xi64>) -> tensor<6xf32>
  func.return %0 : tensor<6xf32>
}

// CHECK-LABEL: func @pad_bf16
func.func @pad_bf16(%arg0: tensor<3xbf16>) -> tensor<6xbf16> {
  %padding = "tf.Const"() { value = dense<[[1, 2]]> : tensor<1x2xi64> } : () -> tensor<1x2xi64>
  // CHECK-DAG: [[PAD:%.+]] = "tf.Const"() {{.+}}  -> tensor<1x2xi64>
  // CHECK-DAG: [[CST:%.+]] = "tf.Const"() {value = dense<0.000000e+00> : tensor<bf16>}
  // CHECK: "tf.PadV2"(%arg0, [[PAD]], [[CST]])
  %0 = "tf.Pad"(%arg0, %padding) : (tensor<3xbf16>, tensor<1x2xi64>) -> tensor<6xbf16>
  func.return %0 : tensor<6xbf16>
}

// CHECK-LABEL: func @add_f32
func.func @add_f32(%arg0: tensor<3xf32>, %arg1: tensor<3xf32>) -> tensor<3xf32> {
  // CHECK-DAG: [[ADD:%.+]] = "tf.AddV2"(%arg0, %arg1) : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xf32>
  %add = "tf.Add"(%arg0, %arg1) : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xf32>
  func.return %add : tensor<3xf32>
}

// CHECK-LABEL: func @BiasAddGrad_NHWC
func.func @BiasAddGrad_NHWC(%arg0: tensor<2x3x4x5xf32>) -> tensor<5xf32> {
  // CHECK: "tf.Const"() {value = dense<[0, 1, 2]> : tensor<3xi64>}
  // CHECK: "tf.Sum"({{.*}}) {keep_dims = false}

  %0 = "tf.BiasAddGrad"(%arg0) {data_format = "NHWC"} : (tensor<2x3x4x5xf32>) -> tensor<5xf32>
  func.return %0 : tensor<5xf32>
}

// CHECK-LABEL: func @BiasAddGrad_NCHW
func.func @BiasAddGrad_NCHW(%arg0: tensor<2x3x4x5xf32>) -> tensor<3xf32> {
  // CHECK: "tf.Const"() {value = dense<[0, 2, 3]> : tensor<3xi64>}
  // CHECK: "tf.Sum"({{.*}}) {keep_dims = false}

  %0 = "tf.BiasAddGrad"(%arg0) {data_format = "NCHW"} : (tensor<2x3x4x5xf32>) -> tensor<3xf32>
  func.return %0 : tensor<3xf32>
}

// CHECK-LABEL: func @BiasAddGrad_dynamic
func.func @BiasAddGrad_dynamic(%arg0: tensor<?x?x?x?xf32>) -> tensor<?xf32> {
  // CHECK: tf.Sum
  %0 = "tf.BiasAddGrad"(%arg0) {data_format = "NCHW"} : (tensor<?x?x?x?xf32>) -> tensor<?xf32>
  func.return %0 : tensor<?xf32>
}

// CHECK-LABEL: func @BiasAddGrad_unranked
func.func @BiasAddGrad_unranked(%arg0: tensor<*xf32>) -> tensor<?xf32> {
  // CHECK: tf.BiasAddGrad
  %0 = "tf.BiasAddGrad"(%arg0) {data_format = "NCHW"} : (tensor<*xf32>) -> tensor<?xf32>
  func.return %0 : tensor<?xf32>
}

// CHECK-LABEL: func @rsqrt_grad
// CHECK-SAME: (%[[ARG0:.*]]: tensor<2xf32>, %[[ARG1:.*]]: tensor<2xf32>)
func.func @rsqrt_grad(%arg0: tensor<2xf32>, %arg1: tensor<2xf32>) -> tensor<2xf32> {
  // CHECK: %[[CST:.*]] = "tf.Const"() {value = dense<-2.000000e+00> : tensor<f32>}
  // CHECK: %[[LHS2:.*]] = "tf.Mul"(%[[ARG0]], %[[ARG0]])
  // CHECK: %[[LHS3:.*]] = "tf.Mul"(%[[LHS2]], %[[ARG0]])
  // CHECK: %[[DIV:.*]] = "tf.Div"(%[[ARG1]], %[[CST]])
  // CHECK: %[[RET:.*]] = "tf.Mul"(%[[LHS3]], %[[DIV]])

  %0 = "tf.RsqrtGrad"(%arg0, %arg1) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
  // CHECK: return %[[RET]]
  func.return %0 : tensor<2xf32>
}

// CHECK-LABEL: func @rsqrt_grad_unranked
func.func @rsqrt_grad_unranked(%arg0: tensor<*xf32>, %arg1: tensor<*xf32>) -> tensor<*xf32> {
  // CHECK: tf.Const
  // CHECK: tf.Mul
  // CHECK: tf.Mul
  // CHECK: tf.Div
  // CHECK: tf.Mul
  %0 = "tf.RsqrtGrad"(%arg0, %arg1) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// CHECK-LABEL: func @sqrt_grad_unranked
// CHECK-SAME: (%[[ARG0:.*]]: tensor<*xcomplex<f32>>, %[[ARG1:.*]]: tensor<*xcomplex<f32>>)
func.func @sqrt_grad_unranked(%arg0: tensor<*xcomplex<f32>>, %arg1: tensor<*xcomplex<f32>>) -> tensor<*xcomplex<f32>> {
  // CHECK: %[[CST:.*]] = "tf.Const"() {value = dense<(5.000000e-01,0.000000e+00)> : tensor<complex<f32>>} : () -> tensor<complex<f32>>
  // CHECK: %[[MUL:.*]] = "tf.Mul"(%arg1, %[[CST]]) : (tensor<*xcomplex<f32>>, tensor<complex<f32>>) -> tensor<*xcomplex<f32>>
  // CHECK: %[[RET:.*]] = "tf.Div"(%[[MUL]], %arg0) : (tensor<*xcomplex<f32>>, tensor<*xcomplex<f32>>) -> tensor<*xcomplex<f32>>

  %0 = "tf.SqrtGrad"(%arg0, %arg1) : (tensor<*xcomplex<f32>>, tensor<*xcomplex<f32>>) -> tensor<*xcomplex<f32>>
  // CHECK: return %[[RET]]
  func.return %0 : tensor<*xcomplex<f32>>
}

// %input has 1 batch dimension then 2 block dimensions then 1 remainder
// dimension.
// CHECK-LABEL: fourdim_space_to_batch_nd
func.func @fourdim_space_to_batch_nd(%input: tensor<3x5x7x10xf32>, %block_shape: tensor<2xi64>, %paddings: tensor<2x2xi64>) -> tensor<?x?x?x10xf32> {
  // CHECK-DAG: [[PAD00:%.+]] = "tf.Const"() {value = dense<0> : tensor<1x2xi64>}
  // CHECK-DAG: [[ZERO_I32:%.+]] = "tf.Const"() {value = dense<0> : tensor<i32>}
  // CHECK-DAG: [[ZERO_I64:%.+]] = "tf.Const"() {value = dense<0> : tensor<i64>}
  // CHECK-DAG: [[FULL_PADDINGS:%.+]] = "tf.ConcatV2"([[PAD00]], %arg2, [[PAD00]], [[ZERO_I64]])
  // CHECK-DAG: [[PAD_DEFAULT:%.+]] = "tf.Const"() {value = dense<0.000000e+00> : tensor<f32>}
  // CHECK-DAG: [[PADDED:%.+]] = "tf.PadV2"(%arg0, [[FULL_PADDINGS]], [[PAD_DEFAULT]])
  // CHECK-DAG: [[PADDINGS:%.+]]:2 = "tf.Unpack"([[FULL_PADDINGS]]) {axis = 1 : i64}
  // CHECK-DAG: [[PADDINGS_SUM:%.+]] = "tf.AddV2"([[PADDINGS]]#0, [[PADDINGS]]#1)
  // CHECK-DAG: [[INPUT_SHAPE:%.+]] = "tf.Const"() {value = dense<[3, 5, 7, 10]> : tensor<4xi64>}
  // CHECK-DAG: [[PADDED_SHAPE:%.+]] = "tf.AddV2"([[PADDINGS_SUM]], [[INPUT_SHAPE]])
  // CHECK-DAG: [[PADDED_SHAPE_SPLITS:%.+]]:4 = "tf.Split"([[ZERO_I32]], [[PADDED_SHAPE]])
  // CHECK-DAG: [[BLOCK_SHAPE_SPLITS:%.+]]:2 = "tf.Split"([[ZERO_I32]], %arg1)
  // CHECK-DAG: [[OUTER_SHAPE_0:%.+]] = "tf.Div"([[PADDED_SHAPE_SPLITS]]#1, [[BLOCK_SHAPE_SPLITS]]#0)
  // CHECK-DAG: [[OUTER_SHAPE_1:%.+]] = "tf.Div"([[PADDED_SHAPE_SPLITS]]#2, [[BLOCK_SHAPE_SPLITS]]#1)
  // CHECK-DAG: [[RESHAPED_SHAPE:%.+]] = "tf.ConcatV2"([[PADDED_SHAPE_SPLITS]]#0, [[OUTER_SHAPE_0]], [[BLOCK_SHAPE_SPLITS]]#0, [[OUTER_SHAPE_1]], [[BLOCK_SHAPE_SPLITS]]#1, [[PADDED_SHAPE_SPLITS]]#3, [[ZERO_I64]])
  // CHECK-DAG: [[PERMUTATION:%.+]] = "tf.Const"() {value = dense<[2, 4, 0, 1, 3, 5]> : tensor<6xi64>}
  // CHECK-DAG: [[OUTPUT_BATCH_PART:%.+]] = "tf.Mul"([[PADDED_SHAPE_SPLITS]]#0, [[BLOCK_SHAPE_SPLITS]]#0)
  // CHECK-DAG: [[OUTPUT_BATCH:%.+]] = "tf.Mul"([[OUTPUT_BATCH_PART]], [[BLOCK_SHAPE_SPLITS]]#1)
  // CHECK-DAG: [[OUTPUT_SHAPE:%.+]] = "tf.ConcatV2"([[OUTPUT_BATCH]], [[OUTER_SHAPE_0]], [[OUTER_SHAPE_1]], [[PADDED_SHAPE_SPLITS]]#3, [[ZERO_I64]])
  // CHECK-DAG: [[RESHAPED:%.+]] = "tf.Reshape"([[PADDED]], [[RESHAPED_SHAPE]])
  // CHECK-DAG: [[PERMUTED:%.+]] = "tf.Transpose"([[RESHAPED]], [[PERMUTATION]])
  // CHECK-DAG: [[RESULT:%.+]] = "tf.Reshape"([[PERMUTED]], [[OUTPUT_SHAPE]])
  // CHECK-DAG: return [[RESULT]]
  %0 = "tf.SpaceToBatchND"(%input, %block_shape, %paddings) : (tensor<3x5x7x10xf32>, tensor<2xi64>, tensor<2x2xi64>) -> tensor<?x?x?x10xf32>
  func.return %0 : tensor<?x?x?x10xf32>
}

// Verify SpaceToBatchND with input tensor of element type f16. This test case is derived from 'fourdim_space_to_batch_nd'. It checks the output
// tensor shape and element type in a few lines in the resulting lowering.
// CHECK-LABEL: space_to_batch_nd_element_type_f16
func.func @space_to_batch_nd_element_type_f16(%input: tensor<3x5x7x10xf16>, %block_shape: tensor<2xi64>, %paddings: tensor<2x2xi64>) -> tensor<?x?x?x10xf16> {
  // CHECK-DAG: "tf.PadV2"(%arg0, {{.*}}, {{.*}}) {{.*}} -> tensor<3x?x?x10xf16>
  // CHECK-DAG: return {{.*}}: tensor<?x?x?x10xf16>
  %0 = "tf.SpaceToBatchND"(%input, %block_shape, %paddings) : (tensor<3x5x7x10xf16>, tensor<2xi64>, tensor<2x2xi64>) -> tensor<?x?x?x10xf16>
  func.return %0 : tensor<?x?x?x10xf16>
}

// Verify the result shape for the tf.PadV2 op.
// CHECK-LABEL: const_paddings_space_to_batch_nd
func.func @const_paddings_space_to_batch_nd(%arg0: tensor<1x8x2xf32>) -> (tensor<3x5x2xf32>) {
  %0 = "tf.Const"() {value = dense<3> : tensor<1xi32>} : () -> tensor<1xi32>
  %1 = "tf.Const"() {value = dense<[[3, 4]]> : tensor<1x2xi32>} : () -> tensor<1x2xi32>


  // CHECK-DAG: [[VAL0:%.+]] = "tf.Const"() {value = dense<[3, 5, 2]> : tensor<3xi64>}
  // CHECK-DAG: [[VAL1:%.+]] = "tf.Const"() {value = dense<[1, 5, 3, 2]> : tensor<4xi64>}
  // CHECK-DAG: [[VAL2:%.+]] = "tf.Const"() {value = dense<{{\[\[}}0, 0], [3, 4], [0, 0{{\]\]}}> : tensor<3x2xi64>}
  // CHECK-DAG: [[VAL3:%.+]] = "tf.Const"() {value = dense<0.000000e+00> : tensor<f32>}
  // CHECK-DAG: [[VAL4:%.+]] = "tf.Const"() {value = dense<[2, 0, 1, 3]> : tensor<4xi64>}
  // CHECK-DAG: [[VAL5:%.+]] = "tf.PadV2"(%arg0, [[VAL2]], [[VAL3]])
  // CHECK-SAME: tensor<1x15x2xf32>
  // CHECK-DAG: [[VAL6:%.+]] = "tf.Reshape"([[VAL5]], [[VAL1]])
  // CHECK-DAG: [[VAL7:%.+]] = "tf.Transpose"([[VAL6]], [[VAL4]])
  // CHECK-DAG: [[VAL8:%.+]] = "tf.Reshape"([[VAL7]], [[VAL0]])
  %2 = "tf.SpaceToBatchND"(%arg0, %0, %1) : (tensor<1x8x2xf32>, tensor<1xi32>, tensor<1x2xi32>) -> tensor<3x5x2xf32>

  // CHECK: return [[VAL8]]
  func.return %2 : tensor<3x5x2xf32>
}

// CHECK-LABEL: avoid_lowering_space_to_batch_nd
func.func @avoid_lowering_space_to_batch_nd(%arg0: tensor<1x8x2xf32>, %arg1: tensor<*xi32>) -> (tensor<3x5x2xf32>) {
  %0 = "tf.Const"() {value = dense<3> : tensor<1xi32>} : () -> tensor<1xi32>
  %1 = "tf.SpaceToBatchND"(%arg0, %0, %arg1) : (tensor<1x8x2xf32>, tensor<1xi32>, tensor<*xi32>) -> tensor<3x5x2xf32>
  func.return %1 : tensor<3x5x2xf32>

  // CHECK: "tf.SpaceToBatchND"
}

// %input has 1 batch dimension then 3 block dimensions then 2 remainder
// dimensions. This checks only ops that are specific to the case with 3 block
// dimension and 2 remainder dimensions.
// CHECK-LABEL: sixdim_space_to_batch_nd
func.func @sixdim_space_to_batch_nd(%input: tensor<3x5x7x9x10x11xf32>, %block_shape: tensor<3xi64>, %paddings: tensor<3x2xi64>) -> tensor<?x?x?x?x10x11xf32> {
  // CHECK-DAG: [[PAD00:%.+]] = "tf.Const"()
  // CHECK-DAG: [[FULL_PADDINGS:%.+]] = "tf.ConcatV2"([[PAD00]], %arg2, [[PAD00]], [[PAD00]], {{.+}})
  // CHECK-DAG: [[INPUT_SHAPE:%.+]] = "tf.Const"() {value = dense<[3, 5, 7, 9, 10, 11]> : tensor<6xi64>}
  // CHECK-DAG: [[PADDED_SHAPE_SPLITS:%.+]]:6 = "tf.Split"
  // CHECK-DAG: [[BLOCK_SHAPE_SPLITS:%.+]]:3 = "tf.Split"
  // CHECK-DAG: [[OUTER_SHAPE_0:%.+]] = "tf.Div"([[PADDED_SHAPE_SPLITS]]#1, [[BLOCK_SHAPE_SPLITS]]#0)
  // CHECK-DAG: [[OUTER_SHAPE_1:%.+]] = "tf.Div"([[PADDED_SHAPE_SPLITS]]#2, [[BLOCK_SHAPE_SPLITS]]#1)
  // CHECK-DAG: [[OUTER_SHAPE_2:%.+]] = "tf.Div"([[PADDED_SHAPE_SPLITS]]#3, [[BLOCK_SHAPE_SPLITS]]#2)
  // CHECK-DAG: [[RESHAPED_SHAPE:%.+]] = "tf.ConcatV2"([[PADDED_SHAPE_SPLITS]]#0, [[OUTER_SHAPE_0]], [[BLOCK_SHAPE_SPLITS]]#0, [[OUTER_SHAPE_1]], [[BLOCK_SHAPE_SPLITS]]#1, [[OUTER_SHAPE_2]], [[BLOCK_SHAPE_SPLITS]]#2, [[PADDED_SHAPE_SPLITS]]#4, [[PADDED_SHAPE_SPLITS]]#5, {{.+}})
  // CHECK-DAG: [[PERMUTATION:%.+]] = "tf.Const"() {value = dense<[2, 4, 6, 0, 1, 3, 5, 7, 8]> : tensor<9xi64>}
  // CHECK-DAG: [[OUTPUT_BATCH_PART1:%.+]] = "tf.Mul"([[PADDED_SHAPE_SPLITS]]#0, [[BLOCK_SHAPE_SPLITS]]#0)
  // CHECK-DAG: [[OUTPUT_BATCH_PART2:%.+]] = "tf.Mul"([[OUTPUT_BATCH_PART1]], [[BLOCK_SHAPE_SPLITS]]#1)
  // CHECK-DAG: [[OUTPUT_BATCH:%.+]] = "tf.Mul"([[OUTPUT_BATCH_PART2]], [[BLOCK_SHAPE_SPLITS]]#2)
  // CHECK-DAG: [[OUTPUT_SHAPE:%.+]] = "tf.ConcatV2"([[OUTPUT_BATCH]], [[OUTER_SHAPE_0]], [[OUTER_SHAPE_1]], [[OUTER_SHAPE_2]], [[PADDED_SHAPE_SPLITS]]#4, [[PADDED_SHAPE_SPLITS]]#5, {{.+}})
  %0 = "tf.SpaceToBatchND"(%input, %block_shape, %paddings) : (tensor<3x5x7x9x10x11xf32>, tensor<3xi64>, tensor<3x2xi64>) -> tensor<?x?x?x?x10x11xf32>
  func.return %0 : tensor<?x?x?x?x10x11xf32>
}

// CHECK-LABEL: func @batchToSpace
func.func @batchToSpace(%arg0: tensor<3x5x2xf32>) -> (tensor<1x8x2xf32>) {
  // CHECK-DAG: [[VAL0:%.+]] = "tf.Const"() {value = dense<[3, 1, 5, 2]> : tensor<4xi64>}
  // CHECK-DAG: [[VAL1:%.+]] = "tf.Const"() {value = dense<[1, 2, 0, 3]> : tensor<4xi64>}
  // CHECK-DAG: [[VAL2:%.+]] = "tf.Const"() {value = dense<[1, 15, 2]> : tensor<3xi64>}
  // CHECK-DAG: [[VAL3:%.+]] = "tf.Const"() {value = dense<[0, 3, 0]> : tensor<3xi64>}
  // CHECK-DAG: [[VAL4:%.+]] = "tf.Const"() {value = dense<[1, 8, 2]> : tensor<3xi64>}
  // CHECK-DAG: [[VAL5:%.+]] = "tf.Reshape"(%arg0, [[VAL0]])
  // CHECK-DAG: [[VAL6:%.+]] = "tf.Transpose"([[VAL5]], [[VAL1]])
  // CHECK-DAG: [[VAL7:%.+]] = "tf.Reshape"([[VAL6]], [[VAL2]])
  // CHECK-DAG: [[VAL8:%.+]] = "tf.Slice"([[VAL7]], [[VAL3]], [[VAL4]])
  %0 = "tf.Const"() {value = dense<3> : tensor<1xi32>} : () -> tensor<1xi32>
  %1 = "tf.Const"() {value = dense<[[3, 4]]> : tensor<1x2xi32>} : () -> tensor<1x2xi32>
  %2 = "tf.BatchToSpaceND"(%arg0, %0, %1) {device = ""} : (tensor<3x5x2xf32>, tensor<1xi32>, tensor<1x2xi32>) -> tensor<1x8x2xf32>

  // CHECK: return [[VAL8]] : tensor<1x8x2xf32>
  func.return %2 : tensor<1x8x2xf32>
}

func.func @fake_quant_with_min_max_args(%arg0 : tensor<?x?xf32>) -> tensor<?x?xf32> {
  // CHECK-DAG: [[VAL0:%.+]] = "tf.Const"() {value = dense<1.275000e+02> : tensor<f32>}
  // CHECK-DAG: [[VAL1:%.+]] = "tf.Const"() {value = dense<1.00392163> : tensor<f32>}
  // CHECK-DAG: [[VAL2:%.+]] = "tf.Const"() {value = dense<-0.996078491> : tensor<f32>}
  // CHECK-DAG: [[VAL3:%.+]] = "tf.Const"() {value = dense<0.00784313772> : tensor<f32>}
  // CHECK-DAG: [[VAL4:%.+]] = "tf.Const"() {value = dense<5.000000e-01> : tensor<f32>}
  // CHECK-DAG: [[VAL5:%.+]] = "tf.ClipByValue"(%arg0, [[VAL2]], [[VAL1]])
  // CHECK-DAG: [[VAL6:%.+]] = "tf.Sub"([[VAL5]], [[VAL2]])
  // CHECK-DAG: [[VAL7:%.+]] = "tf.Mul"([[VAL6]], [[VAL0]])
  // CHECK-DAG: [[VAL8:%.+]] = "tf.AddV2"([[VAL7]], [[VAL4]])
  // CHECK-DAG: [[VAL9:%.+]] = "tf.Floor"([[VAL8]])
  // CHECK-DAG: [[VAL10:%.+]] = "tf.Mul"([[VAL9]], [[VAL3]])
  // CHECK-DAG: [[VAL11:%.+]] = "tf.AddV2"([[VAL10]], [[VAL2]])
  %0 = "tf.FakeQuantWithMinMaxArgs"(%arg0) {max = 1.0 : f32, min = -1.0 : f32, narrow_range = false, num_bits = 8 : i64} : (tensor<?x?xf32>) -> tensor<?x?xf32>

  // CHECK: return [[VAL11]]
  func.return %0 : tensor<?x?xf32>
}

func.func @fake_quant_with_min_max_vars(%arg0 : tensor<?x?xf32>, %arg1 : tensor<f32>, %arg2 : tensor<f32>) -> tensor<?x?xf32> {
  // CHECK-DAG: %[[ZERO:.*]] = "tf.Const"() {value = dense<0.000000e+00> : tensor<f32>} : () -> tensor<f32>
  // CHECK-DAG: %[[VAL1:.*]] = "tf.Const"() {value = dense<2.550000e+02> : tensor<f32>} : () -> tensor<f32>
  // CHECK-DAG: %[[VAL2:.*]] = "tf.Const"() {value = dense<2.000000e+00> : tensor<f32>} : () -> tensor<f32>
  // CHECK-DAG: %[[VAL3:.*]] = "tf.Const"() {value = dense<1.000000e+00> : tensor<f32>} : () -> tensor<f32>
  // CHECK-DAG: %[[VAL4:.*]] = "tf.Const"() {value = dense<5.000000e-01> : tensor<f32>} : () -> tensor<f32>
  // CHECK-DAG: %[[VAL5:.*]] = "tf.Sub"(%arg2, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  // CHECK-DAG: %[[VAL6:.*]] = "tf.Div"(%[[VAL5]], %[[VAL1]]) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  // CHECK-DAG: %[[VAL7:.*]] = "tf.Div"(%[[VAL1]], %[[VAL5]]) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  // CHECK-DAG: %[[VAL8:.*]] = "tf.Div"(%arg1, %[[VAL6]]) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  // CHECK-DAG: %[[VAL9:.*]] = "tf.Sub"(%[[ZERO]], %[[VAL8]]) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  // CHECK-DAG: %[[VAL10:.*]] = "tf.Floor"(%[[VAL9]]) : (tensor<f32>) -> tensor<f32>
  // CHECK-DAG: %[[VAL11:.*]] = "tf.Sub"(%[[VAL9]], %[[VAL10]]) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  // CHECK-DAG: %[[VAL12:.*]] = "tf.Greater"(%[[VAL11]], %[[VAL4]]) : (tensor<f32>, tensor<f32>) -> tensor<i1>
  // CHECK-DAG: %[[VAL13:.*]] = "tf.Equal"(%[[VAL11]], %[[VAL4]]) {incompatible_shape_error = true} : (tensor<f32>, tensor<f32>) -> tensor<i1>
  // CHECK-DAG: %[[VAL14:.*]] = "tf.Mul"(%[[VAL9]], %[[VAL4]]) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  // CHECK-DAG: %[[VAL15:.*]] = "tf.Floor"(%[[VAL14]]) : (tensor<f32>) -> tensor<f32>
  // CHECK-DAG: %[[VAL16:.*]] = "tf.Mul"(%[[VAL15]], %[[VAL2]]) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  // CHECK-DAG: %[[VAL17:.*]] = "tf.Sub"(%[[VAL10]], %[[VAL16]]) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  // CHECK-DAG: %[[VAL18:.*]] = "tf.Equal"(%[[VAL17]], %[[VAL3]]) {incompatible_shape_error = true} : (tensor<f32>, tensor<f32>) -> tensor<i1>
  // CHECK-DAG: %[[VAL19:.*]] = "tf.LogicalAnd"(%[[VAL13]], %[[VAL18]]) : (tensor<i1>, tensor<i1>) -> tensor<i1>
  // CHECK-DAG: %[[VAL20:.*]] = "tf.LogicalOr"(%[[VAL12]], %[[VAL19]]) : (tensor<i1>, tensor<i1>) -> tensor<i1>
  // CHECK-DAG: %[[VAL21:.*]] = "tf.AddV2"(%[[VAL10]], %[[VAL3]]) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  // CHECK-DAG: %[[INNER_SELECT:.*]] = "tf.SelectV2"(%[[VAL20]], %[[VAL21]], %[[VAL10]]) : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
  // CHECK-DAG: %[[IS_ZERO:.*]] = "tf.Equal"(%[[INNER_SELECT]], %[[ZERO]]) {incompatible_shape_error = true}
  // CHECK-DAG: %[[VAL22:.*]] = "tf.SelectV2"(%[[IS_ZERO]], %[[ZERO]], %[[INNER_SELECT]])
  // CHECK-DAG: %[[VAL23:.*]] = "tf.ClipByValue"(%[[VAL22]], %[[ZERO]], %[[VAL1]]) : (tensor<f32>, tensor<f32>, tensor<f32>) -> tensor<f32>
  // CHECK-DAG: %[[VAL24:.*]] = "tf.Sub"(%[[ZERO]], %[[VAL23]]) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  // CHECK-DAG: %[[VAL25:.*]] = "tf.Sub"(%[[VAL1]], %[[VAL23]]) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  // CHECK-DAG: %[[VAL26:.*]] = "tf.Mul"(%[[VAL24]], %[[VAL6]]) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  // CHECK-DAG: %[[VAL27:.*]] = "tf.Mul"(%[[VAL25]], %[[VAL6]]) : (tensor<f32>, tensor<f32>) -> tensor<f32>
  // CHECK-DAG: %[[VAL28:.*]] = "tf.ClipByValue"(%arg0, %[[VAL26]], %[[VAL27]]) : (tensor<?x?xf32>, tensor<f32>, tensor<f32>) -> tensor<?x?xf32>
  // CHECK-DAG: %[[VAL29:.*]] = "tf.Sub"(%[[VAL28]], %[[VAL26]]) : (tensor<?x?xf32>, tensor<f32>) -> tensor<?x?xf32>
  // CHECK-DAG: %[[VAL30:.*]] = "tf.Mul"(%[[VAL29]], %[[VAL7]]) : (tensor<?x?xf32>, tensor<f32>) -> tensor<?x?xf32>
  // CHECK-DAG: %[[VAL31:.*]] = "tf.AddV2"(%[[VAL30]], %[[VAL4]]) : (tensor<?x?xf32>, tensor<f32>) -> tensor<?x?xf32>
  // CHECK-DAG: %[[VAL32:.*]] = "tf.Floor"(%[[VAL31]]) : (tensor<?x?xf32>) -> tensor<?x?xf32>
  // CHECK-DAG: %[[VAL33:.*]] = "tf.Mul"(%[[VAL32]], %[[VAL6]]) : (tensor<?x?xf32>, tensor<f32>) -> tensor<?x?xf32>
  // CHECK-DAG: %[[VAL34:.*]] = "tf.AddV2"(%[[VAL33]], %[[VAL26]]) : (tensor<?x?xf32>, tensor<f32>) -> tensor<?x?xf32>
  %0 = "tf.FakeQuantWithMinMaxVars"(%arg0, %arg1, %arg2) {narrow_range = false, num_bits = 8 : i64} : (tensor<?x?xf32>, tensor<f32>, tensor<f32>) -> tensor<?x?xf32>

  // CHECK: return %[[VAL34]]
  func.return %0 : tensor<?x?xf32>
}

// CHECK-LABEL: SoftmaxCrossEntropyWithLogits
// CHECK-SAME: %[[FEATURES:.*]]: tensor<2x3xf32>, %[[LABELS:.*]]: tensor<2x3xf32>
func.func @SoftmaxCrossEntropyWithLogits(%features: tensor<2x3xf32>, %labels: tensor<2x3xf32>) -> (tensor<2xf32>, tensor<2x3xf32>) {
  // CHECK-DAG: %[[AXIS:.*]] = "tf.Const"() {value = dense<-1> : tensor<1xi64>} : () -> tensor<1xi64>
  // CHECK-DAG: %[[ZERO:.*]] = "tf.Const"() {value = dense<0.000000e+00> : tensor<f32>} : () -> tensor<f32>
  // CHECK-DAG: %[[NEG_LABELS:.*]] = "tf.Neg"(%[[LABELS]]) : (tensor<2x3xf32>) -> tensor<2x3xf32>

  // LogSoftmax expansion.
  // CHECK-DAG: %[[LOG_SOFTMAX_MAX:.*]] = "tf.Max"(%[[FEATURES]], %[[AXIS]]) {keep_dims = true} : (tensor<2x3xf32>, tensor<1xi64>) -> tensor<2x1xf32>
  // CHECK-DAG: %[[LOG_SOFTMAX_SHIFTED:.*]] = "tf.Sub"(%[[FEATURES]], %[[LOG_SOFTMAX_MAX]]) : (tensor<2x3xf32>, tensor<2x1xf32>) -> tensor<2x3xf32>
  // CHECK-DAG: %[[LOG_SOFTMAX_EXP:.*]] = "tf.Exp"(%[[LOG_SOFTMAX_SHIFTED]]) : (tensor<2x3xf32>) -> tensor<2x3xf32>
  // CHECK-DAG: %[[LOG_SOFTMAX_SUM:.*]] = "tf.Sum"(%[[LOG_SOFTMAX_EXP]], %[[AXIS]]) {keep_dims = true} : (tensor<2x3xf32>, tensor<1xi64>) -> tensor<2x1xf32>
  // CHECK-DAG: %[[LOG_SOFTMAX_LOG:.*]] = "tf.Log"(%[[LOG_SOFTMAX_SUM]]) : (tensor<2x1xf32>) -> tensor<2x1xf32>
  // CHECK-DAG: %[[LOG_SOFTMAX:.*]] = "tf.Sub"(%[[LOG_SOFTMAX_SHIFTED]], %[[LOG_SOFTMAX_LOG]]) : (tensor<2x3xf32>, tensor<2x1xf32>) -> tensor<2x3xf32>


  // CHECK-DAG: %[[IS_LABEL_ZERO:.*]] = "tf.Equal"(%[[NEG_LABELS]], %[[ZERO]]) {incompatible_shape_error = true} : (tensor<2x3xf32>, tensor<f32>) -> tensor<2x3xi1>
  // CHECK-DAG: %[[LOSS_INP:.*]] = "tf.Mul"(%[[LOG_SOFTMAX]], %[[NEG_LABELS]]) : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
  // CHECK-DAG: %[[SAFE_LOSS_INP:.*]] = "tf.SelectV2"(%[[IS_LABEL_ZERO]], %[[ZERO]], %[[LOSS_INP]]) : (tensor<2x3xi1>, tensor<f32>, tensor<2x3xf32>) -> tensor<2x3xf32>
  // CHECK-DAG: %[[LOSS:.*]] = "tf.Sum"(%[[SAFE_LOSS_INP]], %[[AXIS]]) {keep_dims = false} : (tensor<2x3xf32>, tensor<1xi64>) -> tensor<2xf32>

  // Softmax expansion.
  // CHECK-DAG: %[[SOFTMAX_MAX:.*]] = "tf.Max"(%arg0, %[[AXIS]]) {keep_dims = true} : (tensor<2x3xf32>, tensor<1xi64>) -> tensor<2x1xf32>
  // CHECK-DAG: %[[SOFTMAX_SHIFTED:.*]] = "tf.Sub"(%[[FEATURES]], %[[SOFTMAX_MAX]]) : (tensor<2x3xf32>, tensor<2x1xf32>) -> tensor<2x3xf32>
  // CHECK-DAG: %[[SOFTMAX_EXP:.*]] = "tf.Exp"(%[[SOFTMAX_SHIFTED]]) : (tensor<2x3xf32>) -> tensor<2x3xf32>
  // CHECK-DAG: %[[SOFTMAX_SUM:.*]] = "tf.Sum"(%[[SOFTMAX_EXP]], %[[AXIS]]) {keep_dims = true} : (tensor<2x3xf32>, tensor<1xi64>) -> tensor<2x1xf32>
  // CHECK-DAG: %[[SOFTMAX:.*]] = "tf.Div"(%[[SOFTMAX_EXP]], %[[SOFTMAX_SUM]]) : (tensor<2x3xf32>, tensor<2x1xf32>) -> tensor<2x3xf32>

  // CHECK-DAG: %[[BACKPROP:.*]] = "tf.Sub"(%[[SOFTMAX]], %[[LABELS]]) : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
  // CHECK: return %[[LOSS]], %[[BACKPROP]]

  %0:2 = "tf.SoftmaxCrossEntropyWithLogits"(%features, %labels) : (tensor<2x3xf32>, tensor<2x3xf32>) -> (tensor<2xf32>, tensor<2x3xf32>)
  func.return %0#0, %0#1 : tensor<2xf32>, tensor<2x3xf32>
}

// CHECK-LABEL: unranked_SoftmaxCrossEntropyWithLogits
func.func @unranked_SoftmaxCrossEntropyWithLogits(%features: tensor<?x?xf32>, %labels: tensor<?x?xf32>) -> (tensor<?xf32>, tensor<?x?xf32>) {
  // Check that unranked inputs are lowered successfully.
  // CHECK-NOT: tf.SoftmaxCrossEntropyWithLogits
  // CHECK-NOT: tf.Softmax
  // CHECK-NOT: tf.LogSoftmax
  %0:2 = "tf.SoftmaxCrossEntropyWithLogits"(%features, %labels) : (tensor<?x?xf32>, tensor<?x?xf32>) -> (tensor<?xf32>, tensor<?x?xf32>)
  func.return %0#0, %0#1 : tensor<?xf32>, tensor<?x?xf32>
}

// CHECK-LABEL: broadcasted_SoftmaxCrossEntropyWithLogits
func.func @broadcasted_SoftmaxCrossEntropyWithLogits(%features: tensor<?x?xf32>, %labels: tensor<3xf32>) -> (tensor<?xf32>, tensor<?x3xf32>) {
  // Check that inputs of different ranks are broadcasted and are lowered successfully.
  // CHECK-NOT: tf.SoftmaxCrossEntropyWithLogits
  // CHECK-NOT: tf.Softmax
  // CHECK-NOT: tf.LogSoftmax
  %0:2 = "tf.SoftmaxCrossEntropyWithLogits"(%features, %labels) : (tensor<?x?xf32>, tensor<3xf32>) -> (tensor<?xf32>, tensor<?x3xf32>)
  func.return %0#0, %0#1 : tensor<?xf32>, tensor<?x3xf32>
}

// CHECK-LABEL: scalar_SoftmaxCrossEntropyWithLogits
func.func @scalar_SoftmaxCrossEntropyWithLogits(%features: tensor<f32>, %labels: tensor<?x?xf32>) -> (tensor<?xf32>, tensor<?x?xf32>) {
  // CHECK: tf.SoftmaxCrossEntropyWithLogits
  // CHECK-NOT: tf.Softmax
  // CHECK-NOT: tf.LogSoftmax
  %0:2 = "tf.SoftmaxCrossEntropyWithLogits"(%features, %labels) : (tensor<f32>, tensor<?x?xf32>) -> (tensor<?xf32>, tensor<?x?xf32>)
  func.return %0#0, %0#1 : tensor<?xf32>, tensor<?x?xf32>
}

// CHECK-LABEL: SparseSoftmaxCrossEntropyWithLogits
// CHECK-SAME: %[[FEATURES:.*]]: tensor<2x3xf32>, %[[SPARSE_LABELS:.*]]: tensor<2xi32>
func.func @SparseSoftmaxCrossEntropyWithLogits(%features: tensor<2x3xf32>, %labels: tensor<2xi32>) -> (tensor<2xf32>, tensor<2x3xf32>) {
  // Convert SPARSE_LABELS to dense LABELS.
  // CHECK-DAG: %[[DEPTH:.*]] = "tf.Const"() {value = dense<3> : tensor<i32>} : () -> tensor<i32>
  // CHECK-DAG: %[[ONE:.*]] = "tf.Const"() {value = dense<1.000000e+00> : tensor<f32>} : () -> tensor<f32>
  // CHECK-DAG: %[[ZERO:.*]] = "tf.Const"() {value = dense<0.000000e+00> : tensor<f32>} : () -> tensor<f32>
  // CHECK-DAG: %[[LABELS:.*]] = "tf.OneHot"(%[[SPARSE_LABELS]], %[[DEPTH]], %[[ONE]], %[[ZERO]]) {axis = 1 : i64} : (tensor<2xi32>, tensor<i32>, tensor<f32>, tensor<f32>) -> tensor<2x3xf32>

  // Adjust labels to have Nan for out of range labels.
  // CHECK-DAG: %[[ZERO_I32:.*]] = "tf.Const"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
  // CHECK-DAG: %[[IS_NEGATIVE:.*]] = "tf.LessEqual"(%[[ZERO_I32]], %arg1) : (tensor<i32>, tensor<2xi32>) -> tensor<2xi1>
  // CHECK-DAG: %[[IS_LESS:.*]] = "tf.Less"(%arg1, %[[DEPTH]]) : (tensor<2xi32>, tensor<i32>) -> tensor<2xi1>
  // CHECK-DAG: %[[IS_WITHIN_RANGE:.*]] = "tf.LogicalAnd"(%[[IS_NEGATIVE]], %[[IS_LESS]]) : (tensor<2xi1>, tensor<2xi1>) -> tensor<2xi1>
  // CHECK-DAG: %[[NAN:.*]] = "tf.Const"() {value = dense<0x7FC00000> : tensor<f32>} : () -> tensor<f32>
  // CHECK-DAG: %[[ZERO_OR_NAN:.*]] = "tf.SelectV2"(%[[IS_WITHIN_RANGE]], %[[ZERO]], %[[NAN]]) : (tensor<2xi1>, tensor<f32>, tensor<f32>) -> tensor<2xf32>
  // CHECK-DAG: %[[NEG_ONE:.*]] = "tf.Const"() {value = dense<-1> : tensor<1xi64>} : () -> tensor<1xi64>
  // CHECK-DAG: %[[RESHAPE:.*]] = "tf.ExpandDims"(%[[ZERO_OR_NAN]], %[[NEG_ONE]]) : (tensor<2xf32>, tensor<1xi64>) -> tensor<2x1xf32>
  // CHECK-DAG: %[[ADJUSTED_LABELS:.*]] = "tf.AddV2"(%[[LABELS]], %[[RESHAPE]]) : (tensor<2x3xf32>, tensor<2x1xf32>) -> tensor<2x3xf32>

  // SoftmaxCrossEntropyWithLogits expansion
  // CHECK: "tf.Log"
  // CHECK: "tf.Exp"
  // CHECK: "tf.Div"
  %0:2 = "tf.SparseSoftmaxCrossEntropyWithLogits"(%features, %labels) : (tensor<2x3xf32>, tensor<2xi32>) -> (tensor<2xf32>, tensor<2x3xf32>)
  func.return %0#0, %0#1 : tensor<2xf32>, tensor<2x3xf32>
}

// CHECK-LABEL: SparseSoftmaxCrossEntropyWithLogits_with_bf16_i64
func.func @SparseSoftmaxCrossEntropyWithLogits_with_bf16_i64(%features: tensor<2x3xbf16>, %labels: tensor<2xi64>) -> (tensor<2xbf16>, tensor<2x3xbf16>) {
  // CHECK-NOT: tf.SparseSoftmaxCrossEntropyWithLogits
  %0:2 = "tf.SparseSoftmaxCrossEntropyWithLogits"(%features, %labels) : (tensor<2x3xbf16>, tensor<2xi64>) -> (tensor<2xbf16>, tensor<2x3xbf16>)
  func.return %0#0, %0#1 : tensor<2xbf16>, tensor<2x3xbf16>
}

// CHECK-LABEL: SparseSoftmaxCrossEntropyWithLogits_with_unranked_labels
func.func @SparseSoftmaxCrossEntropyWithLogits_with_unranked_labels(%features: tensor<2x3xf32>, %labels: tensor<?xi64>) -> (tensor<2xf32>, tensor<2x3xf32>) {
  // CHECK-NOT: tf.SparseSoftmaxCrossEntropyWithLogits
  %0:2 = "tf.SparseSoftmaxCrossEntropyWithLogits"(%features, %labels) : (tensor<2x3xf32>, tensor<?xi64>) -> (tensor<2xf32>, tensor<2x3xf32>)
  func.return %0#0, %0#1 : tensor<2xf32>, tensor<2x3xf32>
}

// CHECK-LABEL: SparseSoftmaxCrossEntropyWithLogits_with_dynamic_labels
func.func @SparseSoftmaxCrossEntropyWithLogits_with_dynamic_labels(%features: tensor<2x3xf32>, %labels: tensor<*xi64>) -> (tensor<2xf32>, tensor<2x3xf32>) {
  // CHECK-NOT: tf.SparseSoftmaxCrossEntropyWithLogits
  %0:2 = "tf.SparseSoftmaxCrossEntropyWithLogits"(%features, %labels) : (tensor<2x3xf32>, tensor<*xi64>) -> (tensor<2xf32>, tensor<2x3xf32>)
  func.return %0#0, %0#1 : tensor<2xf32>, tensor<2x3xf32>
}

// CHECK-LABEL: SparseSoftmaxCrossEntropyWithLogits_with_dynamic
func.func @SparseSoftmaxCrossEntropyWithLogits_with_dynamic(%features: tensor<*xbf16>, %labels: tensor<*xi64>) -> (tensor<2xbf16>, tensor<*xbf16>) {
  // CHECK: tf.SparseSoftmaxCrossEntropyWithLogits
  %0:2 = "tf.SparseSoftmaxCrossEntropyWithLogits"(%features, %labels) : (tensor<*xbf16>, tensor<*xi64>) -> (tensor<2xbf16>, tensor<*xbf16>)
  func.return %0#0, %0#1 : tensor<2xbf16>, tensor<*xbf16>
}

// CHECK-LABEL: func @tanhgrad_float
// CHECK-SAME: (%[[Y:.*]]: tensor<*xf32>, %[[DY:.*]]: tensor<*xf32>)
func.func @tanhgrad_float(%y : tensor<*xf32>, %dy: tensor<*xf32>) -> tensor<*xf32> {
  // CHECK: %[[ONE:.*]] = "tf.Const"() {value = dense<1.000000e+00> : tensor<f32>} : () -> tensor<f32>
  // CHECK: %[[Y_SQUARE:.*]] = "tf.Mul"(%[[Y]], %[[Y]]) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
  // CHECK: %[[SUB:.*]] = "tf.Sub"(%[[ONE]], %[[Y_SQUARE]]) : (tensor<f32>, tensor<*xf32>) -> tensor<*xf32>
  // CHECK: %[[RESULT:.*]] = "tf.Mul"(%[[DY]], %[[SUB]]) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
  %0 = "tf.TanhGrad"(%y, %dy) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>

  // CHECK: return %[[RESULT]]
  func.return %0 : tensor<*xf32>
}

// CHECK-LABEL: func @tanhgrad_complex
// CHECK-SAME: (%[[Y:.*]]: tensor<*xcomplex<f32>>, %[[DY:.*]]: tensor<*xcomplex<f32>>)
func.func @tanhgrad_complex(%y : tensor<*xcomplex<f32>>, %dy: tensor<*xcomplex<f32>>) -> tensor<*xcomplex<f32>> {
  // CHECK-NOT: tf.TanhGrad
  %0 = "tf.TanhGrad"(%y, %dy) : (tensor<*xcomplex<f32>>, tensor<*xcomplex<f32>>) -> tensor<*xcomplex<f32>>

  func.return %0 : tensor<*xcomplex<f32>>
}

// CHECK-LABEL: func @ZerosLike_unranked
func.func @ZerosLike_unranked(%arg0: tensor<*xi32>) -> tensor<*xi32> {
  // CHECK: %[[ZERO:.*]] = "tf.Const"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
  // CHECK: %[[SHAPE:.*]] = "tf.Shape"(%arg0) : (tensor<*xi32>) -> tensor<?xi64>
  // CHECK: "tf.BroadcastTo"(%[[ZERO]], %[[SHAPE]]) : (tensor<i32>, tensor<?xi64>) -> tensor<*xi32>

  %0 = "tf.ZerosLike"(%arg0) : (tensor<*xi32>) -> tensor<*xi32>
  func.return %0 : tensor<*xi32>
}

// CHECK-LABEL: func @ZerosLike_variant
func.func @ZerosLike_variant(%arg0: tensor<!tf_type.variant<tensor<2xi32>>>) -> tensor<!tf_type.variant<tensor<2xi32>>> {
  // CHECK: tf.ZerosLike
  %0 = "tf.ZerosLike"(%arg0) : (tensor<!tf_type.variant<tensor<2xi32>>>) -> tensor<!tf_type.variant<tensor<2xi32>>>
  func.return %0 : tensor<!tf_type.variant<tensor<2xi32>>>
}

// CHECK-LABEL: func @OnesLike_unranked
func.func @OnesLike_unranked(%arg0: tensor<*xi32>) -> tensor<*xi32> {
  // CHECK: %[[ONE:.*]] = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
  // CHECK: %[[SHAPE:.*]] = "tf.Shape"(%arg0) : (tensor<*xi32>) -> tensor<?xi64>
  // CHECK: "tf.BroadcastTo"(%[[ONE]], %[[SHAPE]]) : (tensor<i32>, tensor<?xi64>) -> tensor<*xi32>

  %0 = "tf.OnesLike"(%arg0) : (tensor<*xi32>) -> tensor<*xi32>
  func.return %0 : tensor<*xi32>
}

// CHECK-LABEL: func @addN_2
func.func @addN_2(%arg0: tensor<*xf32>, %arg1: tensor<*xf32>) -> tensor<*xf32> {
  // CHECK: %[[SUM0:.*]] = "tf.AddV2"(%arg0, %arg1)
  // return %[[SUM0]]
  %0 = "tf.AddN"(%arg0, %arg1) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// CHECK-LABEL: func @addN_3
func.func @addN_3(%arg0: tensor<*xf32>, %arg1: tensor<*xf32>, %arg2: tensor<*xf32>) -> tensor<*xf32> {
  // CHECK: %[[SUM0:.*]] = "tf.AddV2"(%arg0, %arg1)
  // CHECK: %[[SUM1:.*]] = "tf.AddV2"(%[[SUM0]], %arg2)
  // return %[[SUM1]]
  %0 = "tf.AddN"(%arg0, %arg1, %arg2) : (tensor<*xf32>, tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// CHECK-LABEL: func @addN_4
func.func @addN_4(%arg0: tensor<*xf32>, %arg1: tensor<*xf32>, %arg2: tensor<*xf32>, %arg3: tensor<*xf32>) -> tensor<*xf32> {
  // CHECK: %[[SUM0:.*]] = "tf.AddV2"(%arg0, %arg1)
  // CHECK: %[[SUM1:.*]] = "tf.AddV2"(%arg2, %arg3)
  // CHECK: %[[SUM2:.*]] = "tf.AddV2"(%[[SUM0]], %[[SUM1]])
  // return %[[SUM2]]
  %0 = "tf.AddN"(%arg0, %arg1, %arg2, %arg3) : (tensor<*xf32>, tensor<*xf32>, tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// CHECK-LABEL: func @addN_5
func.func @addN_5(%arg0: tensor<*xf32>, %arg1: tensor<*xf32>, %arg2: tensor<*xf32>, %arg3: tensor<*xf32>, %arg4: tensor<*xf32>) -> tensor<*xf32> {
  // CHECK: %[[SUM0:.*]] = "tf.AddV2"(%arg0, %arg1)
  // CHECK: %[[SUM1:.*]] = "tf.AddV2"(%arg2, %arg3)
  // CHECK: %[[SUM2:.*]] = "tf.AddV2"(%[[SUM0]], %[[SUM1]])
  // CHECK: %[[SUM3:.*]] = "tf.AddV2"(%[[SUM2]], %arg4)
  // return %[[SUM3]]
  %0 = "tf.AddN"(%arg0, %arg1, %arg2, %arg3, %arg4) : (tensor<*xf32>, tensor<*xf32>, tensor<*xf32>, tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// CHECK-LABEL: func @addN_variant
func.func @addN_variant(%arg0: tensor<!tf_type.variant<tensor<2xf32>>>, %arg1: tensor<!tf_type.variant<tensor<2xf32>>>, %arg2: tensor<!tf_type.variant<tensor<2xf32>>>) -> tensor<!tf_type.variant<tensor<2xf32>>> {
  // CHECK: tf.AddN
  %0 = "tf.AddN"(%arg0, %arg1, %arg2) : (tensor<!tf_type.variant<tensor<2xf32>>>, tensor<!tf_type.variant<tensor<2xf32>>>, tensor<!tf_type.variant<tensor<2xf32>>>) -> tensor<!tf_type.variant<tensor<2xf32>>>
  func.return %0 : tensor<!tf_type.variant<tensor<2xf32>>>
}

// CHECK-LABEL: func @DynamicStitch_simple
func.func @DynamicStitch_simple(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
  // CHECK: %[[AXIS:.*]] = "tf.Const"() {value = dense<0> : tensor<i64>} : () -> tensor<i64>
  // CHECK: %[[ITEMS:.*]]:2 = "tf.Unpack"(%arg0) {axis = 0 : i64} : (tensor<2x2xf32>) -> (tensor<2xf32>, tensor<2xf32>)
  // CHECK-DAG: %[[ITEMS_1:.*]] = "tf.ExpandDims"(%[[ITEMS]]#1, %[[AXIS]])
  // CHECK-DAG: %[[ITEMS_0:.*]] = "tf.ExpandDims"(%[[ITEMS]]#0, %[[AXIS]])
  // CHECK: %[[RESULT:.*]] = "tf.ConcatV2"(%[[ITEMS_1]], %[[ITEMS_0]], %[[AXIS]]) : (tensor<1x2xf32>, tensor<1x2xf32>, tensor<i64>) -> tensor<2x2xf32>
  // CHECK: return %[[RESULT]]

  %indices = "tf.Const"() {value = dense<[1, 0]> : tensor<2xi32>} : () -> tensor<2xi32>
  %0 = "tf.DynamicStitch"(%indices, %arg0) : (tensor<2xi32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  func.return %0 : tensor<2x2xf32>
}

// CHECK-LABEL: DynamicStitch_scalar_matrix_indices
func.func @DynamicStitch_scalar_matrix_indices(%arg0: tensor<2xf32>, %arg1: tensor<2x2x2xf32>) -> (tensor<5x2xf32>) {
  // CHECK-DAG: %[[SHAPE:.*]] = "tf.Const"() {value = dense<[-1, 2]> : tensor<2xi64>} : () -> tensor<2xi64>
  // CHECK-DAG: %[[INP0:.*]] = "tf.Reshape"(%arg0, %[[SHAPE]]) : (tensor<2xf32>, tensor<2xi64>) -> tensor<1x2xf32>
  // CHECK-DAG: %[[ITEMS0:.*]] = "tf.Unpack"(%[[INP0]]) {axis = 0 : i64} : (tensor<1x2xf32>) -> tensor<2xf32>
  // CHECK-DAG: %[[INP1:.*]] = "tf.Reshape"(%arg1, %[[SHAPE]]) : (tensor<2x2x2xf32>, tensor<2xi64>) -> tensor<4x2xf32>
  // CHECK-DAG: %[[ITEMS1:.*]]:4 = "tf.Unpack"(%[[INP1]]) {axis = 0 : i64} : (tensor<4x2xf32>) -> (tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>)
  // CHECK-DAG: %[[AXIS:.*]] = "tf.Const"() {value = dense<0> : tensor<i64>} : () -> tensor<i64>
  // CHECK-DAG: %[[ITEMS1_3:.*]] = "tf.ExpandDims"(%[[ITEMS1]]#3, %[[AXIS]])
  // CHECK-DAG: %[[ITEMS1_2:.*]] = "tf.ExpandDims"(%[[ITEMS1]]#2, %[[AXIS]])
  // CHECK-DAG: %[[ITEMS1_1:.*]] = "tf.ExpandDims"(%[[ITEMS1]]#1, %[[AXIS]])
  // CHECK-DAG: %[[ITEMS1_0:.*]] = "tf.ExpandDims"(%[[ITEMS1]]#0, %[[AXIS]])
  // CHECK-DAG: %[[ITEMS0_0:.*]] = "tf.ExpandDims"(%[[ITEMS0]], %[[AXIS]])
  // CHECK-DAG: "tf.ConcatV2"(%[[ITEMS1_3]], %[[ITEMS1_2]], %[[ITEMS1_1]], %[[ITEMS1_0]], %[[ITEMS0_0]], %[[AXIS]]) : (tensor<1x2xf32>, tensor<1x2xf32>, tensor<1x2xf32>, tensor<1x2xf32>, tensor<1x2xf32>, tensor<i64>) -> tensor<5x2xf32>

  %indices0 = "tf.Const"() {value = dense<4> : tensor<i32>} : () -> tensor<i32>
  %indices1 = "tf.Const"() {value = dense<[[3, 2], [1, 0]]> : tensor<2x2xi32>} : () -> tensor<2x2xi32>
  %0 = "tf.DynamicStitch"(%indices0, %indices1, %arg0, %arg1) : (tensor<i32>, tensor<2x2xi32>, tensor<2xf32>, tensor<2x2x2xf32>) -> tensor<5x2xf32>
  func.return %0 : tensor<5x2xf32>
}

// Verify that custom types are lowered and have legal output.
// CHECK-LABEL: func @DynamicStitch_uint8
func.func @DynamicStitch_uint8(%arg0: tensor<2x2xui8>) -> tensor<2x2xui8> {
  // CHECK-NOT: tf.DynamicStitch

  %indices = "tf.Const"() {value = dense<[1, 0]> : tensor<2xi32>} : () -> tensor<2xi32>
  %0 = "tf.DynamicStitch"(%indices, %arg0) : (tensor<2xi32>, tensor<2x2xui8>) -> tensor<2x2xui8>
  func.return %0 : tensor<2x2xui8>
}

// CHECK-LABEL: func @DynamicStitch_scalar_item
func.func @DynamicStitch_scalar_item(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  // CHECK-DAG: %[[ITEMS:.*]]:2 = "tf.Unpack"(%arg0) {axis = 0 : i64} : (tensor<2xf32>) -> (tensor<f32>, tensor<f32>)
  // CHECK-DAG: %[[AXIS:.*]] = "tf.Const"() {value = dense<0> : tensor<i64>} : () -> tensor<i64>
  // CHECK-DAG: %[[ITEMS_1:.*]] = "tf.ExpandDims"(%[[ITEMS]]#1, %[[AXIS]])
  // CHECK-DAG: %[[ITEMS_0:.*]] = "tf.ExpandDims"(%[[ITEMS]]#0, %[[AXIS]])
  // CHECK-DAG: %[[RESULT:.*]] = "tf.ConcatV2"(%[[ITEMS_1]], %[[ITEMS_0]], %[[AXIS]]) : (tensor<1xf32>, tensor<1xf32>, tensor<i64>) -> tensor<2xf32>
  // CHECK: return %[[RESULT]]

  %indices = "tf.Const"() {value = dense<[1, 0]> : tensor<2xi32>} : () -> tensor<2xi32>
  %0 = "tf.DynamicStitch"(%indices, %arg0) : (tensor<2xi32>, tensor<2xf32>) -> tensor<2xf32>
  func.return %0 : tensor<2xf32>
}

// CHECK-LABEL: func @DynamicStitch_matrix_item
func.func @DynamicStitch_matrix_item(%arg0: tensor<2x2x2xf32>) -> tensor<2x2x2xf32> {
  // CHECK-DAG: %[[ITEMS:.*]]:2 = "tf.Unpack"(%arg0) {axis = 0 : i64} : (tensor<2x2x2xf32>) -> (tensor<2x2xf32>, tensor<2x2xf32>)
  // CHECK-DAG: %[[AXIS:.*]] = "tf.Const"() {value = dense<0> : tensor<i64>} : () -> tensor<i64>
  // CHECK-DAG: %[[ITEMS_1:.*]] = "tf.ExpandDims"(%[[ITEMS]]#1, %[[AXIS]])
  // CHECK-DAG: %[[ITEMS_0:.*]] = "tf.ExpandDims"(%[[ITEMS]]#0, %[[AXIS]])
  // CHECK-DAG: %[[RESULT:.*]] = "tf.ConcatV2"(%[[ITEMS_1]], %[[ITEMS_0]], %[[AXIS]]) : (tensor<1x2x2xf32>, tensor<1x2x2xf32>, tensor<i64>) -> tensor<2x2x2xf32>
  // CHECK: return %[[RESULT]]

  %indices = "tf.Const"() {value = dense<[1, 0]> : tensor<2xi32>} : () -> tensor<2xi32>
  %0 = "tf.DynamicStitch"(%indices, %arg0) : (tensor<2xi32>, tensor<2x2x2xf32>) -> tensor<2x2x2xf32>
  func.return %0 : tensor<2x2x2xf32>
}

// CHECK-LABEL: func @DynamicStitch_dynamic
func.func @DynamicStitch_dynamic(%arg0: tensor<*xi32>, %arg1: tensor<*xf32>) -> tensor<*xf32> {
  // CHECK: tf.DynamicStitch
  %0 = "tf.DynamicStitch"(%arg0, %arg1) : (tensor<*xi32>, tensor<*xf32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// CHECK-LABEL: func @DynamicStitch_duplicates
func.func @DynamicStitch_duplicates(%arg0: tensor<2x2xf32>) -> tensor<1x2xf32> {
  // CHECK-DAG: %[[ITEMS:.*]]:2 = "tf.Unpack"(%arg0) {axis = 0 : i64} : (tensor<2x2xf32>) -> (tensor<2xf32>, tensor<2xf32>)
  // CHECK-DAG: %[[AXIS:.*]] = "tf.Const"() {value = dense<0> : tensor<i64>} : () -> tensor<i64>
  // CHECK-DAG: %[[ITEMS_1:.*]] = "tf.ExpandDims"(%[[ITEMS]]#1, %[[AXIS]])
  // CHECK-DAG: %[[RESULT:.*]] = "tf.ConcatV2"(%[[ITEMS_1]], %[[AXIS]]) : (tensor<1x2xf32>, tensor<i64>) -> tensor<1x2xf32>
  // CHECK: return %[[RESULT]]

  %indices = "tf.Const"() {value = dense<[0, 0]> : tensor<2xi32>} : () -> tensor<2xi32>
  %0 = "tf.DynamicStitch"(%indices, %arg0) : (tensor<2xi32>, tensor<2x2xf32>) -> tensor<1x2xf32>
  func.return %0 : tensor<1x2xf32>
}

// CHECK-LABEL: func @ParallelDynamicStitch
func.func @ParallelDynamicStitch(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %indices = "tf.Const"() {value = dense<[1, 0]> : tensor<2xi32>} : () -> tensor<2xi32>
  // CHECK-NOT: tf.ParallelDynamicStitch
  %0 = "tf.ParallelDynamicStitch"(%indices, %arg0) : (tensor<2xi32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  func.return %0 : tensor<2x2xf32>
}

// CHECK-LABEL: @Reciprocal_i32
func.func @Reciprocal_i32(%arg0: tensor<*xi32>) -> tensor<*xi32> {
  // CHECK: %[[ONE:.*]] = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
  // CHECK: "tf.Div"(%[[ONE]], %arg0) : (tensor<i32>, tensor<*xi32>) -> tensor<*xi32>
  %0 = "tf.Reciprocal"(%arg0) : (tensor<*xi32>) -> tensor<*xi32>
  func.return %0 : tensor<*xi32>
}

// CHECK-LABEL: @Reciprocal_f32
func.func @Reciprocal_f32(%arg0: tensor<*xf32>) -> tensor<*xf32> {
  // CHECK: %[[ONE:.*]] = "tf.Const"() {value = dense<1.000000e+00> : tensor<f32>} : () -> tensor<f32>
  // CHECK: "tf.Div"(%[[ONE]], %arg0) : (tensor<f32>, tensor<*xf32>) -> tensor<*xf32>
  %0 = "tf.Reciprocal"(%arg0) : (tensor<*xf32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// CHECK-LABEL: @Reciprocal_complexf32
func.func @Reciprocal_complexf32(%arg0: tensor<*xcomplex<f32>>) -> tensor<*xcomplex<f32>> {
  // CHECK: %[[ONE:.*]] = "tf.Const"() {value = dense<(1.000000e+00,0.000000e+00)> : tensor<complex<f32>>} : () -> tensor<complex<f32>>
  // CHECK: "tf.Div"(%[[ONE]], %arg0) : (tensor<complex<f32>>, tensor<*xcomplex<f32>>) -> tensor<*xcomplex<f32>>
  %0 = "tf.Reciprocal"(%arg0) : (tensor<*xcomplex<f32>>) -> tensor<*xcomplex<f32>>
  func.return %0 : tensor<*xcomplex<f32>>
}

// CHECK-LABEL: @Reciprocal_complexf64
func.func @Reciprocal_complexf64(%arg0: tensor<*xcomplex<f64>>) -> tensor<*xcomplex<f64>> {
  // CHECK: %[[ONE:.*]] = "tf.Const"() {value = dense<(1.000000e+00,0.000000e+00)> : tensor<complex<f64>>} : () -> tensor<complex<f64>>
  // CHECK: "tf.Div"(%[[ONE]], %arg0) : (tensor<complex<f64>>, tensor<*xcomplex<f64>>) -> tensor<*xcomplex<f64>>
  %0 = "tf.Reciprocal"(%arg0) : (tensor<*xcomplex<f64>>) -> tensor<*xcomplex<f64>>
  func.return %0 : tensor<*xcomplex<f64>>
}

// Inv is the same as Reciprocal
// CHECK-LABEL: @Inv_i32
func.func @Inv_i32(%arg0: tensor<*xi32>) -> tensor<*xi32> {
  // CHECK: %[[ONE:.*]] = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
  // CHECK: "tf.Div"(%[[ONE]], %arg0) : (tensor<i32>, tensor<*xi32>) -> tensor<*xi32>
  %0 = "tf.Inv"(%arg0) : (tensor<*xi32>) -> tensor<*xi32>
  func.return %0 : tensor<*xi32>
}

// CHECK-LABEL: @ScatterNd
func.func @ScatterNd(%arg0: tensor<4x1xi32>, %arg1: tensor<4xf32>) -> tensor<8xf32> {
  // CHECK: %[[ZERO:.*]] = "tf.Const"() {value = dense<0.000000e+00> : tensor<8xf32>} : () -> tensor<8xf32>
  // CHECK: "tf.TensorScatterAdd"(%[[ZERO]], %arg0, %arg1) : (tensor<8xf32>, tensor<4x1xi32>, tensor<4xf32>) -> tensor<8xf32>

  %shape = "tf.Const"() {value = dense<[8]> : tensor<1xi32>} : () -> tensor<1xi32>
  %0 = "tf.ScatterNd"(%arg0, %arg1, %shape) : (tensor<4x1xi32>, tensor<4xf32>, tensor<1xi32>) -> tensor<8xf32>
  func.return %0 : tensor<8xf32>
}

// CHECK-LABEL: @_UnaryOpsComposition
// CHECK-SAME: %[[ARG0:.*]]: tensor<4xf32>
func.func @_UnaryOpsComposition(%arg0: tensor<4xf32>) -> tensor<4xf32> {

  // CHECK: %[[RESULT0:.*]] = "tf.Asin"(%[[ARG0]])
  // CHECK: %[[RESULT1:.*]] = "tf.Abs"(%[[RESULT0]])
  // CHECK: %[[RESULT2:.*]] = "tf.Log"(%[[RESULT1]])
  // CHECK: return %[[RESULT2]]

  %0 = "tf._UnaryOpsComposition"(%arg0) {op_names = ["Asin", "Abs", "Log"]} : (tensor<4xf32>) -> tensor<4xf32>
  func.return %0 : tensor<4xf32>
}


// CHECK-LABEL: @round_int
func.func @round_int(%arg0: tensor<2xi32>) -> tensor<2xi32> {
  // CHECK: [[IDENTITY:%.+]] = "tf.Identity"(%arg0)
  %0 = "tf.Round"(%arg0) : (tensor<2xi32>) -> tensor<2xi32>
  // CHECK: return [[IDENTITY]]
  func.return %0 : tensor<2xi32>
}

// CHECK-LABEL: @round
func.func @round(%arg0: tensor<2xf32>) -> tensor<2xf32> {
  // CHECK-DAG: %[[ZERO:.*]] = "tf.Const"() {value = dense<0.000000e+00> : tensor<f32>} : () -> tensor<f32>
  // CHECK-DAG: %[[HALF:.*]] = "tf.Const"() {value = dense<5.000000e-01> : tensor<f32>} : () -> tensor<f32>
  // CHECK-DAG: %[[TWO:.*]] = "tf.Const"() {value = dense<2.000000e+00> : tensor<f32>} : () -> tensor<f32>
  // CHECK-DAG: %[[ONE:.*]] = "tf.Const"() {value = dense<1.000000e+00> : tensor<f32>} : () -> tensor<f32>
  // CHECK: %[[ROUND_VAL:.*]] = "tf.Floor"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>
  // CHECK: %[[FRACTION:.*]] = "tf.Sub"(%arg0, %[[ROUND_VAL]]) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
  // CHECK: %[[GT:.*]] = "tf.Greater"(%[[FRACTION]], %[[HALF]]) : (tensor<2xf32>, tensor<f32>) -> tensor<2xi1>
  // CHECK: %[[EQ:.*]] = "tf.Equal"(%[[FRACTION]], %[[HALF]]) {incompatible_shape_error = true} : (tensor<2xf32>, tensor<f32>) -> tensor<2xi1>
  // CHECK: %[[MUL1:.*]] = "tf.Mul"(%arg0, %[[HALF]]) : (tensor<2xf32>, tensor<f32>) -> tensor<2xf32>
  // CHECK: %[[FLOOR:.*]] = "tf.Floor"(%[[MUL1]]) : (tensor<2xf32>) -> tensor<2xf32>
  // CHECK: %[[MUL2:.*]] = "tf.Mul"(%[[FLOOR]], %[[TWO]]) : (tensor<2xf32>, tensor<f32>) -> tensor<2xf32>
  // CHECK: %[[NEAREST_EVEN_INT:.*]] = "tf.Sub"(%[[ROUND_VAL]], %[[MUL2]]) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
  // CHECK: %[[IS_ODD:.*]] = "tf.Equal"(%[[NEAREST_EVEN_INT]], %[[ONE]]) {incompatible_shape_error = true} : (tensor<2xf32>, tensor<f32>) -> tensor<2xi1>
  // CHECK: %[[AND:.*]] = "tf.LogicalAnd"(%[[EQ]], %[[IS_ODD]]) : (tensor<2xi1>, tensor<2xi1>) -> tensor<2xi1>
  // CHECK: %[[OR:.*]] = "tf.LogicalOr"(%[[GT]], %[[AND]]) : (tensor<2xi1>, tensor<2xi1>) -> tensor<2xi1>
  // CHECK: %[[ADD:.*]] = "tf.AddV2"(%[[ROUND_VAL]], %[[ONE]]) : (tensor<2xf32>, tensor<f32>) -> tensor<2xf32>
  // CHECK: %[[INNER_SELECT:.*]] = "tf.SelectV2"(%[[OR]], %[[ADD]], %[[ROUND_VAL]]) : (tensor<2xi1>, tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
  // CHECK-DAG: %[[IS_ZERO:.*]] = "tf.Equal"(%[[INNER_SELECT]], %[[ZERO]]) {incompatible_shape_error = true}
  // CHECK-DAG: %[[SELECT:.*]] = "tf.SelectV2"(%[[IS_ZERO]], %[[ZERO]], %[[INNER_SELECT]])
  %0 = "tf.Round"(%arg0) : (tensor<2xf32>) -> tensor<2xf32>

  // CHECK: return %[[SELECT]]
  func.return %0 : tensor<2xf32>
}

// CHECK-LABEL: func @round_dynamic
func.func @round_dynamic(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  // CHECK-NOT: tf.Round
  %0 = "tf.Round"(%arg0) : (tensor<?xf32>) -> tensor<?xf32>
  func.return %0 : tensor<?xf32>
}

// CHECK-LABEL: func @rint_dynamic
func.func @rint_dynamic(%arg0: tensor<?xf32>) -> tensor<?xf32> {
  // CHECK-DAG: %[[ZERO:.*]] = "tf.Const"() {value = dense<0.000000e+00> : tensor<f32>} : () -> tensor<f32>
  // CHECK-DAG: %[[HALF:.*]] = "tf.Const"() {value = dense<5.000000e-01> : tensor<f32>} : () -> tensor<f32>
  // CHECK-DAG: %[[TWO:.*]] = "tf.Const"() {value = dense<2.000000e+00> : tensor<f32>} : () -> tensor<f32>
  // CHECK-DAG: %[[ONE:.*]] = "tf.Const"() {value = dense<1.000000e+00> : tensor<f32>} : () -> tensor<f32>
  // CHECK: %[[ROUND_VAL:.*]] = "tf.Floor"(%arg0) : (tensor<?xf32>) -> tensor<?xf32>
  // CHECK: %[[FRACTION:.*]] = "tf.Sub"(%arg0, %[[ROUND_VAL]]) : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  // CHECK: %[[GT:.*]] = "tf.Greater"(%[[FRACTION]], %[[HALF]]) : (tensor<?xf32>, tensor<f32>) -> tensor<?xi1>
  // CHECK: %[[EQ:.*]] = "tf.Equal"(%[[FRACTION]], %[[HALF]]) {incompatible_shape_error = true} : (tensor<?xf32>, tensor<f32>) -> tensor<?xi1>
  // CHECK: %[[MUL1:.*]] = "tf.Mul"(%arg0, %[[HALF]]) : (tensor<?xf32>, tensor<f32>) -> tensor<?xf32>
  // CHECK: %[[FLOOR:.*]] = "tf.Floor"(%[[MUL1]]) : (tensor<?xf32>) -> tensor<?xf32>
  // CHECK: %[[MUL2:.*]] = "tf.Mul"(%[[FLOOR]], %[[TWO]]) : (tensor<?xf32>, tensor<f32>) -> tensor<?xf32>
  // CHECK: %[[NEAREST_EVEN_INT:.*]] = "tf.Sub"(%[[ROUND_VAL]], %[[MUL2]]) : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  // CHECK: %[[IS_ODD:.*]] = "tf.Equal"(%[[NEAREST_EVEN_INT]], %[[ONE]]) {incompatible_shape_error = true} : (tensor<?xf32>, tensor<f32>) -> tensor<?xi1>
  // CHECK: %[[AND:.*]] = "tf.LogicalAnd"(%[[EQ]], %[[IS_ODD]]) : (tensor<?xi1>, tensor<?xi1>) -> tensor<?xi1>
  // CHECK: %[[OR:.*]] = "tf.LogicalOr"(%[[GT]], %[[AND]]) : (tensor<?xi1>, tensor<?xi1>) -> tensor<?xi1>
  // CHECK: %[[ADD:.*]] = "tf.AddV2"(%[[ROUND_VAL]], %[[ONE]]) : (tensor<?xf32>, tensor<f32>) -> tensor<?xf32>
  // CHECK: %[[INNER_SELECT:.*]] = "tf.SelectV2"(%[[OR]], %[[ADD]], %[[ROUND_VAL]]) : (tensor<?xi1>, tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  // CHECK: %[[IS_ZERO:.*]] = "tf.Equal"(%[[INNER_SELECT]], %[[ZERO]]) {incompatible_shape_error = true}
  // CHECK: %[[SELECT:.*]] = "tf.SelectV2"(%[[IS_ZERO]], %[[ZERO]], %[[INNER_SELECT]])
  %0 = "tf.Rint"(%arg0) : (tensor<?xf32>) -> tensor<?xf32>

  // CHECK: return %[[SELECT]]
  func.return %0 : tensor<?xf32>
}

// CHECK-LABEL: func @round_unranked
func.func @round_unranked(%arg0: tensor<*xf32>) -> tensor<*xf32> {
  // CHECK-NOT: tf.Round
  %0 = "tf.Round"(%arg0) : (tensor<*xf32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// CHECK-LABEL: func @lgamma
func.func @lgamma(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  // The lowering for lgamma is complicated, which makes it awkward to write a
  // complete test for it here. Instead we test that Lgamma is at least being
  // lowered here and rely on UnaryOpsTest.testFloatOps and other TensorFlow
  // tests to check it is lowered correctly and with sufficient precision.
  // CHECK-NOT: tf.Lgamma
  %0 = "tf.Lgamma"(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
  func.return %0 : tensor<4xf32>
}

// CHECK-LABEL: func @imag_resize_nearest
func.func @imag_resize_nearest(%arg0: tensor<1x7x7x1xi32>) -> tensor<1x3x3x1xi32> {
  %shape = "tf.Const"() {device = "", value = dense<3> : tensor<2xi32>} : () -> tensor<2xi32>

  // CHECK-DAG: [[VAL0:%.+]] = "tf.Const"() {value = dense<1> : tensor<i32>}
  // CHECK-DAG: [[VAL1:%.+]] = "tf.Const"() {value = dense<[1, 3, 3, 1]>
  // CHECK-DAG: [[VAL2:%.+]] = "tf.Const"() {value = dense<[1, 49, 1]>
  // CHECK-DAG: [[VAL3:%.+]] = "tf.Const"() {value = dense<[0, 2, 4, 14, 16, 18, 28, 30, 32]> : tensor<9xi32>}
  // CHECK: [[VAL4:%.+]] = "tf.Reshape"(%arg0, [[VAL2]])
  // CHECK: [[VAL5:%.+]] = "tf.GatherV2"([[VAL4]], [[VAL3]], [[VAL0]]) {batch_dims = 0 : i64}
  // CHECK: [[VAL6:%.+]] = "tf.Reshape"([[VAL5]], [[VAL1]])
  // CHECK: return [[VAL6]]
  %resize = "tf.ResizeNearestNeighbor"(%arg0, %shape) {align_corners = false, device = "", half_pixel_centers = false} : (tensor<1x7x7x1xi32>, tensor<2xi32>) -> tensor<1x3x3x1xi32>
  func.return %resize: tensor<1x3x3x1xi32>
}

// CHECK-LABEL: func @imag_resize_nearest_dyn_img
func.func @imag_resize_nearest_dyn_img(%arg0: tensor<1x?x?x1xi32>) -> tensor<1x3x3x1xi32> {
  %shape = "tf.Const"() {device = "", value = dense<3> : tensor<2xi32>} : () -> tensor<2xi32>

  // CHECK-DAG: [[VAL0:%.+]] = "tf.Const"() {value = dense<1> : tensor<i32>}
  // CHECK-DAG: [[VAL1:%.+]] = "tf.Const"() {value = dense<[3, 1]> : tensor<2xi32>}
  // CHECK-DAG: [[VAL2:%.+]] = "tf.Const"() {value = dense<9> : tensor<1xi32>}
  // CHECK-DAG: [[VAL3:%.+]] = "tf.Const"() {value = dense<3> : tensor<1xi32>}
  // CHECK-DAG: [[VAL4:%.+]] = "tf.Const"() {value = dense<[1, 3]> : tensor<2xi32>}
  // CHECK-DAG: [[VAL5:%.+]] = "tf.Const"() {value = dense<[0.000000e+00, 1.000000e+00, 2.000000e+00]>
  // CHECK-DAG: [[VAL6:%.+]] = "tf.Const"() {value = dense<3.000000e+00> : tensor<f32>}
  // CHECK-DAG: [[VAL7:%.+]] = "tf.Const"() {value = dense<0> : tensor<i64>}
  // CHECK: [[VAL8:%.+]] = "tf.Shape"(%arg0)
  // CHECK: [[VAL9:%.+]] = "tf.Cast"([[VAL8]])
  // CHECK: [[VAL10:%.+]]:4 = "tf.Unpack"([[VAL9]]) {axis = 0 : i64}
  // CHECK: [[VAL11:%.+]] = "tf.Mul"([[VAL10]]#1, [[VAL10]]#2)
  // CHECK: [[VAL12:%.+]] = "tf.ExpandDims"([[VAL10]]#0, [[VAL7]])
  // CHECK: [[VAL13:%.+]] = "tf.ExpandDims"([[VAL10]]#3, [[VAL7]])
  // CHECK: [[VAL14:%.+]] = "tf.ConcatV2"([[VAL12]], [[VAL3]], [[VAL3]], [[VAL13]], [[VAL7]])
  // CHECK: [[VAL15:%.+]] = "tf.Cast"([[VAL10]]#1)
  // CHECK: [[VAL16:%.+]] = "tf.Div"([[VAL15]], [[VAL6]])
  // CHECK: [[VAL17:%.+]] = "tf.Mul"([[VAL16]], [[VAL5]])
  // CHECK: [[VAL18:%.+]] = "tf.Cast"([[VAL17]])
  // CHECK: [[VAL19:%.+]] = "tf.Reshape"([[VAL18]], [[VAL1]])
  // CHECK: [[VAL20:%.+]] = "tf.Mul"([[VAL19]], [[VAL10]]#2)
  // CHECK: [[VAL21:%.+]] = "tf.Cast"([[VAL10]]#2)
  // CHECK: [[VAL22:%.+]] = "tf.Div"([[VAL21]], [[VAL6]])
  // CHECK: [[VAL23:%.+]] = "tf.Mul"([[VAL22]], [[VAL5]])
  // CHECK: [[VAL24:%.+]] = "tf.Cast"([[VAL23]])
  // CHECK: [[VAL25:%.+]] = "tf.Reshape"([[VAL24]], [[VAL4]])
  // CHECK: [[VAL26:%.+]] = "tf.AddV2"([[VAL20]], [[VAL25]])
  // CHECK: [[VAL27:%.+]] = "tf.Reshape"([[VAL26]], [[VAL2]])
  // CHECK: [[VAL28:%.+]] = "tf.ExpandDims"([[VAL10]]#0, [[VAL7]])
  // CHECK: [[VAL29:%.+]] = "tf.ExpandDims"([[VAL11]], [[VAL7]])
  // CHECK: [[VAL30:%.+]] = "tf.ExpandDims"([[VAL10]]#3, [[VAL7]])
  // CHECK: [[VAL31:%.+]] = "tf.ConcatV2"([[VAL28]], [[VAL29]], [[VAL30]], [[VAL7]])
  // CHECK: [[VAL32:%.+]] = "tf.Reshape"(%arg0, [[VAL31]])
  // CHECK: [[VAL33:%.+]] = "tf.GatherV2"([[VAL32]], [[VAL27]], [[VAL0]]) {batch_dims = 0 : i64}
  // CHECK: [[VAL34:%.+]] = "tf.Reshape"([[VAL33]], [[VAL14]])
  // CHECK: return [[VAL34]]
  %resize = "tf.ResizeNearestNeighbor"(%arg0, %shape) {align_corners = false, device = "", half_pixel_centers = false} : (tensor<1x?x?x1xi32>, tensor<2xi32>) -> tensor<1x3x3x1xi32>
  func.return %resize: tensor<1x3x3x1xi32>
}

// CHECK-LABEL: func @imag_resize_nearest_full_dyn
func.func @imag_resize_nearest_full_dyn(%arg0: tensor<1x?x?x1xi32>, %arg1: tensor<2xi32>) -> tensor<1x?x?x1xi32> {

  // CHECK-DAG: [[VAL0:%.+]] = "tf.Const"() {value = dense<1> : tensor<i32>}
  // CHECK-DAG: [[VAL1:%.+]] = "tf.Const"() {value = dense<0.000000e+00> : tensor<f32>}
  // CHECK-DAG: [[VAL2:%.+]] = "tf.Const"() {value = dense<1.000000e+00> : tensor<f32>}
  // CHECK-DAG: [[VAL3:%.+]] = "tf.Const"() {value = dense<1> : tensor<1xi32>}
  // CHECK-DAG: [[VAL4:%.+]] = "tf.Const"() {value = dense<1> : tensor<1xi64>}
  // CHECK-DAG: [[VAL5:%.+]] = "tf.Const"() {value = dense<0> : tensor<i64>}
  // CHECK: [[VAL6:%.+]] = "tf.Shape"(%arg0)
  // CHECK: [[VAL7:%.+]] = "tf.Cast"([[VAL6]])
  // CHECK: [[VAL8:%.+]]:4 = "tf.Unpack"([[VAL7]]) {axis = 0 : i64}
  // CHECK: [[VAL9:%.+]] = "tf.Mul"([[VAL8]]#1, [[VAL8]]#2)
  // CHECK: [[VAL10:%.+]]:2 = "tf.Unpack"(%arg1) {axis = 0 : i64}
  // CHECK: [[VAL11:%.+]] = "tf.Mul"([[VAL10]]#0, [[VAL10]]#1)
  // CHECK: [[VAL12:%.+]] = "tf.ExpandDims"([[VAL8]]#0, [[VAL5]])
  // CHECK: [[VAL13:%.+]] = "tf.ExpandDims"([[VAL10]]#0, [[VAL5]])
  // CHECK: [[VAL14:%.+]] = "tf.ExpandDims"([[VAL10]]#1, [[VAL5]])
  // CHECK: [[VAL15:%.+]] = "tf.ExpandDims"([[VAL8]]#3, [[VAL5]])
  // CHECK: [[VAL16:%.+]] = "tf.ConcatV2"([[VAL12]], [[VAL13]], [[VAL14]], [[VAL15]], [[VAL5]])
  // CHECK: [[VAL17:%.+]] = "tf.Cast"([[VAL8]]#1)
  // CHECK: [[VAL18:%.+]] = "tf.Cast"([[VAL10]]#0)
  // CHECK: [[VAL19:%.+]] = "tf.Div"([[VAL17]], [[VAL18]])
  // CHECK: [[VAL20:%.+]] = "tf.Range"([[VAL1]], [[VAL18]], [[VAL2]])
  // CHECK: [[VAL21:%.+]] = "tf.Mul"([[VAL20]], [[VAL19]])
  // CHECK: [[VAL22:%.+]] = "tf.Cast"([[VAL21]])
  // CHECK: [[VAL23:%.+]] = "tf.ExpandDims"([[VAL10]]#0, [[VAL5]])
  // CHECK: [[VAL24:%.+]] = "tf.ConcatV2"([[VAL23]], [[VAL3]], [[VAL5]])
  // CHECK: [[VAL25:%.+]] = "tf.Reshape"([[VAL22]], [[VAL24]])
  // CHECK: [[VAL26:%.+]] = "tf.Mul"([[VAL25]], [[VAL8]]#2)
  // CHECK: [[VAL27:%.+]] = "tf.Cast"([[VAL8]]#2)
  // CHECK: [[VAL28:%.+]] = "tf.Cast"([[VAL10]]#1)
  // CHECK: [[VAL29:%.+]] = "tf.Div"([[VAL27]], [[VAL28]])
  // CHECK: [[VAL30:%.+]] = "tf.Range"([[VAL1]], [[VAL28]], [[VAL2]])
  // CHECK: [[VAL31:%.+]] = "tf.Mul"([[VAL30]], [[VAL29]])
  // CHECK: [[VAL32:%.+]] = "tf.Cast"([[VAL31]])
  // CHECK: [[VAL33:%.+]] = "tf.ExpandDims"([[VAL10]]#1, [[VAL5]])
  // CHECK: [[VAL34:%.+]] = "tf.ConcatV2"([[VAL3]], [[VAL33]], [[VAL5]])
  // CHECK: [[VAL35:%.+]] = "tf.Reshape"([[VAL32]], [[VAL34]])
  // CHECK: [[VAL36:%.+]] = "tf.AddV2"([[VAL26]], [[VAL35]])
  // CHECK: [[VAL37:%.+]] = "tf.Reshape"([[VAL11]], [[VAL4]])
  // CHECK: [[VAL38:%.+]] = "tf.Reshape"([[VAL36]], [[VAL37]])
  // CHECK: [[VAL39:%.+]] = "tf.ExpandDims"([[VAL8]]#0, [[VAL5]])
  // CHECK: [[VAL40:%.+]] = "tf.ExpandDims"([[VAL9]], [[VAL5]])
  // CHECK: [[VAL41:%.+]] = "tf.ExpandDims"([[VAL8]]#3, [[VAL5]])
  // CHECK: [[VAL42:%.+]] = "tf.ConcatV2"([[VAL39]], [[VAL40]], [[VAL41]], [[VAL5]])
  // CHECK: [[VAL43:%.+]] = "tf.Reshape"(%arg0, [[VAL42]])
  // CHECK: [[VAL44:%.+]] = "tf.GatherV2"([[VAL43]], [[VAL38]], [[VAL0]]) {batch_dims = 0 : i64}
  // CHECK: [[VAL45:%.+]] = "tf.Reshape"([[VAL44]], [[VAL16]])
  // CHECK: return [[VAL45]]
  %resize = "tf.ResizeNearestNeighbor"(%arg0, %arg1) {align_corners = false, device = "", half_pixel_centers = false} : (tensor<1x?x?x1xi32>, tensor<2xi32>) -> tensor<1x?x?x1xi32>
  func.return %resize: tensor<1x?x?x1xi32>
}

// CHECK-LABEL: func @xdivy
// CHECK-SAME: (%[[X:.*]]: tensor<*xf32>, %[[Y:.*]]: tensor<*xf32>)
func.func @xdivy(%lhs: tensor<*xf32>, %rhs: tensor<*xf32>) -> tensor<*xf32> {
  // CHECK:  %[[ZERO:.*]] = "tf.Const"() {value = dense<0.000000e+00> : tensor<f32>} : () -> tensor<f32>
  // CHECK:  %[[IS_ZERO:.*]] = "tf.Equal"(%[[X]], %[[ZERO]]) {incompatible_shape_error = true} : (tensor<*xf32>, tensor<f32>) -> tensor<*xi1>
  // CHECK:  %[[MUL:.*]] = "tf.Div"(%[[X]], %[[Y]]) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
  // CHECK:  %[[RESULT:.*]] = "tf.SelectV2"(%[[IS_ZERO]], %[[X]], %[[MUL]]) : (tensor<*xi1>, tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
  %0 = "tf.Xdivy"(%lhs, %rhs) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
  // CHECK: return %[[RESULT]]
  func.return %0 : tensor<*xf32>
}

// CHECK-LABEL: func @xlog1py
// CHECK-SAME: (%[[X:.*]]: tensor<*xf32>, %[[Y:.*]]: tensor<*xf32>)
func.func @xlog1py(%lhs: tensor<*xf32>, %rhs: tensor<*xf32>) -> tensor<*xf32> {
  // CHECK:  %[[ZERO:.*]] = "tf.Const"() {value = dense<0.000000e+00> : tensor<f32>} : () -> tensor<f32>
  // CHECK:  %[[IS_ZERO:.*]] = "tf.Equal"(%[[X]], %[[ZERO]]) {incompatible_shape_error = true} : (tensor<*xf32>, tensor<f32>) -> tensor<*xi1>
  // CHECK:  %[[LOG:.*]] = "tf.Log1p"(%[[Y]]) : (tensor<*xf32>) -> tensor<*xf32>
  // CHECK:  %[[MUL:.*]] = "tf.Mul"(%[[X]], %[[LOG]]) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
  // CHECK:  %[[RESULT:.*]] = "tf.SelectV2"(%[[IS_ZERO]], %[[X]], %[[MUL]]) : (tensor<*xi1>, tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
  %0 = "tf.Xlog1py"(%lhs, %rhs) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
  // CHECK: return %[[RESULT]]
  func.return %0 : tensor<*xf32>
}

// CHECK-LABEL: func @xlogy
// CHECK-SAME: (%[[X:.*]]: tensor<*xf32>, %[[Y:.*]]: tensor<*xf32>)
func.func @xlogy(%lhs: tensor<*xf32>, %rhs: tensor<*xf32>) -> tensor<*xf32> {
  // CHECK:  %[[ZERO:.*]] = "tf.Const"() {value = dense<0.000000e+00> : tensor<f32>} : () -> tensor<f32>
  // CHECK:  %[[IS_ZERO:.*]] = "tf.Equal"(%[[X]], %[[ZERO]]) {incompatible_shape_error = true} : (tensor<*xf32>, tensor<f32>) -> tensor<*xi1>
  // CHECK:  %[[LOG:.*]] = "tf.Log"(%[[Y]]) : (tensor<*xf32>) -> tensor<*xf32>
  // CHECK:  %[[MUL:.*]] = "tf.Mul"(%[[X]], %[[LOG]]) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
  // CHECK:  %[[RESULT:.*]] = "tf.SelectV2"(%[[IS_ZERO]], %[[X]], %[[MUL]]) : (tensor<*xi1>, tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
  %0 = "tf.Xlogy"(%lhs, %rhs) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
  // CHECK: return %[[RESULT]]
  func.return %0 : tensor<*xf32>
}

// CHECK-LABEL: size_to_prod_shape_i32
func.func @size_to_prod_shape_i32(%arg0 : tensor<1x?x2x3xf32>) -> tensor<i32> {
  %0 = "tf.Size"(%arg0) : (tensor<1x?x2x3xf32>) -> tensor<i32>
  func.return %0 : tensor<i32>
  // CHECK: %[[CONSTANT:.*]] = "tf.Const"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
  // CHECK: %[[SHAPE:.*]] = "tf.Shape"(%arg0) : (tensor<1x?x2x3xf32>) -> tensor<4xi32>
  // CHECK: %[[PROD:.*]] = "tf.Prod"(%[[SHAPE]], %[[CONSTANT]]) {keep_dims = false} : (tensor<4xi32>, tensor<i32>) -> tensor<i32>
  // CHECK: return %[[PROD]]
}

// CHECK-LABEL: size_to_prod_shape_i64
func.func @size_to_prod_shape_i64(%arg0 : tensor<1x?x2x3xf32>) -> tensor<i64> {
  %0 = "tf.Size"(%arg0) : (tensor<1x?x2x3xf32>) -> tensor<i64>
  func.return %0 : tensor<i64>
  // CHECK: %[[CONSTANT:.*]] = "tf.Const"() {value = dense<0> : tensor<i64>} : () -> tensor<i64>
  // CHECK: %[[SHAPE:.*]] = "tf.Shape"(%arg0) : (tensor<1x?x2x3xf32>) -> tensor<4xi64>
  // CHECK: %[[PROD:.*]] = "tf.Prod"(%[[SHAPE]], %[[CONSTANT]]) {keep_dims = false} : (tensor<4xi64>, tensor<i64>) -> tensor<i64>
  // CHECK: return %[[PROD]]
}

// CHECK-LABEL: @is_finite
func.func @is_finite(%arg0: tensor<3x4xf32>) -> tensor<3x4xi1> {
  %0 = "tf.IsFinite"(%arg0) : (tensor<3x4xf32>) -> tensor<3x4xi1>
  func.return %0 : tensor<3x4xi1>
  // CHECK: %[[ZERO:.*]] = "tf.Const"() {value = dense<0.000000e+00> : tensor<f32>} : () -> tensor<f32>
  // CHECK: %[[SUB:.*]] = "tf.Sub"(%arg0, %arg0) : (tensor<3x4xf32>, tensor<3x4xf32>) -> tensor<3x4xf32>
  // CHECK: %[[RESULT:.*]] = "tf.Equal"(%[[SUB]], %[[ZERO]]) {incompatible_shape_error = true} : (tensor<3x4xf32>, tensor<f32>) -> tensor<3x4xi1>
  // CHECK: return %[[RESULT]]
}

// CHECK-LABEL: @is_finite_dynamic
func.func @is_finite_dynamic(%arg0: tensor<?x4xf32>) -> tensor<?x4xi1> {
  %0 = "tf.IsFinite"(%arg0) : (tensor<?x4xf32>) -> tensor<?x4xi1>
  func.return %0 : tensor<?x4xi1>
  // CHECK: %[[ZERO:.*]] = "tf.Const"() {value = dense<0.000000e+00> : tensor<f32>} : () -> tensor<f32>
  // CHECK: %[[SUB:.*]] = "tf.Sub"(%arg0, %arg0) : (tensor<?x4xf32>, tensor<?x4xf32>) -> tensor<?x4xf32>
  // CHECK: %[[RESULT:.*]] = "tf.Equal"(%[[SUB]], %[[ZERO]]) {incompatible_shape_error = true} : (tensor<?x4xf32>, tensor<f32>) -> tensor<?x4xi1>
  // CHECK: return %[[RESULT]]
}

func.func @roll_scalar_axis(%arg0: tensor<3x8x4xi32>) -> tensor<3x8x4xi32> {
  %axis = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
  %shift = "tf.Const"() {value = dense<2> : tensor<i32>} : () -> tensor<i32>
  %0 = "tf.Roll"(%arg0, %shift, %axis) : (tensor<3x8x4xi32>, tensor<i32>, tensor<i32>) -> tensor<3x8x4xi32>
  func.return %0 : tensor<3x8x4xi32>
  // CHECK-LABEL: roll_scalar_axis
  // CHECK-DAG:  %[[CST:.*]] = "tf.Const"() {value = dense<[0, 6, 0]> : tensor<3xi64>} : () -> tensor<3xi64>
  // CHECK-DAG:  %[[CST0:.*]] = "tf.Const"() {value = dense<[3, 2, 4]> : tensor<3xi64>} : () -> tensor<3xi64>
  // CHECK-DAG:  %[[CST1:.*]] = "tf.Const"() {value = dense<0> : tensor<3xi64>} : () -> tensor<3xi64>
  // CHECK-DAG:  %[[CST2:.*]] = "tf.Const"() {value = dense<[3, 6, 4]> : tensor<3xi64>} : () -> tensor<3xi64>
  // CHECK-DAG:  %[[CST3:.*]] = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
  // CHECK:  %[[SLICE:.*]] = "tf.Slice"(%arg0, %[[CST]], %[[CST0]]) : (tensor<3x8x4xi32>, tensor<3xi64>, tensor<3xi64>) -> tensor<3x2x4xi32>
  // CHECK:  %[[SLICE1:.*]] = "tf.Slice"(%arg0, %[[CST1]], %[[CST2]]) : (tensor<3x8x4xi32>, tensor<3xi64>, tensor<3xi64>) -> tensor<3x6x4xi32>
  // CHECK:  %[[CONCAT:.*]] = "tf.ConcatV2"(%[[SLICE]], %[[SLICE1]], %[[CST3]]) : (tensor<3x2x4xi32>, tensor<3x6x4xi32>, tensor<i32>) -> tensor<3x8x4xi32>
  // CHECK:  return %[[CONCAT]] : tensor<3x8x4xi32>
}

func.func @roll_1d_axis(%arg0: tensor<3x8x4xi32>) -> tensor<3x8x4xi32> {
  %axis = "tf.Const"() {value = dense<[1]> : tensor<1xi32>} : () -> tensor<1xi32>
  %shift = "tf.Const"() {value = dense<[2]> : tensor<1xi32>} : () -> tensor<1xi32>
  %0 = "tf.Roll"(%arg0, %shift, %axis) : (tensor<3x8x4xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<3x8x4xi32>
  func.return %0 : tensor<3x8x4xi32>
  // CHECK-LABEL: roll_1d_axis
  // CHECK-DAG:  %[[CST:.*]] = "tf.Const"() {value = dense<[0, 6, 0]> : tensor<3xi64>} : () -> tensor<3xi64>
  // CHECK-DAG:  %[[CST0:.*]] = "tf.Const"() {value = dense<[3, 2, 4]> : tensor<3xi64>} : () -> tensor<3xi64>
  // CHECK-DAG:  %[[CST1:.*]] = "tf.Const"() {value = dense<0> : tensor<3xi64>} : () -> tensor<3xi64>
  // CHECK-DAG:  %[[CST2:.*]] = "tf.Const"() {value = dense<[3, 6, 4]> : tensor<3xi64>} : () -> tensor<3xi64>
  // CHECK-DAG:  %[[CST3:.*]] = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
  // CHECK:  %[[SLICE:.*]] = "tf.Slice"(%arg0, %[[CST]], %[[CST0]]) : (tensor<3x8x4xi32>, tensor<3xi64>, tensor<3xi64>) -> tensor<3x2x4xi32>
  // CHECK:  %[[SLICE1:.*]] = "tf.Slice"(%arg0, %[[CST1]], %[[CST2]]) : (tensor<3x8x4xi32>, tensor<3xi64>, tensor<3xi64>) -> tensor<3x6x4xi32>
  // CHECK:  %[[CONCAT:.*]] = "tf.ConcatV2"(%[[SLICE]], %[[SLICE1]], %[[CST3]]) : (tensor<3x2x4xi32>, tensor<3x6x4xi32>, tensor<i32>) -> tensor<3x8x4xi32>
  // CHECK:  return %[[CONCAT]] : tensor<3x8x4xi32>
}

func.func @roll_multiple_axis(%arg0: tensor<3x8x4xi32>) -> tensor<3x8x4xi32> {
  %axis = "tf.Const"() {value = dense<[0, 1]> : tensor<2xi32>} : () -> tensor<2xi32>
  %shift = "tf.Const"() {value = dense<[2, -6]> : tensor<2xi32>} : () -> tensor<2xi32>
  %0 = "tf.Roll"(%arg0, %shift, %axis) : (tensor<3x8x4xi32>, tensor<2xi32>, tensor<2xi32>) -> tensor<3x8x4xi32>
  func.return %0 : tensor<3x8x4xi32>
  // CHECK-LABEL: roll_multiple_axis
  // CHECK-DAG:  %[[CST:.*]] = "tf.Const"() {value = dense<[1, 0, 0]> : tensor<3xi64>} : () -> tensor<3xi64>
  // CHECK-DAG:  %[[CST0:.*]] = "tf.Const"() {value = dense<[2, 8, 4]> : tensor<3xi64>} : () -> tensor<3xi64>
  // CHECK-DAG:  %[[CST1:.*]] = "tf.Const"() {value = dense<[1, 8, 4]> : tensor<3xi64>} : () -> tensor<3xi64>
  // CHECK-DAG:  %[[CST2:.*]] = "tf.Const"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
  // CHECK-DAG:  %[[CST3:.*]] = "tf.Const"() {value = dense<[0, 6, 0]> : tensor<3xi64>} : () -> tensor<3xi64>
  // CHECK-DAG:  %[[CST4:.*]] = "tf.Const"() {value = dense<[3, 2, 4]> : tensor<3xi64>} : () -> tensor<3xi64>
  // CHECK-DAG:  %[[CST5:.*]] = "tf.Const"() {value = dense<0> : tensor<3xi64>} : () -> tensor<3xi64>
  // CHECK-DAG:  %[[CST6:.*]] = "tf.Const"() {value = dense<[3, 6, 4]> : tensor<3xi64>} : () -> tensor<3xi64>
  // CHECK-DAG:  %[[CST7:.*]] = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
  // CHECK:      %[[SLICE:.*]] = "tf.Slice"(%arg0, %[[CST]], %[[CST0]]) : (tensor<3x8x4xi32>, tensor<3xi64>, tensor<3xi64>) -> tensor<2x8x4xi32>
  // CHECK:      %[[SLICE1:.*]] = "tf.Slice"(%arg0, %[[CST5]], %[[CST1]]) : (tensor<3x8x4xi32>, tensor<3xi64>, tensor<3xi64>) -> tensor<1x8x4xi32>
  // CHECK:      %[[CONCAT:.*]] = "tf.ConcatV2"(%[[SLICE]], %[[SLICE1]], %[[CST2]]) : (tensor<2x8x4xi32>, tensor<1x8x4xi32>, tensor<i32>) -> tensor<3x8x4xi32>
  // CHECK:      %[[SLICE2:.*]] = "tf.Slice"(%[[CONCAT]], %[[CST3]], %[[CST4]]) : (tensor<3x8x4xi32>, tensor<3xi64>, tensor<3xi64>) -> tensor<3x2x4xi32>
  // CHECK:      %[[SLICE3:.*]] = "tf.Slice"(%[[CONCAT]], %[[CST5]], %[[CST6]]) : (tensor<3x8x4xi32>, tensor<3xi64>, tensor<3xi64>) -> tensor<3x6x4xi32>
  // CHECK:      %[[CONCAT1:.*]] = "tf.ConcatV2"(%[[SLICE2]], %[[SLICE3]], %[[CST7]]) : (tensor<3x2x4xi32>, tensor<3x6x4xi32>, tensor<i32>) -> tensor<3x8x4xi32>
  // CHECK:      return %[[CONCAT1]] : tensor<3x8x4xi32>
}

func.func @roll_dynamic_shape(%arg0: tensor<?x8x4xi32>) -> tensor<?x8x4xi32> {
  %axis = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
  %shift = "tf.Const"() {value = dense<2> : tensor<i32>} : () -> tensor<i32>
  %0 = "tf.Roll"(%arg0, %shift, %axis) : (tensor<?x8x4xi32>, tensor<i32>, tensor<i32>) -> tensor<?x8x4xi32>
  func.return %0 : tensor<?x8x4xi32>
  // CHECK-LABEL: roll_dynamic_shape
  // CHECK:  "tf.Roll"
}

func.func @roll_non_constant_axis(%arg0: tensor<3x8x4xi32>, %arg1: tensor<i32>) -> tensor<3x8x4xi32> {
  %shift = "tf.Const"() {value = dense<2> : tensor<i32>} : () -> tensor<i32>
  %0 = "tf.Roll"(%arg0, %shift, %arg1) : (tensor<3x8x4xi32>, tensor<i32>, tensor<i32>) -> tensor<3x8x4xi32>
  func.return %0 : tensor<3x8x4xi32>
  // CHECK-LABEL: roll_non_constant_axis
  // CHECK:  "tf.Roll"
}

func.func @roll_non_constant_shift(%arg0: tensor<3x8x4xi32>, %arg1: tensor<i32>) -> tensor<3x8x4xi32> {
  %axis = "tf.Const"() {value = dense<2> : tensor<i32>} : () -> tensor<i32>
  %0 = "tf.Roll"(%arg0, %arg1, %axis) : (tensor<3x8x4xi32>, tensor<i32>, tensor<i32>) -> tensor<3x8x4xi32>
  func.return %0 : tensor<3x8x4xi32>
  // CHECK-LABEL: roll_non_constant_shift
  // CHECK:  "tf.Roll"
}

func.func @scatter_nd_updates(%arg0: tensor<14xf32>, %arg1: tensor<1x1xi32>, %arg2: tensor<1xf32>) -> tensor<14xf32> {
  %0 = "tf.TensorScatterUpdate"(%arg0, %arg1, %arg2) : (tensor<14xf32>, tensor<1x1xi32>, tensor<1xf32>) -> tensor<14xf32>
  func.return %0 : tensor<14xf32>

  // CHECK-LABEL: scatter_nd_updates
  // CHECK-DAG: %[[CST:.*]] = "tf.Const"() {value = dense<1.000000e+00> : tensor<f32>} : () -> tensor<f32>
  // CHECK-DAG: %[[CST0:.*]] = "tf.Const"() {value = dense<1.000000e+00> : tensor<1xf32>} : () -> tensor<1xf32>
  // CHECK-DAG: %[[CST1:.*]] = "tf.Const"() {value = dense<0.000000e+00> : tensor<14xf32>} : () -> tensor<14xf32>
  // CHECK: %[[SCATTER:.*]] = "tf.TensorScatterAdd"(%cst_1, %arg1, %[[CST0]]) : (tensor<14xf32>, tensor<1x1xi32>, tensor<1xf32>) -> tensor<14xf32>
  // CHECK: %[[SUB:.*]] = "tf.Sub"(%[[CST]], %[[SCATTER]]) : (tensor<f32>, tensor<14xf32>) -> tensor<14xf32>
  // CHECK: %[[MUL:.*]] = "tf.Mul"(%[[SUB]], %arg0) : (tensor<14xf32>, tensor<14xf32>) -> tensor<14xf32>
  // CHECK: %[[SCATTER1:.*]] = "tf.TensorScatterAdd"(%[[CST1]], %arg1, %arg2) : (tensor<14xf32>, tensor<1x1xi32>, tensor<1xf32>) -> tensor<14xf32>
  // CHECK: %[[ADD:.*]] = "tf.AddV2"(%[[MUL]], %[[SCATTER1]]) : (tensor<14xf32>, tensor<14xf32>) -> tensor<14xf32>
  // CHECK: return %[[ADD]] : tensor<14xf32>
}

func.func @scatter_nd_updates_bool(%arg0: tensor<1x24xi1>, %arg1: tensor<1x2x2xi32>, %arg2: tensor<1x2xi1>) -> tensor<1x24xi1> {
  %0 = "tf.TensorScatterUpdate"(%arg0, %arg1, %arg2) : (tensor<1x24xi1>, tensor<1x2x2xi32>, tensor<1x2xi1>) -> tensor<1x24xi1>
  func.return %0 : tensor<1x24xi1>

// CHECK-LABEL: scatter_nd_updates_bool(
// CHECK-DAG:       %[[CST:.*]] = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
// CHECK-DAG:       %[[CST0:.*]] = "tf.Const"() {value = dense<1> : tensor<1x2xi32>} : () -> tensor<1x2xi32>
// CHECK-DAG:       %[[CST1:.*]] = "tf.Const"() {value = dense<0> : tensor<1x24xi32>} : () -> tensor<1x24xi32>
// CHECK:           %[[CAST0:.*]] = "tf.Cast"(%arg0) {Truncate = false} : (tensor<1x24xi1>) -> tensor<1x24xi32>
// CHECK:           %[[CAST1:.*]] = "tf.Cast"(%arg2) {Truncate = false} : (tensor<1x2xi1>) -> tensor<1x2xi32>
// CHECK:           %[[SCATTER:.*]] = "tf.TensorScatterAdd"(%[[CST1]], %arg1, %[[CST0]]) : (tensor<1x24xi32>, tensor<1x2x2xi32>, tensor<1x2xi32>) -> tensor<1x24xi32>
// CHECK:           %[[SUB:.*]] = "tf.Sub"(%[[CST]], %[[SCATTER]]) : (tensor<i32>, tensor<1x24xi32>) -> tensor<1x24xi32>
// CHECK:           %[[MUL:.*]] = "tf.Mul"(%[[SUB]], %[[CAST0]]) : (tensor<1x24xi32>, tensor<1x24xi32>) -> tensor<1x24xi32>
// CHECK:           %[[SCATTER1:.*]] = "tf.TensorScatterAdd"(%[[CST1]], %arg1, %[[CAST1]]) : (tensor<1x24xi32>, tensor<1x2x2xi32>, tensor<1x2xi32>) -> tensor<1x24xi32>
// CHECK:           %[[ADD:.*]] = "tf.AddV2"(%[[MUL]], %[[SCATTER1]]) : (tensor<1x24xi32>, tensor<1x24xi32>) -> tensor<1x24xi32>
// CHECK:           %[[CAST2:.*]] = "tf.Cast"(%[[ADD]]) {Truncate = false} : (tensor<1x24xi32>) -> tensor<1x24xi1>
// CHECK:           return %[[CAST2]] : tensor<1x24xi1>
}

//===----------------------------------------------------------------------===//
// Softmax op legalizations.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @simple_softmax
// CHECK-SAME: (%[[ARG0:.*]]: tensor<2x3xf32>)
func.func @simple_softmax(%arg0: tensor<2x3xf32>) -> tensor<2x3xf32> {
  // CHECK-DAG: %[[AXIS:.*]] = "tf.Const"() {value = dense<-1> : tensor<1xi64>} : () -> tensor<1xi64>
  // CHECK-DAG: %[[MAX:.*]] = "tf.Max"(%[[ARG0]], %[[AXIS]]) {keep_dims = true} : (tensor<2x3xf32>, tensor<1xi64>) -> tensor<2x1xf32>
  // CHECK-DAG: %[[SHIFTED:.*]] = "tf.Sub"(%[[ARG0]], %[[MAX]]) : (tensor<2x3xf32>, tensor<2x1xf32>) -> tensor<2x3xf32>
  // CHECK-DAG: %[[EXP:.*]] = "tf.Exp"(%[[SHIFTED]]) : (tensor<2x3xf32>) -> tensor<2x3xf32>
  // CHECK-DAG: %[[SUM:.*]] = "tf.Sum"(%[[EXP]], %[[AXIS]]) {keep_dims = true} : (tensor<2x3xf32>, tensor<1xi64>) -> tensor<2x1xf32>
  // CHECK-DAG: %[[RESULT:.*]] = "tf.Div"(%[[EXP]], %[[SUM]]) : (tensor<2x3xf32>, tensor<2x1xf32>) -> tensor<2x3xf32>
  // CHECK: return %[[RESULT]]
  %0 = "tf.Softmax"(%arg0) : (tensor<2x3xf32>) -> tensor<2x3xf32>
  func.return %0: tensor<2x3xf32>
}

// Verify intermediate and final shape are correct with dynamic shapes.
// CHECK-LABEL: func @unranked_softmax
func.func @unranked_softmax(%arg0: tensor<*xf32>) -> tensor<*xf32> {
  // CHECK-NOT: "tf.Softmax"
  %0 = "tf.Softmax"(%arg0) : (tensor<*xf32>) -> tensor<*xf32>
  func.return %0: tensor<*xf32>
}

//===----------------------------------------------------------------------===//
// LogSoftmax op legalizations.
// This just changes the tail of the regular Softmax legalization
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @simple_logsoftmax
// CHECK-SAME: (%[[ARG0:.*]]: tensor<2x3xf32>)
func.func @simple_logsoftmax(%arg0: tensor<2x3xf32>) -> tensor<2x3xf32> {
  // CHECK-DAG: %[[AXIS:.*]] = "tf.Const"() {value = dense<-1> : tensor<1xi64>} : () -> tensor<1xi64>
  // CHECK-DAG: %[[MAX:.*]] = "tf.Max"(%[[ARG0]], %[[AXIS]]) {keep_dims = true} : (tensor<2x3xf32>, tensor<1xi64>) -> tensor<2x1xf32>
  // CHECK-DAG: %[[SHIFTED:.*]] = "tf.Sub"(%[[ARG0]], %[[MAX]]) : (tensor<2x3xf32>, tensor<2x1xf32>) -> tensor<2x3xf32>
  // CHECK-DAG: %[[EXP:.*]] = "tf.Exp"(%[[SHIFTED]]) : (tensor<2x3xf32>) -> tensor<2x3xf32>
  // CHECK-DAG: %[[SUM:.*]] = "tf.Sum"(%[[EXP]], %[[AXIS]]) {keep_dims = true} : (tensor<2x3xf32>, tensor<1xi64>) -> tensor<2x1xf32>
  // CHECK-DAG: %[[LOG:.*]] = "tf.Log"(%[[SUM]]) : (tensor<2x1xf32>) -> tensor<2x1xf32>
  // CHECK-DAG: %[[RESULT:.*]] = "tf.Sub"(%[[SHIFTED]], %[[LOG]]) : (tensor<2x3xf32>, tensor<2x1xf32>) -> tensor<2x3xf32>
  // CHECK: return %[[RESULT]]
  %0 = "tf.LogSoftmax"(%arg0) : (tensor<2x3xf32>) -> tensor<2x3xf32>
  func.return %0: tensor<2x3xf32>
}

// CHECK-LABEL: func @unranked_logsoftmax
func.func @unranked_logsoftmax(%arg0: tensor<*xf32>) -> tensor<*xf32> {
  // CHECK-NOT: "tf.LogSoftmax"
  %0 = "tf.LogSoftmax"(%arg0) : (tensor<*xf32>) -> tensor<*xf32>
  func.return %0: tensor<*xf32>
}

// CHECK-LABEL: func @selu
// CHECK-SAME:  (%[[FEATURES:.*]]: tensor<1x4x4x3xf32>) -> tensor<1x4x4x3xf32> {
func.func @selu(%arg0: tensor<1x4x4x3xf32>) -> tensor<1x4x4x3xf32> {
    // CHECK-DAG:   %[[ZERO:.*]] = "tf.Const"() {value = dense<0.000000e+00> : tensor<f32>} : () -> tensor<f32>
    // CHECK-DAG:   %[[SCALE:.*]] = "tf.Const"() {value = dense<1.05070102> : tensor<f32>} : () -> tensor<f32>
    // CHECK-DAG:   %[[SCALED_ALPHA:.*]] = "tf.Const"() {value = dense<1.75809932> : tensor<f32>} : () -> tensor<f32>
    // CHECK-NEXT:  %[[ONE:.*]] = "tf.Const"() {value = dense<1.000000e+00> : tensor<f32>} : () -> tensor<f32>
    // CHECK-DAG:   %[[PRED:.*]] = "tf.Greater"(%[[FEATURES]], %[[ZERO]]) : (tensor<1x4x4x3xf32>, tensor<f32>) -> tensor<1x4x4x3xi1>
    // CHECK-NEXT:  %[[SCALED_FEATURES:.*]] = "tf.Mul"(%[[FEATURES]], %[[SCALE]]) : (tensor<1x4x4x3xf32>, tensor<f32>) -> tensor<1x4x4x3xf32>
    // CHECK-NEXT:  %[[EXP:.*]] = "tf.Exp"(%[[FEATURES]]) : (tensor<1x4x4x3xf32>) -> tensor<1x4x4x3xf32>
    // CHECK-NEXT:  %[[ELU_VAL:.*]] = "tf.Sub"(%[[EXP]], %[[ONE]]) : (tensor<1x4x4x3xf32>, tensor<f32>) -> tensor<1x4x4x3xf32>
    // CHECK-NEXT:  %[[SELU_VAL:.*]] = "tf.Mul"(%[[ELU_VAL]], %[[SCALED_ALPHA]]) : (tensor<1x4x4x3xf32>, tensor<f32>) -> tensor<1x4x4x3xf32>
    // CHECK-NEXT:  %[[RES:.*]] = "tf.SelectV2"(%[[PRED]], %[[SCALED_FEATURES]], %[[SELU_VAL]]) : (tensor<1x4x4x3xi1>, tensor<1x4x4x3xf32>, tensor<1x4x4x3xf32>) -> tensor<1x4x4x3xf32>
    // CHECK-NEXT:  return %[[RES]] : tensor<1x4x4x3xf32>
    %0 = "tf.Selu"(%arg0) : (tensor<1x4x4x3xf32>) -> tensor<1x4x4x3xf32>
    func.return %0 : tensor<1x4x4x3xf32>
}

// CHECK-LABEL: func @selu_grad
// CHECK-SAME: (%[[GRADIENTS:.*]]: tensor<4x8xf32>, %[[FEATURES:.*]]: tensor<4x8xf32>) -> tensor<4x8xf32> {
func.func @selu_grad(%gradients: tensor<4x8xf32>, %features: tensor<4x8xf32>) -> tensor<4x8xf32> {
    // CHECK-DAG:   %[[ZERO:.*]] = "tf.Const"() {value = dense<0.000000e+00> : tensor<f32>} : () -> tensor<f32>
    // CHECK-DAG:   %[[SCALE:.*]] = "tf.Const"() {value = dense<1.05070102> : tensor<f32>} : () -> tensor<f32>
    // CHECK-DAG:   %[[SCALED_ALPHA:.*]] = "tf.Const"() {value = dense<1.75809932> : tensor<f32>} : () -> tensor<f32>
    // CHECK-DAG:   %[[PRED:.*]] = "tf.Greater"(%[[FEATURES]], %[[ZERO]]) : (tensor<4x8xf32>, tensor<f32>) -> tensor<4x8xi1>
    // CHECK-NEXT:  %[[SCALED_GRADIENTS:.*]] = "tf.Mul"(%[[GRADIENTS]], %[[SCALE]]) : (tensor<4x8xf32>, tensor<f32>) -> tensor<4x8xf32>
    // CHECK-NEXT:  %[[FEATURES_PLUS_SCALED_ALPHA:.*]] = "tf.AddV2"(%[[FEATURES]], %[[SCALED_ALPHA]]) : (tensor<4x8xf32>, tensor<f32>) -> tensor<4x8xf32>
    // CHECK-NEXT:  %[[SELU_GRAD_VALUE:.*]] = "tf.Mul"(%[[GRADIENTS]], %[[FEATURES_PLUS_SCALED_ALPHA]]) : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
    // CHECK-NEXT:  %[[RES:.*]] = "tf.SelectV2"(%[[PRED]], %[[SCALED_GRADIENTS]], %[[SELU_GRAD_VALUE]]) : (tensor<4x8xi1>, tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
    // CHECK-NEXT:  return %[[RES]] : tensor<4x8xf32>
    %2 = "tf.SeluGrad"(%gradients, %features) : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
    func.return %2 : tensor<4x8xf32>
}

// CHECK-LABEL: func @expm1
// CHECK-SAME: (%[[ARG0:.*]]: tensor<3x4xf32>)
func.func @expm1(%arg0: tensor<3x4xf32>) -> tensor<3x4xf32> {
  %0 = "tf.Expm1"(%arg0) : (tensor<3x4xf32>) -> tensor<3x4xf32>
  func.return %0 : tensor<3x4xf32>
  // CHECK: %[[ONE:.*]] = "tf.Const"() {value = dense<1.000000e+00> : tensor<f32>} : () -> tensor<f32>
  // CHECK: %[[EXP:.*]] = "tf.Exp"(%[[ARG0]]) : (tensor<3x4xf32>) -> tensor<3x4xf32>
  // CHECK: %[[RESULT:.*]] = "tf.Sub"(%[[EXP]], %[[ONE]]) : (tensor<3x4xf32>, tensor<f32>) -> tensor<3x4xf32>
  // CHECK: return %[[RESULT]]
}

// CHECK-LABEL: func @matrix_band_part
// CHECK-SAME: (%[[INPUT:.*]]: tensor<4x5xf32>, %[[NUM_LOWER:.*]]: tensor<i64>, %[[NUM_UPPER:.*]]: tensor<i64>) -> tensor<4x5xf32> {
func.func @matrix_band_part(%input: tensor<4x5xf32>, %num_lower: tensor<i64>, %num_upper: tensor<i64>) -> tensor<4x5xf32> {
  // CHECK-DAG: %[[ZERO:.*]] = "tf.Const"() {value = dense<0> : tensor<i64>} : () -> tensor<i64>
  // CHECK-DAG: %[[OFFSET:.*]] = "tf.Const"() {{.+}} : () -> tensor<4x5xi64>
  // CHECK-DAG: %[[M:.*]] = "tf.Const"() {value = dense<4> : tensor<i64>} : () -> tensor<i64>
  // CHECK-DAG: %[[N:.*]] = "tf.Const"() {value = dense<5> : tensor<i64>} : () -> tensor<i64>
  // CHECK-DAG: %[[ZEROS_LIKE:.*]] = "tf.Const"() {value = dense<0.000000e+00> : tensor<4x5xf32>} : () -> tensor<4x5xf32>
  // CHECK-DAG: %[[LE:.*]] = "tf.Less"(%[[NUM_LOWER]], %[[ZERO]]) : (tensor<i64>, tensor<i64>) -> tensor<i1>
  // CHECK-DAG: %[[NUM_LOWER_OR_M:.*]] = "tf.SelectV2"(%[[LE]], %[[M]], %[[NUM_LOWER]]) : (tensor<i1>, tensor<i64>, tensor<i64>) -> tensor<i64>
  // CHECK-DAG: %[[LE1:.*]] = "tf.Less"(%[[NUM_UPPER]], %[[ZERO]]) : (tensor<i64>, tensor<i64>) -> tensor<i1>
  // CHECK-DAG: %[[NUM_UPPER_OR_N:.*]] = "tf.SelectV2"(%[[LE1]], %[[N]], %[[NUM_UPPER]]) : (tensor<i1>, tensor<i64>, tensor<i64>) -> tensor<i64>
  // CHECK-DAG: %[[LE2:.*]] = "tf.LessEqual"(%[[OFFSET]], %[[NUM_LOWER_OR_M]]) : (tensor<4x5xi64>, tensor<i64>) -> tensor<4x5xi1>
  // CHECK-DAG: %[[NEG:.*]] = "tf.Neg"(%[[NUM_UPPER_OR_N]]) : (tensor<i64>) -> tensor<i64>
  // CHECK-DAG: %[[GE:.*]] = "tf.GreaterEqual"(%[[OFFSET]], %[[NEG]]) : (tensor<4x5xi64>, tensor<i64>) -> tensor<4x5xi1>
  // CHECK-DAG: %[[INDICATOR:.*]] = "tf.LogicalAnd"(%[[LE2]], %[[GE]]) : (tensor<4x5xi1>, tensor<4x5xi1>) -> tensor<4x5xi1>
  // CHECK-DAG: %[[RET:.*]] = "tf.SelectV2"(%[[INDICATOR]], %[[INPUT]], %[[ZEROS_LIKE]]) : (tensor<4x5xi1>, tensor<4x5xf32>, tensor<4x5xf32>) -> tensor<4x5xf32>
  // CHECK-DAG: return %[[RET]]
  %0 = "tf.MatrixBandPart"(%input, %num_lower, %num_upper) : (tensor<4x5xf32>, tensor<i64>, tensor<i64>) -> tensor<4x5xf32>
  func.return %0 : tensor<4x5xf32>
}

// CHECK-LABEL: func @rank3_matrix_band_part
func.func @rank3_matrix_band_part(%input: tensor<?x4x5xf32>, %num_lower: tensor<i64>, %num_upper: tensor<i64>) -> tensor<?x4x5xf32> {
  // CHECK-NOT: tf.MatrixBandPart
  %0 = "tf.MatrixBandPart"(%input, %num_lower, %num_upper) : (tensor<?x4x5xf32>, tensor<i64>, tensor<i64>) -> tensor<?x4x5xf32>
  func.return %0 : tensor<?x4x5xf32>
}

// CHECK-LABEL: func @dynamic_shape_matrix_band_part
// CHECK-SAME: (%[[INPUT:.*]]: tensor<?x?xf32>, %[[NUM_LOWER:.*]]: tensor<i32>, %[[NUM_UPPER:.*]]: tensor<i32>) -> tensor<?x?xf32> {
func.func @dynamic_shape_matrix_band_part(%input: tensor<?x?xf32>, %num_lower: tensor<i32>, %num_upper: tensor<i32>) -> tensor<?x?xf32> {
  // CHECK-DAG: %[[ZERO:.*]] = "tf.Const"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
  // CHECK-DAG: %[[ONE:.*]] = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
  // CHECK-DAG: %[[NEG_ONE:.*]] = "tf.Const"() {value = dense<-1> : tensor<i32>} : () -> tensor<i32>
  // CHECK-DAG: %[[ZERO_1D:.*]] = "tf.Const"() {value = dense<0> : tensor<1xi32>} : () -> tensor<1xi32>
  // CHECK-DAG: %[[ONE_1D:.*]] = "tf.Const"() {value = dense<1> : tensor<1xi32>} : () -> tensor<1xi32>
  // CHECK-DAG: %[[TWO_1D:.*]] = "tf.Const"() {value = dense<2> : tensor<1xi32>} : () -> tensor<1xi32>
  // CHECK-DAG: %[[ZERO_F32:.*]] = "tf.Const"() {value = dense<0.000000e+00> : tensor<f32>} : () -> tensor<f32>
  // CHECK-DAG: %[[SHAPE:.*]] = "tf.Shape"(%[[INPUT]]) : (tensor<?x?xf32>) -> tensor<2xi32>
  // CHECK-DAG: %[[M:.*]] = "tf.StridedSlice"(%[[SHAPE]], %[[ZERO_1D]], %[[ONE_1D]], %[[ONE_1D]]) {begin_mask = 0 : i64, ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<2xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i32>
  // CHECK-DAG: %[[SHAPE1:.*]] = "tf.Shape"(%[[INPUT]]) : (tensor<?x?xf32>) -> tensor<2xi32>
  // CHECK-DAG: %[[N:.*]] = "tf.StridedSlice"(%[[SHAPE1]], %[[ONE_1D]], %[[TWO_1D]], %[[ONE_1D]]) {begin_mask = 0 : i64, ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 1 : i64} : (tensor<2xi32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<i32>
  // CHECK-DAG: %[[LE:.*]] = "tf.Less"(%[[NUM_LOWER]], %[[ZERO]]) : (tensor<i32>, tensor<i32>) -> tensor<i1>
  // CHECK-DAG: %[[NUM_LOWER_OR_M:.*]] = "tf.SelectV2"(%[[LE]], %[[M]], %[[NUM_LOWER]]) : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
  // CHECK-DAG: %[[LE1:.*]] = "tf.Less"(%[[NUM_UPPER]], %[[ZERO]]) : (tensor<i32>, tensor<i32>) -> tensor<i1>
  // CHECK-DAG: %[[NUM_UPPER_OR_N:.*]] = "tf.SelectV2"(%[[LE1]], %[[N]], %[[NUM_UPPER]]) : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
  // CHECK-DAG: %[[RANGE_M:.*]] = "tf.Range"(%[[ZERO]], %[[M]], %[[ONE]]) : (tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<?xi32>
  // CHECK-DAG: %[[RANGE_N:.*]] = "tf.Range"(%[[ZERO]], %[[N]], %[[ONE]]) : (tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<?xi32>
  // CHECK-DAG: %[[EXPAND_DIMS:.*]] = "tf.ExpandDims"(%[[RANGE_M]], %[[NEG_ONE]]) : (tensor<?xi32>, tensor<i32>) -> tensor<?x1xi32>
  // CHECK-DAG: %[[OFFSET:.*]] = "tf.Sub"(%[[EXPAND_DIMS]], %[[RANGE_N]]) : (tensor<?x1xi32>, tensor<?xi32>) -> tensor<?x?xi32>
  // CHECK-DAG: %[[LE2:.*]] = "tf.LessEqual"(%[[OFFSET]], %[[NUM_LOWER_OR_M]]) : (tensor<?x?xi32>, tensor<i32>) -> tensor<?x?xi1>
  // CHECK-DAG: %[[NEG:.*]] = "tf.Neg"(%[[NUM_UPPER_OR_N]]) : (tensor<i32>) -> tensor<i32>
  // CHECK-DAG: %[[GE:.*]] = "tf.GreaterEqual"(%[[OFFSET]], %[[NEG]]) : (tensor<?x?xi32>, tensor<i32>) -> tensor<?x?xi1>
  // CHECK-DAG: %[[INDICATOR:.*]] = "tf.LogicalAnd"(%[[LE2]], %[[GE]]) : (tensor<?x?xi1>, tensor<?x?xi1>) -> tensor<?x?xi1>
  // CHECK-DAG: %[[SHAPE_I64:.*]] = "tf.Shape"(%[[INPUT]]) : (tensor<?x?xf32>) -> tensor<2xi64>
  // CHECK-DAG: %[[ZEROS_LIKE:.*]] = "tf.BroadcastTo"(%[[ZERO_F32]], %[[SHAPE_I64]]) : (tensor<f32>, tensor<2xi64>) -> tensor<?x?xf32>
  // CHECK-DAG: %[[RET:.*]] = "tf.SelectV2"(%[[INDICATOR]], %[[INPUT]], %[[ZEROS_LIKE]]) : (tensor<?x?xi1>, tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  // CHECK-DAG: return %[[RET]]
  %0 = "tf.MatrixBandPart"(%input, %num_lower, %num_upper) : (tensor<?x?xf32>, tensor<i32>, tensor<i32>) -> tensor<?x?xf32>
  func.return %0 : tensor<?x?xf32>
}

// CHECK-LABEL: func @unranked_matrix_band_part
func.func @unranked_matrix_band_part(%input: tensor<*xf32>, %num_lower: tensor<i64>, %num_upper: tensor<i64>) -> tensor<*xf32> {
  // CHECK-NOT: tf.MatrixBandPart
  %0 = "tf.MatrixBandPart"(%input, %num_lower, %num_upper) : (tensor<*xf32>, tensor<i64>, tensor<i64>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}
