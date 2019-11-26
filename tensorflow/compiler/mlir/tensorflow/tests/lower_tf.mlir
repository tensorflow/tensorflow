// RUN: tf-opt %s -test-tf-lower-tf | FileCheck %s --dump-input-on-failure

// CHECK-LABEL: simple_pack
// CHECK-SAME: %[[ARG0:.*]]: tensor<3x5xf32>, %[[ARG1:.*]]: tensor<3x5xf32>
func @simple_pack(%arg0: tensor<3x5xf32>, %arg1: tensor<3x5xf32>) -> tensor<2x3x5xf32> {
  // CHECK: %[[AXIS:.*]] = "tf.Const"() {value = dense<0> : tensor<i64>}
  // CHECK: %[[INP0:.*]] = "tf.ExpandDims"(%[[ARG0]], %[[AXIS]]) : (tensor<3x5xf32>, tensor<i64>) -> tensor<1x3x5xf32>
  // CHECK: %[[INP1:.*]] = "tf.ExpandDims"(%[[ARG1]], %[[AXIS]]) : (tensor<3x5xf32>, tensor<i64>) -> tensor<1x3x5xf32>
  // CHECK: "tf.ConcatV2"(%[[INP0]], %[[INP1]], %[[AXIS]]) : (tensor<1x3x5xf32>, tensor<1x3x5xf32>, tensor<i64>) -> tensor<2x3x5xf32>

  %0 = "tf.Pack"(%arg0, %arg1) : (tensor<3x5xf32>, tensor<3x5xf32>) -> tensor<2x3x5xf32>
  return %0 : tensor<2x3x5xf32>
}

// CHECK-LABEL: func @square
func @square(%arg0: tensor<3xf32>) -> tensor<3xf32> {
  // CHECK: "tf.Mul"(%arg0, %arg0)
  %1 = "tf.Square"(%arg0) : (tensor<3xf32>) -> tensor<3xf32>
  return %1 : tensor<3xf32>
}

// CHECK-LABEL: func @squared_difference_real
func @squared_difference_real(%arg0: tensor<3xf32>, %arg1: tensor<3xf32>) -> tensor<3xf32> {
  // CHECK: [[R1:%.+]] = "tf.Sub"(%arg0, %arg1)
  // CHECK: "tf.Mul"([[R1]], [[R1]])
  %1 = "tf.SquaredDifference"(%arg0, %arg1) : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xf32>
  return %1 : tensor<3xf32>
}

// CHECK-LABEL: func @squared_difference_complex
func @squared_difference_complex(%arg0: tensor<3xcomplex<f32>>, %arg1: tensor<3xcomplex<f32>>) -> tensor<3xcomplex<f32>> {
  // CHECK-DAG: [[R1:%.+]] = "tf.Sub"(%arg0, %arg1)
  // CHECK-DAG: [[R2:%.+]] = "tf.Conj"([[R1]])
  // CHECK-DAG: "tf.Mul"([[R1]], [[R2]])
  %1 = "tf.SquaredDifference"(%arg0, %arg1) : (tensor<3xcomplex<f32>>, tensor<3xcomplex<f32>>) -> tensor<3xcomplex<f32>>
  return %1 : tensor<3xcomplex<f32>>
}

// CHECK-LABEL: func @div_no_nan
// CHECK-SAME: (%[[X:.*]]: tensor<*xf32>, %[[Y:.*]]: tensor<*xf32>)
func @div_no_nan(%arg0: tensor<*xf32>, %arg1: tensor<*xf32>) -> tensor<*xf32> {
  // CHECK:  %[[ZERO:.*]] = "tf.Const"() {value = dense<0.000000e+00> : tensor<f32>} : () -> tensor<f32>
  // CHECK:  %[[IS_ZERO:.*]] = "tf.Equal"(%[[Y]], %[[ZERO]]) {incompatible_shape_error = true} : (tensor<*xf32>, tensor<f32>) -> tensor<*xi1>
  // CHECK:  %[[DIV:.*]] = "tf.Div"(%[[X]], %[[Y]]) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
  // CHECK:  %[[RESULT:.*]] = "tf.SelectV2"(%[[IS_ZERO]], %[[ZERO]], %[[DIV]]) : (tensor<*xi1>, tensor<f32>, tensor<*xf32>) -> tensor<*xf32>
  %0 = "tf.DivNoNan"(%arg0, %arg1) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>

  // CHECK: return %[[RESULT]]
  return %0 : tensor<*xf32>
}

// CHECK-LABEL: func @mul_no_nan
// CHECK-SAME: (%[[X:.*]]: tensor<2x3xf32>, %[[Y:.*]]: tensor<3xf32>)
func @mul_no_nan(%arg0: tensor<2x3xf32>, %arg1: tensor<3xf32>) -> tensor<2x3xf32> {
  // CHECK:  %[[ZERO:.*]] = "tf.Const"() {value = dense<0.000000e+00> : tensor<f32>} : () -> tensor<f32>
  // CHECK:  %[[IS_ZERO:.*]] = "tf.Equal"(%[[Y]], %[[ZERO]]) {incompatible_shape_error = true} : (tensor<3xf32>, tensor<f32>) -> tensor<3xi1>
  // CHECK:  %[[MUL:.*]] = "tf.Mul"(%[[X]], %[[Y]]) : (tensor<2x3xf32>, tensor<3xf32>) -> tensor<2x3xf32>
  // CHECK:  %[[RESULT:.*]] = "tf.SelectV2"(%[[IS_ZERO]], %[[ZERO]], %[[MUL]]) : (tensor<3xi1>, tensor<f32>, tensor<2x3xf32>) -> tensor<2x3xf32>
  %0 = "tf.MulNoNan"(%arg0, %arg1) : (tensor<2x3xf32>, tensor<3xf32>) -> tensor<2x3xf32>

  // CHECK: return %[[RESULT]]
  return %0 : tensor<2x3xf32>
}

// CHECK-LABEL: func @fill
// CHECK-SAME: (%[[ARG0:.*]]: tensor<*xi64>, %[[ARG1:.*]]: tensor<*xf32>)
func @fill(%arg0: tensor<*xi64>, %arg1: tensor<*xf32>) -> tensor<*xf32> {
  // CHECK: "tf.BroadcastTo"(%[[ARG1]], %[[ARG0]])
  %0 = "tf.Fill"(%arg0, %arg1) : (tensor<*xi64>, tensor<*xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}

// CHECK-LABEL: func @l2_loss
// CHECK-SAME: (%[[INPUT:.*]]: tensor<?x?xf32>)
func @l2_loss(%arg0: tensor<?x?xf32>) -> tensor<f32> {

  // CHECK-DAG: %[[SQUARE:.*]] = "tf.Mul"(%[[INPUT]], %[[INPUT]]) : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  // CHECK-DAG: %[[REDUCE_AXES:.*]] = "tf.Const"() {value = dense<[0, 1]> : tensor<2xi64>}
  // CHECK-DAG: %[[SUM:.*]] = "tf.Sum"(%[[SQUARE]], %[[REDUCE_AXES]]) {keep_dims = false} : (tensor<?x?xf32>, tensor<2xi64>) -> tensor<f32>
  // CHECK-DAG: %[[TWO:.*]] = "tf.Const"() {value = dense<2.000000e+00> : tensor<f32>}
  // CHECK-DAG: %[[LOSS:.*]] = "tf.Div"(%[[SUM]], %[[TWO]]) : (tensor<f32>, tensor<f32>) -> tensor<f32>

  %0 = "tf.L2Loss"(%arg0) : (tensor<?x?xf32>) -> tensor<f32>

  // CHECK: return %[[LOSS]] : tensor<f32>
  return %0 : tensor<f32>
}

// CHECK-LABEL: func @l2_loss_unranked
func @l2_loss_unranked(%arg0: tensor<*xf32>) -> tensor<f32> {
  // CHECK: tf.L2Loss
  %0 = "tf.L2Loss"(%arg0) : (tensor<*xf32>) -> tensor<f32>
  return %0 : tensor<f32>
}

// CHECK-LABEL: pack_with_unranked
// CHECK-SAME: %[[ARG0:.*]]: tensor<?x5xf32>, %[[ARG1:.*]]: tensor<*xf32>
func @pack_with_unranked(%arg0: tensor<?x5xf32>, %arg1: tensor<*xf32>) -> tensor<*xf32> {
  // CHECK: %[[AXIS:.*]] = "tf.Const"() {value = dense<-2> : tensor<i64>}
  // CHECK: %[[INP0:.*]] = "tf.ExpandDims"(%[[ARG0]], %[[AXIS]]) : (tensor<?x5xf32>, tensor<i64>) -> tensor<?x1x5xf32>
  // CHECK: %[[INP1:.*]] = "tf.ExpandDims"(%[[ARG1]], %[[AXIS]]) : (tensor<*xf32>, tensor<i64>) -> tensor<*xf32>
  // CHECK: "tf.ConcatV2"(%[[INP0]], %[[INP1]], %[[AXIS]]) : (tensor<?x1x5xf32>, tensor<*xf32>, tensor<i64>) -> tensor<*xf32>

  %0 = "tf.Pack"(%arg0, %arg1) {axis = -2 : i64} : (tensor<?x5xf32>, tensor<*xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}

// CHECK-LABEL: func @pad
func @pad(%arg0: tensor<3xf32>) -> tensor<6xf32> {
  %padding = "tf.Const"() { value = dense<[[1, 2]]> : tensor<1x2xi64> } : () -> tensor<1x2xi64>
  // CHECK-DAG: [[PAD:%.+]] = "tf.Const"() {
  // CHECK-DAG: [[CST:%.+]] = "tf.Const"() {value = dense<0.000000e+00> : tensor<f32>}
  // CHECK: "tf.PadV2"(%arg0, [[PAD]], [[CST]])
  %0 = "tf.Pad"(%arg0, %padding) : (tensor<3xf32>, tensor<1x2xi64>) -> tensor<6xf32>
  return %0 : tensor<6xf32>
}

// CHECK-LABEL: func @BiasAddGrad_NHWC
func @BiasAddGrad_NHWC(%arg0: tensor<2x3x4x5xf32>) -> tensor<5xf32> {
  // CHECK: "tf.Const"() {value = dense<[0, 1, 2]> : tensor<3xi64>}
  // CHECK: "tf.Sum"({{.*}}) {keep_dims = false}

  %0 = "tf.BiasAddGrad"(%arg0) {data_format = "NHWC"} : (tensor<2x3x4x5xf32>) -> tensor<5xf32>
  return %0 : tensor<5xf32>
}

// CHECK-LABEL: func @BiasAddGrad_NCHW
func @BiasAddGrad_NCHW(%arg0: tensor<2x3x4x5xf32>) -> tensor<3xf32> {
  // CHECK: "tf.Const"() {value = dense<[0, 2, 3]> : tensor<3xi64>}
  // CHECK: "tf.Sum"({{.*}}) {keep_dims = false}

  %0 = "tf.BiasAddGrad"(%arg0) {data_format = "NCHW"} : (tensor<2x3x4x5xf32>) -> tensor<3xf32>
  return %0 : tensor<3xf32>
}

// CHECK-LABEL: func @BiasAddGrad_dynamic
func @BiasAddGrad_dynamic(%arg0: tensor<?x?x?x?xf32>) -> tensor<?xf32> {
  // CHECK: tf.Sum
  %0 = "tf.BiasAddGrad"(%arg0) {data_format = "NCHW"} : (tensor<?x?x?x?xf32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

// CHECK-LABEL: func @BiasAddGrad_unranked
func @BiasAddGrad_unranked(%arg0: tensor<*xf32>) -> tensor<?xf32> {
  // CHECK: tf.BiasAddGrad
  %0 = "tf.BiasAddGrad"(%arg0) {data_format = "NCHW"} : (tensor<*xf32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

// CHECK-LABEL: func @rsqrt_grad
// CHECK-SAME: (%[[ARG0:.*]]: tensor<2xf32>, %[[ARG1:.*]]: tensor<2xf32>)
func @rsqrt_grad(%arg0: tensor<2xf32>, %arg1: tensor<2xf32>) -> tensor<2xf32> {
  // CHECK: %[[CST:.*]] = "tf.Const"() {value = dense<-2.000000e+00> : tensor<f32>}
  // CHECK: %[[LHS2:.*]] = "tf.Mul"(%[[ARG0]], %[[ARG0]])
  // CHECK: %[[LHS3:.*]] = "tf.Mul"(%[[LHS2]], %[[ARG0]])
  // CHECK: %[[DIV:.*]] = "tf.Div"(%[[ARG1]], %[[CST]])
  // CHECK: %[[RET:.*]] = "tf.Mul"(%[[LHS3]], %[[DIV]])

  %0 = "tf.RsqrtGrad"(%arg0, %arg1) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
  // CHECK: return %[[RET]]
  return %0 : tensor<2xf32>
}

// CHECK-LABEL: func @rsqrt_grad_unranked
func @rsqrt_grad_unranked(%arg0: tensor<*xf32>, %arg1: tensor<*xf32>) -> tensor<*xf32> {
  // CHECK: tf.Const
  // CHECK: tf.Mul
  // CHECK: tf.Mul
  // CHECK: tf.Div
  // CHECK: tf.Mul
  %0 = "tf.RsqrtGrad"(%arg0, %arg1) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}


// CHECK-LABEL: SoftmaxCrossEntropyWithLogits
// CHECK-SAME: %[[FEATURES:.*]]: tensor<2x3xf32>, %[[LABELS:.*]]: tensor<2x3xf32>
func @SoftmaxCrossEntropyWithLogits(%features: tensor<2x3xf32>, %labels: tensor<2x3xf32>) -> (tensor<2xf32>, tensor<2x3xf32>) {
  // CHECK-DAG: %[[NEG_LABELS:.*]] = "tf.Neg"(%[[LABELS]]) : (tensor<2x3xf32>) -> tensor<2x3xf32>
  // CHECK-DAG: %[[LOG_SOFTMAX:.*]] = "tf.LogSoftmax"(%[[FEATURES]]) : (tensor<2x3xf32>) -> tensor<2x3xf32>
  // CHECK-DAG: %[[LOSS_INP:.*]] = "tf.Mul"(%[[NEG_LABELS]], %[[LOG_SOFTMAX]]) : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
  // CHECK-DAG: %[[AXIS:.*]] = "tf.Const"() {value = dense<-1> : tensor<1xi64>} : () -> tensor<1xi64>
  // CHECK-DAG: %[[LOSS:.*]] = "tf.Sum"(%[[LOSS_INP]], %[[AXIS]]) {keep_dims = false} : (tensor<2x3xf32>, tensor<1xi64>) -> tensor<2xf32>
  // CHECK-DAG: %[[SOFTMAX:.*]] = "tf.Softmax"(%[[FEATURES]]) : (tensor<2x3xf32>) -> tensor<2x3xf32>
  // CHECK-DAG: %[[BACKPROP:.*]] = "tf.Sub"(%[[SOFTMAX]], %[[LABELS]]) : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
  // CHECK: return %[[LOSS]], %[[BACKPROP]]

  %0:2 = "tf.SoftmaxCrossEntropyWithLogits"(%features, %labels) : (tensor<2x3xf32>, tensor<2x3xf32>) -> (tensor<2xf32>, tensor<2x3xf32>)
  return %0#0, %0#1 : tensor<2xf32>, tensor<2x3xf32>
}

// CHECK-LABEL: unranked_SoftmaxCrossEntropyWithLogits
func @unranked_SoftmaxCrossEntropyWithLogits(%features: tensor<?x?xf32>, %labels: tensor<?x?xf32>) -> (tensor<?xf32>, tensor<?x?xf32>) {
  // Check that unranked inputs are lowered successfully.
  // CHECK-NOT: tf.SoftmaxCrossEntropyWithLogits
  %0:2 = "tf.SoftmaxCrossEntropyWithLogits"(%features, %labels) : (tensor<?x?xf32>, tensor<?x?xf32>) -> (tensor<?xf32>, tensor<?x?xf32>)
  return %0#0, %0#1 : tensor<?xf32>, tensor<?x?xf32>
}

// CHECK-LABEL: broadcasted_SoftmaxCrossEntropyWithLogits
func @broadcasted_SoftmaxCrossEntropyWithLogits(%features: tensor<?x?xf32>, %labels: tensor<3xf32>) -> (tensor<?xf32>, tensor<?x3xf32>) {
  // Check that inputs of different ranks are broadcasted and are lowered successfully.
  // CHECK-NOT: tf.SoftmaxCrossEntropyWithLogits
  %0:2 = "tf.SoftmaxCrossEntropyWithLogits"(%features, %labels) : (tensor<?x?xf32>, tensor<3xf32>) -> (tensor<?xf32>, tensor<?x3xf32>)
  return %0#0, %0#1 : tensor<?xf32>, tensor<?x3xf32>
}

// CHECK-LABEL: scalar_SoftmaxCrossEntropyWithLogits
func @scalar_SoftmaxCrossEntropyWithLogits(%features: tensor<f32>, %labels: tensor<?x?xf32>) -> (tensor<?xf32>, tensor<?x?xf32>) {
  // CHECK: tf.SoftmaxCrossEntropyWithLogits
  %0:2 = "tf.SoftmaxCrossEntropyWithLogits"(%features, %labels) : (tensor<f32>, tensor<?x?xf32>) -> (tensor<?xf32>, tensor<?x?xf32>)
  return %0#0, %0#1 : tensor<?xf32>, tensor<?x?xf32>
}

// CHECK-LABEL: func @tanhgrad_float
// CHECK-SAME: (%[[Y:.*]]: tensor<*xf32>, %[[DY:.*]]: tensor<*xf32>)
func @tanhgrad_float(%y : tensor<*xf32>, %dy: tensor<*xf32>) -> tensor<*xf32> {
  // CHECK: %[[ONE:.*]] = "tf.Const"() {value = dense<1.000000e+00> : tensor<f32>} : () -> tensor<f32>
  // CHECK: %[[Y_SQUARE:.*]] = "tf.Mul"(%[[Y]], %[[Y]]) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
  // CHECK: %[[SUB:.*]] = "tf.Sub"(%[[ONE]], %[[Y_SQUARE]]) : (tensor<f32>, tensor<*xf32>) -> tensor<*xf32>
  // CHECK: %[[RESULT:.*]] = "tf.Mul"(%[[DY]], %[[SUB]]) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
  %0 = "tf.TanhGrad"(%y, %dy) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>

  // CHECK: return %[[RESULT]]
  return %0 : tensor<*xf32>
}

// CHECK-LABEL: func @tanhgrad_complex
// CHECK-SAME: (%[[Y:.*]]: tensor<*xcomplex<f32>>, %[[DY:.*]]: tensor<*xcomplex<f32>>)
func @tanhgrad_complex(%y : tensor<*xcomplex<f32>>, %dy: tensor<*xcomplex<f32>>) -> tensor<*xcomplex<f32>> {
  // CHECK: tf.TanhGrad
  %0 = "tf.TanhGrad"(%y, %dy) : (tensor<*xcomplex<f32>>, tensor<*xcomplex<f32>>) -> tensor<*xcomplex<f32>>

  return %0 : tensor<*xcomplex<f32>>
}

// CHECK-LABEL: func @ZerosLike_unranked
func @ZerosLike_unranked(%arg0: tensor<*xi32>) -> tensor<*xi32> {
  // CHECK: %[[ZERO:.*]] = "tf.Const"() {value = dense<0> : tensor<i32>} : () -> tensor<i32>
  // CHECK: %[[SHAPE:.*]] = "tf.Shape"(%arg0) : (tensor<*xi32>) -> tensor<?xi64>
  // CHECK: "tf.BroadcastTo"(%[[ZERO]], %[[SHAPE]]) : (tensor<i32>, tensor<?xi64>) -> tensor<*xi32>

  %0 = "tf.ZerosLike"(%arg0) : (tensor<*xi32>) -> tensor<*xi32>
  return %0 : tensor<*xi32>
}

// CHECK-LABEL: func @ZerosLike_variant
func @ZerosLike_variant(%arg0: tensor<!tf.variant<tensor<2xi32>>>) -> tensor<!tf.variant<tensor<2xi32>>> {
  // CHECK: tf.ZerosLike
  %0 = "tf.ZerosLike"(%arg0) : (tensor<!tf.variant<tensor<2xi32>>>) -> tensor<!tf.variant<tensor<2xi32>>>
  return %0 : tensor<!tf.variant<tensor<2xi32>>>
}
