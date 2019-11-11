// RUN: tf-opt %s -test-tf-lower-tf | FileCheck %s --dump-input-on-failure

// CHECK-LABEL: simple_pack
// CHECK-SAME: %[[ARG0:.*]]: tensor<3x5xf32>, %[[ARG1:.*]]: tensor<3x5xf32>
func @simple_pack(%arg0: tensor<3x5xf32>, %arg1: tensor<3x5xf32>) -> tensor<2x3x5xf32> {
  // CHECK: %[[AXIS:.*]] = "tf.Const"() {value = dense<0> : tensor<i64>}
  // CHECK: %[[INP0:.*]] = "tf.ExpandDims"(%[[ARG0]], %[[AXIS]]) : (tensor<3x5xf32>, tensor<i64>) -> tensor<1x3x5xf32>
  // CHECK: %[[INP1:.*]] = "tf.ExpandDims"(%[[ARG1]], %[[AXIS]]) : (tensor<3x5xf32>, tensor<i64>) -> tensor<1x3x5xf32>
  // CHECK: "tf.ConcatV2"(%[[INP0]], %[[INP1]], %[[AXIS]]) {N = 2 : i64} : (tensor<1x3x5xf32>, tensor<1x3x5xf32>, tensor<i64>) -> tensor<2x3x5xf32>

  %0 = "tf.Pack"(%arg0, %arg1) {N = 2 : i64} : (tensor<3x5xf32>, tensor<3x5xf32>) -> tensor<2x3x5xf32>
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

// CHECK-LABEL: pack_with_unranked
// CHECK-SAME: %[[ARG0:.*]]: tensor<?x5xf32>, %[[ARG1:.*]]: tensor<*xf32>
func @pack_with_unranked(%arg0: tensor<?x5xf32>, %arg1: tensor<*xf32>) -> tensor<*xf32> {
  // CHECK: %[[AXIS:.*]] = "tf.Const"() {value = dense<-2> : tensor<i64>}
  // CHECK: %[[INP0:.*]] = "tf.ExpandDims"(%[[ARG0]], %[[AXIS]]) : (tensor<?x5xf32>, tensor<i64>) -> tensor<?x1x5xf32>
  // CHECK: %[[INP1:.*]] = "tf.ExpandDims"(%[[ARG1]], %[[AXIS]]) : (tensor<*xf32>, tensor<i64>) -> tensor<*xf32>
  // CHECK: "tf.ConcatV2"(%[[INP0]], %[[INP1]], %[[AXIS]]) {N = 2 : i64} : (tensor<?x1x5xf32>, tensor<*xf32>, tensor<i64>) -> tensor<*xf32>

  %0 = "tf.Pack"(%arg0, %arg1) {axis = -2 : i64, N = 2 : i64} : (tensor<?x5xf32>, tensor<*xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}

// CHECK-LABEL: func @pad
func @pad(%arg0: tensor<3xf32>) -> tensor<6xf32> {
  %padding = "tf.Const"() { value = dense<[[1, 2]]> : tensor<1x2xi64> } : () -> tensor<1x2xi64>
  // CHECK-DAG: [[PAD:%.+]] = "tf.Const"() {
  // CHECK-DAG: [[CST:%.+]] = "tf.Const"() {value = dense<0.000000e+00> : tensor<f32>}
  // CHECK: "tf.PadV2"(%arg0, [[PAD]], [[CST]])
  %0 = "tf.Pad"(%arg0, %padding) {N = 2 : i64} : (tensor<3xf32>, tensor<1x2xi64>) -> tensor<6xf32>
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
