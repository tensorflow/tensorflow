// RUN: tf-opt %s -test-tf-lower-tf | FILECHECK_OPTS="" FileCheck %s

// CHECK-LABEL: dequantize
func.func @dequantize(%arg0: tensor<2x3x!tf_type.qint8>, %min_range: tensor<f32>, %max_range: tensor<f32>) -> tensor<2x3xf32> {
  // CHECK-DAG: %[[HALF_RANGE:.*]] = "tf.Const"() {value = dense<1.280000e+02> : tensor<f32>}
  // CHECK-DAG: %[[C255:.*]] = "tf.Const"() {value = dense<2.550000e+02> : tensor<f32>}
  // CHECK-DAG: %[[CAST:.*]] = "tf.Cast"(%arg0) {Truncate = false}
  // CHECK-DAG: %[[SHIFT:.*]] = "tf.AddV2"(%[[CAST]], %[[HALF_RANGE]])
  // CHECK-DAG: %[[DRANGE:.*]] = "tf.Sub"(%arg2, %arg1)
  // CHECK-DAG: %[[SCALE:.*]] = "tf.Div"(%[[DRANGE]], %[[C255:.*]])
  // CHECK-DAG: %[[SS:.*]] = "tf.Mul"(%[[SHIFT]], %[[SCALE]])
  // CHECK-DAG: %[[RESULT:.*]] = "tf.AddV2"(%[[SS]], %arg1)
  %0 = "tf.Dequantize"(%arg0, %min_range, %max_range) : (tensor<2x3x!tf_type.qint8>, tensor<f32>, tensor<f32>) -> tensor<2x3xf32>

  // CHECK-DAG: return %[[RESULT]]
  func.return %0 : tensor<2x3xf32>
}

// CHECK-LABEL: dequantize_quint8
func.func @dequantize_quint8(%arg0: tensor<2x3x!tf_type.quint8>, %min_range: tensor<f32>, %max_range: tensor<f32>) -> tensor<2x3xf32> {
  // CHECK-NEXT: %[[C255:.*]] = "tf.Const"() {value = dense<2.550000e+02> : tensor<f32>}
  // CHECK-NEXT: %[[CAST:.*]] = "tf.Cast"(%arg0) {Truncate = false}
  // CHECK-NEXT: %[[DRANGE:.*]] = "tf.Sub"(%arg2, %arg1)
  // CHECK-NEXT: %[[SCALE:.*]] = "tf.Div"(%[[DRANGE]], %[[C255:.*]])
  // CHECK-NEXT: %[[SS:.*]] = "tf.Mul"(%[[CAST]], %[[SCALE]])
  // CHECK-NEXT: %[[RESULT:.*]] = "tf.AddV2"(%[[SS]], %arg1)
  %0 = "tf.Dequantize"(%arg0, %min_range, %max_range) : (tensor<2x3x!tf_type.quint8>, tensor<f32>, tensor<f32>) -> tensor<2x3xf32>

  // CHECK-DAG: return %[[RESULT]]
  func.return %0 : tensor<2x3xf32>
}

// CHECK-LABEL: dequantize_to_bf16
func.func @dequantize_to_bf16(%arg0: tensor<2x3x!tf_type.qint8>, %min_range: tensor<f32>, %max_range: tensor<f32>) -> tensor<2x3xbf16> {
  // CHECK-DAG: %[[HALF_RANGE:.*]] = "tf.Const"() {value = dense<1.280000e+02> : tensor<f32>}
  // CHECK-DAG: %[[C255:.*]] = "tf.Const"() {value = dense<2.550000e+02> : tensor<f32>}
  // CHECK-DAG: %[[CAST:.*]] = "tf.Cast"(%arg0) {Truncate = false}
  // CHECK-DAG: %[[SHIFT:.*]] = "tf.AddV2"(%[[CAST]], %[[HALF_RANGE]])
  // CHECK-DAG: %[[DRANGE:.*]] = "tf.Sub"(%arg2, %arg1)
  // CHECK-DAG: %[[SCALE:.*]] = "tf.Div"(%[[DRANGE]], %[[C255:.*]])
  // CHECK-DAG: %[[SS:.*]] = "tf.Mul"(%[[SHIFT]], %[[SCALE]])
  // CHECK-DAG: %[[F32_RESULT:.*]] = "tf.AddV2"(%[[SS]], %arg1)
  // CHECK-DAG: %[[RESULT:.*]] = "tf.Cast"(%[[F32_RESULT]]) {Truncate = false}
  %0 = "tf.Dequantize"(%arg0, %min_range, %max_range) : (tensor<2x3x!tf_type.qint8>, tensor<f32>, tensor<f32>) -> tensor<2x3xbf16>

  // CHECK-DAG: return %[[RESULT]]
  func.return %0 : tensor<2x3xbf16>
}
