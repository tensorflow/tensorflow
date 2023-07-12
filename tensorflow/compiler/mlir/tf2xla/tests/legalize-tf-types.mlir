// RUN: tf-opt -xla-legalize-tf-types %s | FileCheck %s

func.func @gather_v2_qint8(%arg0: tensor<16x2x3x!tf_type.qint8>, %arg1: tensor<16x5xi32>) -> tensor<16x2x5x!tf_type.qint8> {
  // CHECK: func @gather_v2_qint8(%arg0: tensor<16x2x3xi8>, %arg1: tensor<16x5xi32>) -> tensor<16x2x5xi8> {
  %axis = "tf.Const"() { value = dense<[-1]> : tensor<1xi32> } : () -> tensor<1xi32>
  %1 = "tf.GatherV2"(%arg0, %arg1, %axis) {batch_dims = -1 : i64} : (tensor<16x2x3x!tf_type.qint8>, tensor<16x5xi32>, tensor<1xi32>) -> tensor<16x2x5x!tf_type.qint8>
  func.return %1 : tensor<16x2x5x!tf_type.qint8>
}

func.func @id_qint8(%arg0: tensor<1x!tf_type.qint8>) -> tensor<1x!tf_type.qint8> {
  // CHECK: func @id_qint8(%arg0: tensor<1xi8>) -> tensor<1xi8> {
  // CHECK-NEXT: return %arg0 : tensor<1xi8>
  func.return %arg0: tensor<1x!tf_type.qint8>
}

func.func @id_qint16(%arg0: tensor<1x!tf_type.qint16>) -> tensor<1x!tf_type.qint16> {
  // CHECK: func @id_qint16(%arg0: tensor<1xi16>) -> tensor<1xi16> {
  // CHECK-NEXT: return %arg0 : tensor<1xi16>
  func.return %arg0: tensor<1x!tf_type.qint16>
}

func.func @id_qint32(%arg0: tensor<1x!tf_type.qint32>) -> tensor<1x!tf_type.qint32> {
  // CHECK: func @id_qint32(%arg0: tensor<1xi32>) -> tensor<1xi32> {
  // CHECK-NEXT: return %arg0 : tensor<1xi32>
  func.return %arg0: tensor<1x!tf_type.qint32>
}

func.func @id_quint8(%arg0: tensor<1x!tf_type.quint8>) -> tensor<1x!tf_type.quint8> {
  // CHECK: func @id_quint8(%arg0: tensor<1xui8>) -> tensor<1xui8> {
  // CHECK-NEXT: return %arg0 : tensor<1xui8>
  func.return %arg0: tensor<1x!tf_type.quint8>
}

func.func @id_quint16(%arg0: tensor<1x!tf_type.quint16>) -> tensor<1x!tf_type.quint16> {
  // CHECK: func @id_quint16(%arg0: tensor<1xui16>) -> tensor<1xui16> {
  // CHECK-NEXT: return %arg0 : tensor<1xui16>
  func.return %arg0: tensor<1x!tf_type.quint16>
}

func.func @quantize_dequantize_qint8_not_converted(%arg0: tensor<1xf32>) -> tensor<1xf32> {
  // CHECK: tf_type.qint8
  %scales = "tf.Const"() { value = dense<2.0> : tensor<f32> } : () -> tensor<f32>
  %zps = "tf.Const"() { value = dense<4> : tensor<i32> } : () -> tensor<i32>
  %0 = "tf.UniformQuantize"(%arg0, %scales, %zps) {
    quantization_axis = -1 : i64, quantization_min_val = -128 : i64, quantization_max_val = 127 : i64
  } : (tensor<1xf32>, tensor<f32>, tensor<i32>) -> tensor<1x!tf_type.qint8>
  %1 = "tf.UniformDequantize"(%0, %scales, %zps) {
    quantization_axis = -1 : i64, quantization_min_val = -128 : i64, quantization_max_val = 127 : i64
  } : (tensor<1x!tf_type.qint8>, tensor<f32>, tensor<i32>) -> tensor<1xf32>
  func.return %1 : tensor<1xf32>
}
