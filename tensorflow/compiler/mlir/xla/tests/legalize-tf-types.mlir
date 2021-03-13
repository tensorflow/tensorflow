// RUN: tf-opt -xla-legalize-tf-types %s | FileCheck %s

func @relu_qint8(%arg0: tensor<1x!tf.qint8>) -> tensor<1x!tf.qint8> {
  // CHECK: func @relu_qint8(%arg0: tensor<1xi8>) -> tensor<1xi8> {
  // CHECK-NEXT: %[[X:.*]] = "tf.Relu"(%arg0) : (tensor<1xi8>) -> tensor<1xi8>
  %0 = "tf.Relu"(%arg0) : (tensor<1x!tf.qint8>) -> tensor<1x!tf.qint8>
  return %0: tensor<1x!tf.qint8>
}

func @if_qint8(%arg0: tensor<i1>, %arg1: tensor<1x!tf.qint8>, %arg2: tensor<1x!tf.qint8>) -> tensor<1x!tf.qint8> {
  // CHECK: func @if_qint8(%arg0: tensor<i1>, %arg1: tensor<1xi8>, %arg2: tensor<1xi8>) -> tensor<1xi8>
  // CHECK-NEXT: %0 = "tf.IfRegion"(%arg0) ( {
  // CHECK-NEXT:   "tf.Yield"(%arg1) : (tensor<1xi8>) -> ()
  // CHECK-NEXT:   }, {
  // CHECK-NEXT:   "tf.Yield"(%arg2) : (tensor<1xi8>) -> ()
  // CHECK-NEXT:  }) {is_stateless = false} : (tensor<i1>) -> tensor<1xi8>
  // CHECK-NEXT: return %0 : tensor<1xi8>
  %0 = "tf.IfRegion"(%arg0) ( {
    "tf.Yield"(%arg1) : (tensor<1x!tf.qint8>) -> ()
    }, {
    "tf.Yield"(%arg2) : (tensor<1x!tf.qint8>) -> ()
   }) {is_stateless = false} : (tensor<i1>) -> tensor<1x!tf.qint8>
  return %0 : tensor<1x!tf.qint8>
}

func @id_qint8(%arg0: tensor<1x!tf.qint8>) -> tensor<1x!tf.qint8> {
  // CHECK: func @id_qint8(%arg0: tensor<1xi8>) -> tensor<1xi8> {
  // CHECK-NEXT: return %arg0 : tensor<1xi8>
  return %arg0: tensor<1x!tf.qint8>
}

func @id_qint16(%arg0: tensor<1x!tf.qint16>) -> tensor<1x!tf.qint16> {
  // CHECK: func @id_qint16(%arg0: tensor<1xi16>) -> tensor<1xi16> {
  // CHECK-NEXT: return %arg0 : tensor<1xi16>
  return %arg0: tensor<1x!tf.qint16>
}

func @id_qint32(%arg0: tensor<1x!tf.qint32>) -> tensor<1x!tf.qint32> {
  // CHECK: func @id_qint32(%arg0: tensor<1xi32>) -> tensor<1xi32> {
  // CHECK-NEXT: return %arg0 : tensor<1xi32>
  return %arg0: tensor<1x!tf.qint32>
}

func @id_quint8(%arg0: tensor<1x!tf.quint8>) -> tensor<1x!tf.quint8> {
  // CHECK: func @id_quint8(%arg0: tensor<1xui8>) -> tensor<1xui8> {
  // CHECK-NEXT: return %arg0 : tensor<1xui8>
  return %arg0: tensor<1x!tf.quint8>
}

func @id_quint16(%arg0: tensor<1x!tf.quint16>) -> tensor<1x!tf.quint16> {
  // CHECK: func @id_quint16(%arg0: tensor<1xui16>) -> tensor<1xui16> {
  // CHECK-NEXT: return %arg0 : tensor<1xui16>
  return %arg0: tensor<1x!tf.quint16>
}
