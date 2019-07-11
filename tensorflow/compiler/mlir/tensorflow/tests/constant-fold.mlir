// RUN: tf-opt %s -test-constant-fold | FileCheck %s

// CHECK-LABEL: func @testShape
func @testShape(tensor<f32>, tensor<1x32x32x16xf32>, tensor<*xf32>) -> (tensor<0xi32>, tensor<?xi32>, tensor<?xi32>) {
^bb0(%arg0: tensor<f32>, %arg1: tensor<1x32x32x16xf32>, %arg2: tensor<*xf32>):

  // CHECK: constant dense<[]> : tensor<0xi32>
  %0 = "tf.Shape"(%arg0) {T = "tfdtype$DT_FLOAT", output = "tfdtype$DT_INT32"} : (tensor<f32>) -> tensor<0xi32>

  // Result shape need not be static. Folding harness uses TensorFlow constant
  // in that case.
  // CHECK: "tf.Const"() {value = dense<[1, 32, 32, 16]> : tensor<4xi32>} : () -> tensor<?xi32>
  %1 = "tf.Shape"(%arg1) {T = "tfdtype$DT_FLOAT", output = "tfdtype$DT_INT32"} : (tensor<1x32x32x16xf32>) -> tensor<?xi32>

  // CHECK: "tf.Shape"(%arg2) {T = "tfdtype$DT_FLOAT", output = "tfdtype$DT_INT32"} : (tensor<*xf32>) -> tensor<?xi32>
  %2 = "tf.Shape"(%arg2) {T = "tfdtype$DT_FLOAT", output = "tfdtype$DT_INT32"} : (tensor<*xf32>) -> tensor<?xi32>

  return %0, %1, %2 : tensor<0xi32>, tensor<?xi32>, tensor<?xi32>
}

// CHECK-LABEL: func @testLeakyRelu
func @testLeakyRelu(%arg0 : tensor<16xf32>) -> (tensor<16xf32>, tensor<f32>, tensor<f32>, tensor<16xf32>) {
  %pos = constant dense<5.0> : tensor<f32>
  %neg = constant dense<-5.0> : tensor<f32>
  %no = "tf.LeakyRelu"(%arg0) {alpha = 0.2 : f32} : (tensor<16xf32>) -> tensor<16xf32>
  %0 = "tf.LeakyRelu"(%pos) {alpha = 0.3 : f32} : (tensor<f32>) -> tensor<f32>
  %1 = "tf.LeakyRelu"(%neg) {alpha = 0.2 : f32} : (tensor<f32>) -> tensor<f32>
  %2 = "tf.LeakyRelu"(%arg0) {alpha = 3.0 : f32} : (tensor<16xf32>) -> tensor<16xf32>
  // CHECK: [[POS:%.*]] = constant dense<5.000000e+00> : tensor<f32>
  // CHECK: [[NEG:%.*]] = constant dense<-1.000000e+00> : tensor<f32>
  // CHECK: [[NC1:%.*]] = "tf.LeakyRelu"(%arg0) {alpha = 2.000000e-01 : f32} : (tensor<16xf32>) -> tensor<16xf32>
  // CHECK: [[NC2:%.*]] = "tf.LeakyRelu"(%arg0) {alpha = 3.000000e+00 : f32} : (tensor<16xf32>) -> tensor<16xf32>
  // CHECK: return [[NC1]], [[POS]], [[NEG]], [[NC2]]
  return %no, %0, %1, %2 : tensor<16xf32>, tensor<f32>, tensor<f32>, tensor<16xf32>
}

// CHECK-LABEL: func @tfConst
func @tfConst() -> (tensor<4xf32>, tensor<1x1x6x2xf32>) {
  %0 = "tf.Const"() {device = "", name = "Const", dtype = "tfdtype$DT_FLOAT", value = opaque<"tf", "0x746674656E736F722464747970653A2044545F464C4F41540A74656E736F725F7368617065207B0A202064696D207B0A2020202073697A653A20340A20207D0A7D0A74656E736F725F636F6E74656E743A20225C3030305C3030305C323430405C3030305C30303020405C3030305C303030205C3330315C3030305C3030305C3230305C323737220A"> : tensor<4xf32>} : () -> tensor<4xf32>
  %21 = "tf.Const"() {device = "", name = "Const_143", dtype = "tfdtype$DT_FLOAT", value = dense<0.24288677062973696> : tensor<1x1x6x2xf32>} : () -> tensor<1x1x6x2xf32>
  // CHECK-DAG: value = opaque<"tf"
  // CHECK-DAG: constant dense<0.242886767> : tensor<1x1x6x2xf32>
  return %0, %21 : tensor<4xf32>, tensor<1x1x6x2xf32>
}
