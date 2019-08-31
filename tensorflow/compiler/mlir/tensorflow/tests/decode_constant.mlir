// RUN: tf-opt %s -tf-decode-constant | FileCheck %s

func @decodeOpaque() -> tensor<4xf32> {
  // CHECK: "tf.Const"() {device = "", dtype = "tfdtype$DT_FLOAT", name = "Const", value = dense<[5.000000e+00, 2.500000e+00, -1.000000e+01, -1.000000e+00]> : tensor<4xf32>} : () -> tensor<4xf32>
  %0 = "tf.Const"() {device = "", name = "Const", dtype = "tfdtype$DT_FLOAT", value = opaque<"tf", "0x746674656E736F722464747970653A2044545F464C4F41540A74656E736F725F7368617065207B0A202064696D207B0A2020202073697A653A20340A20207D0A7D0A74656E736F725F636F6E74656E743A20225C3030305C3030305C323430405C3030305C30303020405C3030305C303030205C3330315C3030305C3030305C3230305C323737220A"> : tensor<4xf32>} : () -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

func @skipSplat() -> tensor<32xf32> {
  // CHECK: "tf.Const"() {device = "", dtype = "tfdtype$DT_FLOAT", name = "Const", value = dense<2.500000e-01> : tensor<32xf32>} : () -> tensor<32xf32>
  %0 = "tf.Const"() {device = "", name = "Const", dtype = "tfdtype$DT_FLOAT", value = dense<0.25> : tensor<32xf32>} : () -> tensor<32xf32>
  return %0 : tensor<32xf32>
}
