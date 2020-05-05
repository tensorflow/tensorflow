// RUN: tf-opt -tf-legalize-to-hex %s -o -| FileCheck %s


// CHECK-LABEL: func @constants() {
func @constants() {
  // CHECK: "hex.constant_int"() {value = 1 : i32}
  %0 = "tf.Const"() {device = "", dtype = "tfdtype$DT_INT32", name = "x", value = dense<1> : tensor<i32>} : () -> tensor<i32>
  // CHECK: "hex.constant_int"() {value = 42 : i32}
  %1 = "tf.Const"() {device = "", dtype = "tfdtype$DT_INT32", name = "y", value = dense<42> : tensor<1x1xi32>} : () -> tensor<1x1xi32>
  // CHECK: hex.return
  return
}

// CHECK-LABEL: func @add
func @add(%arg0: tensor<1xi32>) {
  // CHECK: hex.add_int
  %2 = "tf.Add"(%arg0, %arg0) {T = "tfdtype$DT_INT32", device = "", name = "z"} : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
  return
}
