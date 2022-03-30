// RUN: tf-opt %s -tfl-legalize-tf='preserve-assert-op=true' | FileCheck %s --dump-input=fail

func.func @preserve_assert(%arg0: tensor<1xi32>, %arg1: tensor<1xi32>) -> tensor<1xi1> {
  %0 = "tf.LessEqual"(%arg0, %arg1) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
  "tf.Assert"(%0, %arg1) {summarize = 3} : (tensor<1xi1>, tensor<1xi32>) -> ()
  func.return %0 : tensor<1xi1>
  // CHECK-LABEL: preserve_assert
  // CHECK: tfl.less_equal
  // CHECK: Assert
  // CHECK: return
}
