// RUN: tf-opt %s -tf-strip-noinline-attribute | FileCheck %s

// CHECK-LABEL: func @strip_simple(
// CHECK-NOT: tf._noinline
func @strip_simple() -> tensor<2xi32> attributes {tf._noinline = true} {
  // CHECK-NEXT: %[[CST:.*]] = "tf.Const"
  %cst = "tf.Const"() { value = dense<2> : tensor<2xi32> } : () -> tensor<2xi32>
  return %cst : tensor<2xi32>
}
