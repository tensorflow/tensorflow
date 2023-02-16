// RUN: tf-opt "--tfxla-device-specific-transforms=device-type=XLA_CPU_JIT" -verify-diagnostics -split-input-file %s | FileCheck -dump-input=fail %s

module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 1399 : i32}} {

// CHECK-LABEL: stateless_op
func.func @stateless_op() -> tensor<i32> {
  // CHECK: %cst = "tf.Const"() {value = dense<1> : tensor<i32>} : () -> tensor<i32>
  %0 = "tf.StatelessRandomGetAlg"() {device = ""} : () -> tensor<i32>
  return %0 : tensor<i32>
}

}