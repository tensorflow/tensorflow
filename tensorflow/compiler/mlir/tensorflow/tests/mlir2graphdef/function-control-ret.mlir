// RUN: tf-mlir-translate -mlir-to-graphdef %s -o - | FileCheck %s

module attributes {tf.versions = {bad_consumers = [], min_consumer = 12 : i32, producer = 29 : i32}} {
  func @main() {
    tf_executor.graph {
      %0 = tf_executor.island wraps "tf.PartitionedCall"() {Tin = [], Tout = [], config = "", config_proto = "", device = "", executor_type = "", f = @foo, name = "Call_foo"} : () -> ()
      tf_executor.fetch
    }
    return
  }
  func @foo() {
    tf_executor.graph {
      %0:2 = tf_executor.island {
        %1 = "tf.Const"() {device = "", dtype = "tfdtype$DT_INT32", value = dense<5> : tensor<i32>} : () -> tensor<i32> loc("control_const")
        tf_executor.yield %1 : tensor<i32>
      }
      // CHECK: control_output: "control_const"
      // CHECK:       control_ret {
      // CHECK-NEXT:    key: "control_const"
      // CHECK-NEXT:    value: "control_const"
      // CHECK-NEXT:  }
      tf_executor.fetch %0#1 : !tf_executor.control
    }
    return
  }
}
