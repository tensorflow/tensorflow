// RUN: tf-mlir-translate -mlir-to-graphdef %s -o - | FileCheck %s

// Verify that attribute T is not set as ref type.
// CHECK-LABEL: name: "Ref_Variable"
// CHECK-LABEL: name: "foo"
// CHECK:      attr {
// CHECK-NEXT:   key: "T"
// CHECK-NEXT:   value {
// CHECK-NEXT:     type: DT_INT32{{$}}
// CHECK-NEXT:   }
// CHECK-NEXT: }

func @main() {
  tf_executor.graph {
    %0:2 = tf_executor.island wraps "tf.VariableV2"() {dtype = "tfdtype$DT_INT32", value = dense<2> : tensor<i32>, shape = #tf.shape<2>, container = "", shared_name = ""} : () -> tensor<!tf.int32ref> loc("Ref_Variable")
    %1:2 = tf_executor.island wraps "tf.Identity"(%0#0) : (tensor<!tf.int32ref>) -> tensor<*x!tf.int32ref> loc("foo")
    tf_executor.fetch
  }
  return
}
