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
  %0:2 = "_tf.VariableV2"() {dtype = "tfdtype$DT_INT32", value = dense<2> : tensor<i32>} : () -> (tensor<!tf.int32ref>, !_tf.control) loc("Ref_Variable")
  %1:2 = "_tf.Mul"(%0#0, %0#0) : (tensor<!tf.int32ref>, tensor<!tf.int32ref>) -> (tensor<*x!tf.int32ref>, !_tf.control) loc("foo")
  return
}
