// RUN: tf-mlir-translate -mlir-to-graphdef %s -o - | FileCheck %s

// CHECK:      name: "tf.ParseExample"
// CHECK-NEXT: op: "ParseExample"
// CHECK-NEXT: input: "tf.Const{{_.*_3}}"
// CHECK-NEXT: input: "tf.Const"
// CHECK-NEXT: input: "tf.Const{{_.*_1}}"
// CHECK-NEXT: input: "tf.Const{{_.*_2}}"
// CHECK-NEXT: attr {
// CHECK-NEXT:   key: "Ndense"
// CHECK-NEXT:   value {
// CHECK-NEXT:     i: 1
// CHECK-NEXT:   }
// CHECK-NEXT: }
// CHECK-NEXT: attr {
// CHECK-NEXT:   key: "Nsparse"
// CHECK-NEXT:   value {
// CHECK-NEXT:     i: 0
// CHECK-NEXT:   }
// CHECK-NEXT: }
// CHECK-NEXT: attr {
// CHECK-NEXT:   key: "Tdense"
// CHECK-NEXT:   value {
// CHECK-NEXT:     list {
// CHECK-NEXT:       type: DT_INT64
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT: }
// CHECK-NEXT: attr {
// CHECK:        key: "dense_shapes"
// CHECK-NEXT:   value {
// CHECK-NEXT:     list {
// CHECK-NEXT:       shape {
// CHECK-NEXT:         dim {
// CHECK-NEXT:           size: 1
// CHECK-NEXT:         }
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT: }
// CHECK-NEXT: attr {
// CHECK-NEXT:   key: "sparse_types"
// CHECK-NEXT:   value {
// CHECK-NEXT:     list {
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT: }

module attributes {tf.versions = {bad_consumers = [], min_consumer = 0 : i32, producer = 413 : i32}} {
  func.func @main() -> tensor<*xi64> attributes {tf.entry_function = {control_outputs = "", inputs = "", outputs = "result"}} {
    %0 = tf_executor.graph {
      %outputs, %control = tf_executor.island wraps "tf.Const"() {device = "", value = dense<"value"> : tensor<1x!tf_type.string>} : () -> tensor<1x!tf_type.string>
      %outputs_0, %control_1 = tf_executor.island wraps "tf.Const"() {device = "", value = dense<"value"> : tensor<!tf_type.string>} : () -> tensor<!tf_type.string>
      %outputs_2, %control_3 = tf_executor.island wraps "tf.Const"() {device = "", value = dense<-1> : tensor<i64>} : () -> tensor<i64>
      %outputs_4, %control_5 = tf_executor.island wraps "tf.Const"() {device = "", value = dense<""> : tensor<1x!tf_type.string>} : () -> tensor<1x!tf_type.string>
      %outputs_6, %control_7 = tf_executor.island wraps "tf.ParseExample"(%outputs_4, %outputs, %outputs_0, %outputs_2) {dense_shapes = [#tf_type.shape<1>], device = "", operandSegmentSizes = array<i32: 1, 1, 0, 1, 1>, resultSegmentSizes = array<i32: 0, 0, 0, 1>} : (tensor<1x!tf_type.string>, tensor<1x!tf_type.string>, tensor<!tf_type.string>, tensor<i64>) -> tensor<*xi64>
      tf_executor.fetch %outputs_6 : tensor<*xi64>
    }
    func.return %0 : tensor<*xi64>
  }
}
