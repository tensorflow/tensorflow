// RUN: tf-tfrt-opt -tf-to-tfrt-data %s | FileCheck %s

module {

// CHECK-LABEL: func @main() -> !tfrt_data.dataset
  func.func @main() -> tensor<*x!tf_type.variant> {
    // CHECK-NEXT: %[[START:.*]] = tfrt.constant.i64 0
    // CHECK-NEXT: %[[STEP:.*]] = tfrt.constant.i64 1
    // CHECK-NEXT: %[[STOP:.*]] = tfrt.constant.i64 1000
    // CHECK-NEXT: %[[RANGE:.*]] = tfrt_data.range_dataset %[[START]], %[[STOP]], %[[STEP]] {element_type = i64}
    // CHECK-NEXT: tfrt.return %[[RANGE]] : !tfrt_data.dataset
    %1 = "tf.Const"() {value = dense<0> : tensor<i64>} : () -> tensor<i64>
    %2 = "tf.Const"() {value = dense<1> : tensor<i64>} : () -> tensor<i64>
    %3 = "tf.Const"() {value = dense<1000> : tensor<i64>} : () -> tensor<i64>
    %4 = "tf.RangeDataset"(%1, %3, %2) {device = "", output_shapes = [#tf_type.shape<>], output_types = [i64], metadata = ""} : (tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<*x!tf_type.variant>
    func.return %4 : tensor<*x!tf_type.variant>
  }
}
