// RUN: tf-tfrt-opt -tf-to-tfrt-data %s | FileCheck %s

module {

// CHECK-LABEL: func @main() -> !tfrt_data.dataset
  func.func @main() -> tensor<*x!tf_type.variant> {
    // CHECK-NEXT: %[[START:.*]] = tfrt.constant.i64 0
    // CHECK-NEXT: %[[STEP:.*]] = tfrt.constant.i64 1
    // CHECK-NEXT: %[[STOP:.*]] = tfrt.constant.i64 1000
    // CHECK-NEXT: %[[RANGE:.*]] = tfrt_data.range_dataset %[[START]], %[[STOP]], %[[STEP]] {element_type = i64}
    // CHECK-NEXT: %[[BATCH_SIZE:.*]] = tfrt.constant.i64 10
    // CHECK-NEXT: %[[DROP_REMAINDER:.*]] = tfrt.constant.i1 false
    // CHECK-NEXT: %[[BATCH:.*]] = tfrt_data.batch_dataset.i64 %[[RANGE]], %[[BATCH_SIZE]] {same_input_metadata = false}
    // CHECK-NEXT: tfrt.return %[[BATCH]] : !tfrt_data.dataset
    %start = "tf.Const"() {value = dense<0> : tensor<i64>} : () -> tensor<i64>
    %step = "tf.Const"() {value = dense<1> : tensor<i64>} : () -> tensor<i64>
    %stop = "tf.Const"() {value = dense<1000> : tensor<i64>} : () -> tensor<i64>
    %range = "tf.RangeDataset"(%start, %stop, %step) {output_shapes = [#tf_type.shape<>], output_types = [i64], metadata = ""} : (tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<*x!tf_type.variant>
    %batch_size = "tf.Const"() {value = dense<10> : tensor<i64>} : () -> tensor<i64>
    %drop_remainder = "tf.Const"() {value = dense<false> : tensor<i1>} : () -> tensor<i1>
    %batch = "tf.BatchDatasetV2"(%range, %batch_size, %drop_remainder) {output_shapes = [#tf_type.shape<>], output_types = [i64], parallel_copy = false, metadata = ""} : (tensor<*x!tf_type.variant>, tensor<i64>, tensor<i1>) -> tensor<*x!tf_type.variant>
    func.return %batch : tensor<*x!tf_type.variant>
  }
}
