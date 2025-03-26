// RUN: tf-tfrt-opt %s -tf-restore-splitting | FileCheck %s

// CHECK-LABEL: func @single_restore
// CHECK-SAME:    (%[[ARG0:.*]]: {{.*}})
func.func @single_restore(%arg0: tensor<!tf_type.string>) -> (tensor<*xf32>, tensor<*xi32>) {
  %0 = "tf.Const"() {value = dense<["foo", "bar"]> : tensor<2x!tf_type.string>} : () -> tensor<2x!tf_type.string>
  %1 = "tf.Const"() {value = dense<""> : tensor<2x!tf_type.string>} : () -> tensor<2x!tf_type.string>
  %2:2 = "tf.RestoreV2"(%arg0, %0, %1) : (tensor<!tf_type.string>, tensor<2x!tf_type.string>, tensor<2x!tf_type.string>) -> (tensor<*xf32>, tensor<*xi32>)

  // CHECK: %[[FOO_NAME:.*]] = "tf.Const"() <{value = dense<"foo"> : tensor<1x!tf_type.string>}>
  // CHECK: %[[FOO:.*]] = "tf.RestoreV2"(%[[ARG0]], %[[FOO_NAME]], {{.*}})

  // CHECK: %[[BAR_NAME:.*]] = "tf.Const"() <{value = dense<"bar"> : tensor<1x!tf_type.string>}>
  // CHECK: %[[BAR:.*]] = "tf.RestoreV2"(%[[ARG0]], %[[BAR_NAME]], {{.*}})

  // CHECK: return %[[FOO]], %[[BAR]]
  func.return %2#0, %2#1 : tensor<*xf32>, tensor<*xi32>
}
