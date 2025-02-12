// RUN: tf-tfrt-opt %s -tf-restore-merging | FileCheck %s

// CHECK-LABEL: func @single_restore_group
// CHECK-SAME:    (%[[ARG0:.*]]: {{.*}})
func.func @single_restore_group(%arg0: tensor<!tf_type.string>) -> (tensor<*xf32>, tensor<*xi32>) {
  %0 = "tf.Const"() {value = dense<"foo"> : tensor<1x!tf_type.string>} : () -> tensor<1x!tf_type.string>
  %1 = "tf.Const"() {value = dense<""> : tensor<1x!tf_type.string>} : () -> tensor<1x!tf_type.string>
  %2 = "tf.RestoreV2"(%arg0, %0, %1) : (tensor<!tf_type.string>, tensor<1x!tf_type.string>, tensor<1x!tf_type.string>) -> tensor<*xf32>

  %3 = "tf.Const"() {value = dense<"bar"> : tensor<1x!tf_type.string>} : () -> tensor<1x!tf_type.string>
  %4 = "tf.Const"() {value = dense<""> : tensor<1x!tf_type.string>} : () -> tensor<1x!tf_type.string>
  %5 = "tf.RestoreV2"(%arg0, %3, %4) : (tensor<!tf_type.string>, tensor<1x!tf_type.string>, tensor<1x!tf_type.string>) -> tensor<*xi32>

  // CHECK:      %[[NAMES:.*]] = "tf.Const"() <{value = dense<["foo", "bar"]> : tensor<2x!tf_type.string>}>
  // CHECK-NEXT:      %[[SHAPES:.*]] = "tf.Const"() <{value = dense<""> : tensor<2x!tf_type.string>}>
  // CHECK-NEXT:      %[[TENSORS:.*]]:2 = "tf.RestoreV2"(%[[ARG0]], %[[NAMES]], %[[SHAPES]])
  // CHECK-SAME:   -> (tensor<*xf32>, tensor<*xi32>)

  // CHECK:      return %[[TENSORS]]#0, %[[TENSORS]]#1
  func.return %2, %5 : tensor<*xf32>, tensor<*xi32>
}
