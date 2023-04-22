// RUN: kernel-gen-opt %s --tf-to-jit-invocation | \
// RUN: FileCheck %s

// CHECK-LABEL: @rint
// CHECK-SAME: (%[[ARG:.*]]: tensor<*xf32>)
func @rint(%arg : tensor<*xf32>) -> tensor<*xf32> {
  // CHECK: %[[CALLABLE:.*]] = tf_framework.jit_compile_from_str
  // CHECK-SAME: "
  // CHECK-SAME: module {
  // CHECK-SAME: func @main(%arg0: tensor<*xf32>) -> tensor<*xf32> attributes {llvm.emit_c_interface, tf_entry} {
  // CHECK-SAME: %0 = \22tf.Tanh\22(%arg0)
  // CHECK-SAME: %1 = \22tf.Rint\22(%0)
  // CHECK-SAME: return %1
  // CHECK-SAME: }
  // CHECK-SAME: }
  // CHECK-SAME: "
  // CHECK: %[[RES:.*]] = tf_framework.jit_execute %[[CALLABLE]](%[[ARG]])
  // CHECK: return %[[RES]]
  %0 = "tf.Tanh"(%arg) : (tensor<*xf32>) -> tensor<*xf32>
  %1 = "tf.Rint"(%0) : (tensor<*xf32>) -> tensor<*xf32>
  return %1 : tensor<*xf32>
}
