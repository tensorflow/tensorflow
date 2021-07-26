// RUN: kernel-gen-opt %s -pass-pipeline='func(tf-to-jit-invocation{ \
// RUN:   tile-sizes=1,2,3 unroll-factors=3,2,1 cpu-codegen=false \
// RUN:   max-supported-rank=32})' | \
// RUN: FileCheck %s

// CHECK-LABEL: @rint
// CHECK-SAME: (%[[ARG:.*]]: tensor<*xf32>)
func @rint(%arg : tensor<*xf32>) -> tensor<*xf32> {
  // CHECK: %[[CALLABLE:.*]] = tf_framework.jit_compile_from_str
  // CHECK-SAME: "
  // CHECK-SAME: module {
  // CHECK-SAME: func @main(%arg0: tensor<*xf32>) -> tensor<*xf32> attributes {
  // CHECK-DAG:   llvm.emit_c_interface,
  // CHECK-DAG:   tf_entry
  // CHECK-DAG:   max_supported_rank = 32 : i64
  // CHECK-DAG:   tile_sizes = [1, 2, 3]
  // CHECK-DAG:   unroll_factors = [3, 2, 1]
  // CHECK-DAG:   cpu_codegen = false
  // CHECK-SAME: } {
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
