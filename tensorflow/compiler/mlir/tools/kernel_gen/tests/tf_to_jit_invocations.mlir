// RUN: kernel-gen-opt %s --split-input-file \
// RUN:   --tf-to-jit-invocation="tile-sizes=1,2,3 unroll-factors=3,2,1 \
// RUN:     max-supported-rank=32 enable-ftz=false cpu-codegen=false" | \
// RUN: FileCheck %s

// CHECK-LABEL: @unary_tanh
// CHECK-SAME: (%[[ARG:.*]]: tensor<*xf32>)
func.func @unary_tanh(%arg : tensor<*xf32>) -> tensor<*xf32> {
  // CHECK: %[[CALLABLE:.*]] = tf_framework.jit_compile_from_str
  // CHECK-SAME: "
  // CHECK-SAME: module {
  // CHECK-SAME:   func @main(%arg0: tensor<*xf32>) -> tensor<*xf32>
  // CHECK-SAME:     attributes {tf_entry}
  // CHECK-SAME:   {
  // CHECK-SAME:     %0 = \22tf.Tanh\22(%arg0)
  // CHECK-SAME:     return %0
  // CHECK-SAME:   }
  // CHECK-SAME: }
  // CHECK-SAME: "
  // CHECK-SAME: {
  // CHECK-SAME:   cpuCodegen = false
  // CHECK-SAME:   enableFtz = false
  // CHECK-SAME:   maxSupportedRank = 32 : i64
  // CHECK-SAME:   tileSizes = [1, 2, 3]
  // CHECK-SAME:   unrollFactors = [3, 2, 1]
  // CHECK-SAME: }
  // CHECK: %[[RES:.*]] = tf_framework.jit_execute %[[CALLABLE]](%[[ARG]])
  // CHECK: return %[[RES]]
  %0 = "tf.Tanh"(%arg) : (tensor<*xf32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}

// -----

// CHECK-LABEL: @binary_sub
// CHECK-SAME: (%[[ARG0:.*]]: tensor<*xf32>, %[[ARG1:.*]]: tensor<*xf32>)
func.func @binary_sub(%arg0 : tensor<*xf32>, %arg1 : tensor<*xf32>) -> tensor<*xf32> {
  // CHECK: %[[CALLABLE:.*]] = tf_framework.jit_compile_from_str
  // CHECK-SAME: "
  // CHECK-SAME: module {
  // CHECK-SAME:   func @main(%arg0: tensor<*xf32>, %arg1: tensor<*xf32>) -> tensor<*xf32>
  // CHECK-SAME:     attributes {tf_entry}
  // CHECK-SAME:   {
  // CHECK-SAME:     %0 = \22tf.Sub\22(%arg0, %arg1)
  // CHECK-SAME:     return %0
  // CHECK-SAME:   }
  // CHECK-SAME: }
  // CHECK-SAME: "
  // CHECK-SAME: {
  // CHECK-SAME:   cpuCodegen = false
  // CHECK-SAME:   enableFtz = false
  // CHECK-SAME:   maxSupportedRank = 32 : i64
  // CHECK-SAME:   tileSizes = [1, 2, 3]
  // CHECK-SAME:   unrollFactors = [3, 2, 1]
  // CHECK-SAME: }
  // CHECK: %[[RES:.*]] = tf_framework.jit_execute %[[CALLABLE]](%[[ARG0]], %[[ARG1]])
  // CHECK: return %[[RES]]
  %0 = "tf.Sub"(%arg0, %arg1) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}
