// RUN: kernel-gen-opt %s --split-input-file \
// RUN:   --tf-64bit-indexer="architectures=sm_123,sm_456 \
// RUN:   tile-sizes=1,2,3 unroll-factors=3,2,1 max-supported-rank=32 \
// RUN:   enable-ftz=false index_64bit=false cpu-codegen=false \
// RUN:   jit_i64_indexed_for_large_tensors=false" | \
// RUN: FileCheck %s

// CHECK-LABEL: @unary_tanh_rint
// CHECK-SAME: (%[[ARG:.*]]: tensor<*xf32>)
func @unary_tanh_rint(%arg : tensor<*xf32>) -> (tensor<*xf32>) {
  // CHECK-SAME: %c4294967296 = arith.constant 4294967296 : index
  // CHECK-SAME: %0 = shape.shape_of %arg0 : tensor<*xf32> -> tensor<?xindex>
  // CHECK-SAME: %1 = shape.num_elements %0 : tensor<?xindex> -> index
  // CHECK-SAME: %2 = arith.cmpi sgt, %1, %c4294967296 : index
  // CHECK: %[[RES:.*]] = scf.if %2 -> (tensor<*xf32>) {
  // CHECK-SAME:    %4 = tf_framework.jit_compile   {
  // CHECK-SAME:    ^bb0(%arg1: tensor<*xf32>):  // no predecessors
  // CHECK-SAME:      %6 = "tf.Tanh"(%arg1) : (tensor<*xf32>) -> tensor<*xf32>
  // CHECK-SAME:      tf_framework.jit_compile_yield %6 : tensor<*xf32>
  // CHECK-SAME:    }
  // CHECK-SAME:    %5 = tf_framework.jit_execute %4(%arg0) : tensor<*xf32> -> tensor<*xf32>
  // CHECK-SAME:    scf.yield %5 : tensor<*xf32>
  // CHECK-SAME: } else {
  // CHECK-SAME:    %4 = "tf.Tanh"(%arg0) : (tensor<*xf32>) -> tensor<*xf32>
  // CHECK-SAME:    scf.yield %4 : tensor<*xf32>
  // CHECK-SAME: }
  // CHECK: return %[[RES]]
  %0 = "tf.Tanh"(%arg) : (tensor<*xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}