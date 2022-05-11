// RUN: kernel-gen-opt %s -split-input-file -verify-diagnostics -embed-tf-framework |\
// RUN: FileCheck %s

// CHECK-LABEL: func @tf_entry(
// CHECK-SAME:    [[CTX:%.*]]: !tf_framework.op_kernel_context,
// CHECK-SAME:    [[SIZE_0:%.*]]: index,
// CHECK-SAME:    [[SIZE_2:%.*]]: index) -> index attributes {tf_entry} {
func.func @tf_entry(%size_0 : index , %size_2 : index) -> index
    attributes {tf_entry} {
  %buf = memref.alloc(%size_0, %size_2)[] : memref<?x10x?xf32>
  memref.dealloc %buf : memref<?x10x?xf32>
  func.return %size_0 : index
}
// CHECK-NEXT: [[BUF:%.*]] = tf_framework.alloc
// CHECK-SAME:   ([[CTX]], [[SIZE_0]], [[SIZE_2]]) : memref<?x10x?xf32>

// CHECK-NEXT: [[IS_VALID:%.*]] = tf_framework.is_valid_memref([[BUF]])
// CHECK-SAME: memref<?x10x?xf32> -> i1

// CHECK-NEXT: tf_framework.assert [[CTX]], [[IS_VALID]],
// CHECK-SAME:   RESOURCE_EXHAUSTED, "failed to allocate memory"

// CHECK-NEXT: tf_framework.dealloc([[CTX]], [[BUF]]) : memref<?x10x?xf32>
// CHECK-NEXT: return [[SIZE_0]] : index

// -----

// CHECK-LABEL: func @non_tf_entry(
// CHECK-SAME:    [[SIZE_0:%.*]]: index, [[SIZE_2:%.*]]: index) -> index
func.func @non_tf_entry(%size_0 : index , %size_2 : index) -> index {
  func.return %size_0 : index
}

// -----

func.func @tf_entry_no_ctx(%size : index) attributes {tf_entry} {
  // expected-error @+1 {{failed to legalize operation 'memref.alloc' that was explicitly marked illegal}}
  %buf = memref.alloc()[%size] : memref<64xf32, affine_map<(d0)[s0] -> (d0 + s0)>>
  memref.dealloc %buf : memref<64xf32, affine_map<(d0)[s0] -> (d0 + s0)>>
  func.return
}

// -----

// CHECK-LABEL: func @assert(
// CHECK-SAME: [[CTX:%.*]]: !tf_framework.op_kernel_context
func.func @assert(%arg0: !tf_framework.op_kernel_context) attributes {tf_entry} {
  %true = arith.constant true
  cf.assert %true, "the one and only"
  func.return
}
// CHECK:   [[TRUE:%.*]] = arith.constant true
// CHECK-NEXT: tf_framework.assert [[CTX]], [[TRUE]], INVALID_ARGUMENT

// -----

// CHECK-LABEL: func @jit_execute
// CHECK-SAME:  %[[CTX:.*]]: !tf_framework.op_kernel_context,
// CHECK-SAME:  %[[F:.*]]: !tf_framework.jit_callable,
// CHECK-SAME:  %[[ARG0:.*]]: tensor<2x?xf32>,
// CHECK-SAME:  %[[ARG1:.*]]: tensor<2x?xf32>
func.func @jit_execute(%ctx : !tf_framework.op_kernel_context,
    %f : !tf_framework.jit_callable, %arg0 : tensor<2x?xf32>,
    %arg1 : tensor<2x?xf32>) -> tensor<2x?xf32> attributes {tf_entry} {
  // CHECK: %[[RES:.*]] = tf_framework.jit_execute
  // CHECK-SAME: ctx(%[[CTX]]) %[[F]](%[[ARG0]], %[[ARG1]])
  // CHECK: return %[[RES]]
  %0 = tf_framework.jit_execute %f(%arg0, %arg1)
      : tensor<2x?xf32>, tensor<2x?xf32> -> tensor<2x?xf32>
  func.return %0 : tensor<2x?xf32>
}

// -----

// CHECK-LABEL: func @jit_compile_from_str
// CHECK-SAME:  (%[[CTX:.*]]: !tf_framework.op_kernel_context)
func.func @jit_compile_from_str(%ctx : !tf_framework.op_kernel_context)
    -> !tf_framework.jit_callable attributes {tf_entry} {
  // CHECK: %[[RES:.*]] = tf_framework.jit_compile_from_str %[[CTX]], "placeholder"
  // CHECK: return %[[RES]]
  %0 = tf_framework.jit_compile_from_str "placeholder" {
      architectures = ["sm_123", "sm_456"], tileSizes = [1, 2, 3],
      unrollFactors = [4], maxSupportedRank = 3 : i64, enableFtz = false,
      index64Bit = false, cpuCodegen = false }
  func.return %0 : !tf_framework.jit_callable
}
