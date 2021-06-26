// RUN: kernel-gen-opt %s -split-input-file -embed-tf-framework |\
// RUN: FileCheck %s

// CHECK-LABEL: func @tf_entry(
// CHECK-SAME:    [[CTX:%.*]]: !tf_framework.op_kernel_context,
// CHECK-SAME:    [[SIZE_0:%.*]]: index,
// CHECK-SAME:    [[SIZE_2:%.*]]: index) -> index attributes {tf_entry} {
func @tf_entry(%size_0 : index , %size_2 : index) -> index
    attributes {tf_entry} {
  %buf = memref.alloc(%size_0, %size_2)[] : memref<?x10x?xf32>
  memref.dealloc %buf : memref<?x10x?xf32>
  std.return %size_0 : index
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
func @non_tf_entry(%size_0 : index , %size_2 : index) -> index {
  std.return %size_0 : index
}

// -----

// CHECK-LABEL: func @tf_entry(
func @tf_entry(%size : index) attributes {tf_entry} {
  %buf = memref.alloc()[%size] : memref<64xf32, affine_map<(d0)[s0] -> (d0 + s0)>>
  memref.dealloc %buf : memref<64xf32, affine_map<(d0)[s0] -> (d0 + s0)>>
  std.return
}
// CHECK_NOT: tf_framework.alloc
// CHECK: alloc()
// CHECK_NOT: tf_framework.dealloc
// CHECK: dealloc %

// -----

// CHECK-LABEL: func @assert(
// CHECK-SAME: [[CTX:%.*]]: !tf_framework.op_kernel_context
func @assert(%arg0: !tf_framework.op_kernel_context) attributes {tf_entry} {
  %true = constant true
  assert %true, "the one and only"
  return
}
// CHECK:   [[TRUE:%.*]] = constant true
// CHECK-NEXT: tf_framework.assert [[CTX]], [[TRUE]], INVALID_ARGUMENT
