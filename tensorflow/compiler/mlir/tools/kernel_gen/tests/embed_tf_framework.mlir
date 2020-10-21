// RUN: kernel-gen-opt %s -embed-tf-framework -split-input-file | FileCheck %s

// CHECK-LABEL: func @tf_entry(
// CHECK-SAME:    [[CTX:%.*]]: !tf_framework.op_kernel_context,
// CHECK-SAME:    [[SIZE_0:%.*]]: index,
// CHECK-SAME:    [[SIZE_2:%.*]]: index) -> index attributes {tf_entry} {
func @tf_entry(%size_0 : index , %size_2 : index) -> index
    attributes {tf_entry} {
  %buf = alloc(%size_0, %size_2)[] : memref<?x10x?xf32>
  dealloc %buf : memref<?x10x?xf32>
  std.return %size_0 : index
}
// CHECK-NEXT: [[VAL_3:%.*]] = tf_framework.alloc
// CHECK-SAME:   ([[CTX]], [[SIZE_0]], [[SIZE_2]]) : memref<?x10x?xf32>
// CHECK-NEXT: tf_framework.dealloc([[CTX]], [[VAL_3]]) : memref<?x10x?xf32>
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
  %buf = alloc()[%size] : memref<64xf32, affine_map<(d0)[s0] -> (d0 + s0)>>
  dealloc %buf : memref<64xf32, affine_map<(d0)[s0] -> (d0 + s0)>>
  std.return
}
// CHECK_NOT: alloc_raw
// CHECK: alloc()
// CHECK_NOT: dealloc_raw
// CHECK: dealloc %
