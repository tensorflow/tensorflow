// RUN: kernel-gen-opt %s | FileCheck %s
// Verify the printed output can be parsed.
// RUN: kernel-gen-opt %s | kernel-gen-opt | FileCheck %s
// Verify the generic form can be parsed.
// RUN: kernel-gen-opt -mlir-print-op-generic %s | kernel-gen-opt | FileCheck %s

// CHECK-LABEL: func @alloc_raw
func @alloc_raw(%ctx: !tf_framework.op_kernel_context,
                   %size_0 : index , %size_2 : index) {
  %buf_0 = tf_framework.alloc_raw(%ctx) : memref<10xi8>
  %buf_1 = tf_framework.alloc_raw(%ctx, %size_0, %size_2) : memref<?x10x?xi8>
  return
}

// CHECK-LABEL: func @dealloc_raw
func @dealloc_raw(%ctx: !tf_framework.op_kernel_context, %memref : memref<?x10xf32>) {
  tf_framework.dealloc_raw(%ctx, %memref) : memref<?x10xf32>
  return
}

// CHECK-LABEL: func @null_context
func @null_context() {
  tf_framework.null_context() : !tf_framework.op_kernel_context
  return
}
