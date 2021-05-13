// RUN: kernel-gen-opt %s | FileCheck %s
// Verify the printed output can be parsed.
// RUN: kernel-gen-opt %s | kernel-gen-opt | FileCheck %s
// Verify the generic form can be parsed.
// RUN: kernel-gen-opt -mlir-print-op-generic %s | kernel-gen-opt | FileCheck %s

// CHECK-LABEL: func @alloc
func @alloc(%ctx: !tf_framework.op_kernel_context,
                   %size_0 : index , %size_2 : index) {
  %buf_0 = tf_framework.alloc(%ctx) : memref<10xi8>
  %buf_1 = tf_framework.alloc(%ctx, %size_0, %size_2) : memref<?x10x?xi8>
  return
}

// CHECK-LABEL: func @forwarding_alloc
func @forwarding_alloc(%ctx: !tf_framework.op_kernel_context,
                       %size_0 : index , %size_2 : index) {
  %buf = tf_framework.alloc(%ctx, %size_0, %size_2) {
    input_indices = [0 : i32, 1 : i32],
    output_index = 0 : i32
  } : memref<?x10x?xi8>
  return
}

// CHECK-LABEL: func @dealloc
func @dealloc(%ctx: !tf_framework.op_kernel_context,
              %memref : memref<?x10xf32>) {
  tf_framework.dealloc(%ctx, %memref) : memref<?x10xf32>
  return
}

// CHECK-LABEL: func @assert
func @assert(%ctx: !tf_framework.op_kernel_context) {
  tf_framework.report_error %ctx, "INVALID_ARGUMENT", "Everything is awesome"
  return
}

// CHECK-LABEL: func @null_memref
func @null_memref() {
  tf_framework.null_memref : memref<*xf32>
  return
}

// CHECK-LABEL: func @null_context
func @null_context() {
  tf_framework.null_context : !tf_framework.op_kernel_context
  return
}
