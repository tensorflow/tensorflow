// RUN: kernel-gen-opt %s | FileCheck %s
// Verify the printed output can be parsed.
// RUN: kernel-gen-opt %s | kernel-gen-opt -allow-unregistered-dialect | FileCheck %s
// Verify the generic form can be parsed.
// RUN: kernel-gen-opt -mlir-print-op-generic %s | kernel-gen-opt -allow-unregistered-dialect | FileCheck %s

// CHECK-LABEL: func @alloc_output
func @alloc_output(%ctx: !tf_framework.op_kernel_context,
                   %size_0 : index , %size_2 : index) {
  %buf_0 = tf_framework.alloc_output(%ctx) : memref<10xi8>
  %buf_1 = tf_framework.alloc_output(%ctx, %size_0, %size_2) : memref<?x10x?xi8>
  return
}

// CHECK-LABEL: func @alloc_temp
func @alloc_temp(%ctx: !tf_framework.op_kernel_context,
                   %size_0 : index , %size_2 : index) {
  %buf_0 = tf_framework.alloc_temp(%ctx) : memref<10xi8>
  %buf_1 = tf_framework.alloc_temp(%ctx, %size_0, %size_2) : memref<?x10x?xi8>
  return
}
