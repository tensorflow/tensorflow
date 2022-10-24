// RUN: kernel-gen-opt %s | FileCheck %s
// Verify the printed output can be parsed.
// RUN: kernel-gen-opt %s | kernel-gen-opt | FileCheck %s
// Verify the generic form can be parsed.
// RUN: kernel-gen-opt -mlir-print-op-generic %s | kernel-gen-opt | FileCheck %s

// CHECK-LABEL: func @alloc
func.func @alloc(%ctx: !tf_framework.op_kernel_context,
                   %size_0 : index , %size_2 : index) {
  %buf_0 = tf_framework.alloc(%ctx) : memref<10xi8>
  %buf_1 = tf_framework.alloc(%ctx, %size_0, %size_2) : memref<?x10x?xi8>
  func.return
}

// CHECK-LABEL: func @forwarding_alloc
func.func @forwarding_alloc(%ctx: !tf_framework.op_kernel_context,
                       %size_0 : index , %size_2 : index) {
  %buf = tf_framework.alloc(%ctx, %size_0, %size_2) {
    input_indices = [0 : i32, 1 : i32],
    output_index = 0 : i32
  } : memref<?x10x?xi8>
  func.return
}

// CHECK-LABEL: func @dealloc
func.func @dealloc(%ctx: !tf_framework.op_kernel_context,
              %memref : memref<?x10xf32>) {
  tf_framework.dealloc(%ctx, %memref) : memref<?x10xf32>
  func.return
}

// CHECK-LABEL: func @assert
func.func @assert(%ctx: !tf_framework.op_kernel_context, %cond: i1) {
  tf_framework.assert %ctx, %cond, "ALREADY_EXISTS", "Or maybe not"
  func.return
}

// CHECK-LABEL: func @report_error
func.func @report_error(%ctx: !tf_framework.op_kernel_context) {
  tf_framework.report_error %ctx, "INVALID_ARGUMENT", "Everything is awesome"
  func.return
}

// CHECK-LABEL: func @null_memref
func.func @null_memref() {
  tf_framework.null_memref : memref<*xf32>
  func.return
}

// CHECK-LABEL: func @null_context
func.func @null_context() {
  tf_framework.null_context : !tf_framework.op_kernel_context
  func.return
}

// CHECK-LABEL: func @is_valid_memref
func.func @is_valid_memref(%buf: memref<?xf32>) -> i1 {
  %pred = tf_framework.is_valid_memref(%buf) : memref<?xf32> -> i1
  func.return %pred : i1
}

// CHECK-LABEL: func @jit_compile_wo_ctx
func.func @jit_compile_wo_ctx() -> !tf_framework.jit_callable {
  %callable = tf_framework.jit_compile {
  ^bb0(%arg : tensor<2x?xf32>):
    tf_framework.jit_compile_yield %arg : tensor<2x?xf32>
  }
  func.return %callable : !tf_framework.jit_callable
}

// CHECK-LABEL: func @jit_compile
func.func @jit_compile(%ctx : !tf_framework.op_kernel_context)
    -> !tf_framework.jit_callable {
  %callable = tf_framework.jit_compile %ctx {
  ^bb0(%arg : tensor<2x?xf32>):
    tf_framework.jit_compile_yield %arg : tensor<2x?xf32>
  }
  func.return %callable : !tf_framework.jit_callable
}

// CHECK-LABEL: func @jit_compile_from_str_wo_ctx
func.func @jit_compile_from_str_wo_ctx() -> !tf_framework.jit_callable {
  %callable = tf_framework.jit_compile_from_str "placeholder" {
      architectures = ["sm_123", "sm_456"], tileSizes = [1, 2, 3],
      unrollFactors = [4], maxSupportedRank = 3 : i64, enableFtz = false,
      index64Bit = false, cpuCodegen = false }
  func.return %callable : !tf_framework.jit_callable
}

// CHECK-LABEL: func @jit_compile_from_str
func.func @jit_compile_from_str(%ctx : !tf_framework.op_kernel_context)
    -> !tf_framework.jit_callable {
  %callable = tf_framework.jit_compile_from_str %ctx , "placeholder" {
      architectures = ["sm_123", "sm_456"], tileSizes = [1, 2, 3],
      unrollFactors = [4], maxSupportedRank = 3 : i64, enableFtz = false,
      index64Bit = false, cpuCodegen = false }
  func.return %callable : !tf_framework.jit_callable
}

// CHECK-LABEL: func @jit_execute_wo_ctx
func.func @jit_execute_wo_ctx(%callable : !tf_framework.jit_callable,
    %arg : tensor<2x?xf32>) -> tensor<2x?xf32> {
  %0 = tf_framework.jit_execute %callable(%arg)
      : tensor<2x?xf32> -> tensor<2x?xf32>
  func.return %0 : tensor<2x?xf32>
}

// CHECK-LABEL: func @jit_execute
func.func @jit_execute(%ctx : !tf_framework.op_kernel_context,
    %callable : !tf_framework.jit_callable, %arg : tensor<2x?xf32>)
    -> tensor<2x?xf32> {
  %0 = tf_framework.jit_execute ctx(%ctx) %callable(%arg)
      : tensor<2x?xf32> -> tensor<2x?xf32>
  func.return %0 : tensor<2x?xf32>
}
