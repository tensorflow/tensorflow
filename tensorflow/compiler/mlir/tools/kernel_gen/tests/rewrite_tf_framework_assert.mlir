// RUN: kernel-gen-opt %s -rewrite-tf-framework-assert -split-input-file |\
// RUN: FileCheck %s

// CHECK-LABEL: func @assert(
// CHECK-SAME: [[CTX:%.*]]: !tf_framework.op_kernel_context
func @assert(%ctx: !tf_framework.op_kernel_context)
       -> (memref<*xf32>, memref<*xi32>) attributes {tf_entry} {
  %true = constant true
  tf_framework.assert %ctx, %true, INVALID_ARGUMENT, "the one and only"
  %buf_f32 = tf_framework.alloc(%ctx) : memref<2xf32>
  %unranked_f32 = memref.cast %buf_f32 : memref<2xf32> to memref<*xf32>
  %buf_i32 = tf_framework.alloc(%ctx) : memref<3xi32>
  %unranked_i32 = memref.cast %buf_i32 : memref<3xi32> to memref<*xi32>
  return %unranked_f32, %unranked_i32 : memref<*xf32>, memref<*xi32>
}
// CHECK:   [[TRUE:%.*]] = constant true
// CHECK:   cond_br [[TRUE]], ^bb1, ^bb2
// CHECK: ^bb1:
// CHECK:   [[BUF_F32:%.*]] = tf_framework.alloc([[CTX]]) : memref<2xf32>
// CHECK:   [[OUT_F32:%.*]] = memref.cast [[BUF_F32]]
// CHECK:   [[BUF_I32:%.*]] = tf_framework.alloc([[CTX]]) : memref<3xi32>
// CHECK:   [[OUT_I32:%.*]] = memref.cast [[BUF_I32]]
// CHECK:   return [[OUT_F32]], [[OUT_I32]] : memref<*xf32>, memref<*xi32>
// CHECK: ^bb2:
// CHECK:   tf_framework.report_error [[CTX]], INVALID_ARGUMENT,
// CHECK-SAME: "the one and only"
// CHECK:   [[NULL_F32:%.*]] = tf_framework.null_memref : memref<*xf32>
// CHECK:   [[NULL_I32:%.*]] = tf_framework.null_memref : memref<*xi32>
// CHECK:   return [[NULL_F32]], [[NULL_I32]] : memref<*xf32>, memref<*xi32>

// -----

// CHECK-LABEL: func @double_assert(
// CHECK-SAME: [[CTX:%.*]]: !tf_framework.op_kernel_context
func @double_assert(%ctx: !tf_framework.op_kernel_context)
       -> memref<*xf32> attributes {tf_entry} {
  %true = constant true
  %false = constant false
  tf_framework.assert %ctx, %true, INVALID_ARGUMENT, "first assertion"
  tf_framework.assert %ctx, %false, INVALID_ARGUMENT, "second assertion"
  %buf = tf_framework.alloc(%ctx) : memref<2xf32>
  %unranked_buf = memref.cast %buf : memref<2xf32> to memref<*xf32>
  return %unranked_buf : memref<*xf32>
}
// CHECK:   [[TRUE:%.*]] = constant true
// CHECK:   [[FALSE:%.*]] = constant false
// CHECK:   cond_br [[TRUE]], ^bb1, ^bb3
// CHECK: ^bb1:
// CHECK:   cond_br [[FALSE]], ^bb2, ^bb4
// CHECK: ^bb2:
// CHECK:   [[BUF:%.*]] = tf_framework.alloc([[CTX]]) : memref<2xf32>
// CHECK:   [[OUT:%.*]] = memref.cast [[BUF]]
// CHECK:   return [[OUT]] : memref<*xf32>
// CHECK: ^bb3:
// CHECK:   tf_framework.report_error [[CTX]], INVALID_ARGUMENT,
// CHECK-SAME: "first assertion"
// CHECK:   [[NULL:%.*]] = tf_framework.null_memref : memref<*xf32>
// CHECK:   return [[NULL]] : memref<*xf32>
// CHECK: ^bb4:
// CHECK:   tf_framework.report_error [[CTX]], INVALID_ARGUMENT,
// CHECK-SAME: "second assertion"
// CHECK:   [[NULL:%.*]] = tf_framework.null_memref : memref<*xf32>
// CHECK:   return [[NULL]] : memref<*xf32>
