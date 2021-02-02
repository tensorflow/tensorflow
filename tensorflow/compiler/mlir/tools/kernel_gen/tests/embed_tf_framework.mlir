// RUN: kernel-gen-opt %s -embed-tf-framework-func-and-alloc \
// RUN:   -embed-tf-framework-assert -split-input-file | \
// RUN: FileCheck %s

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

// -----

// CHECK-LABEL: func @assert(
// CHECK-SAME: [[CTX:%.*]]: !tf_framework.op_kernel_context
func @assert(%arg0: !tf_framework.op_kernel_context)
       -> (memref<*xf32>, memref<*xi32>) attributes {tf_entry} {
  %true = constant true
  assert %true, "the one and only"
  %buf_f32 = alloc() : memref<2xf32>
  %unranked_f32 = memref_cast %buf_f32 : memref<2xf32> to memref<*xf32>
  %buf_i32 = alloc() : memref<3xi32>
  %unranked_i32 = memref_cast %buf_i32 : memref<3xi32> to memref<*xi32>
  return %unranked_f32, %unranked_i32 : memref<*xf32>, memref<*xi32>
}
// CHECK:   [[TRUE:%.*]] = constant true
// CHECK:   cond_br [[TRUE]], ^bb1, ^bb2
// CHECK: ^bb1:
// CHECK:   [[BUF_F32:%.*]] = tf_framework.alloc([[CTX]]) : memref<2xf32>
// CHECK:   [[OUT_F32:%.*]] = memref_cast [[BUF_F32]]
// CHECK:   [[BUF_I32:%.*]] = tf_framework.alloc([[CTX]]) : memref<3xi32>
// CHECK:   [[OUT_I32:%.*]] = memref_cast [[BUF_I32]]
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
func @double_assert(%arg0: !tf_framework.op_kernel_context)
       -> memref<*xf32> attributes {tf_entry} {
  %true = constant true
  %false = constant false
  assert %true, "first assertion"
  assert %false, "second assertion"
  %buf = alloc() : memref<2xf32>
  %unranked_buf = memref_cast %buf : memref<2xf32> to memref<*xf32>
  return %unranked_buf : memref<*xf32>
}
// CHECK:   [[TRUE:%.*]] = constant true
// CHECK:   [[FALSE:%.*]] = constant false
// CHECK:   cond_br [[TRUE]], ^bb1, ^bb3
// CHECK: ^bb1:
// CHECK:   cond_br [[FALSE]], ^bb2, ^bb4
// CHECK: ^bb2:
// CHECK:   [[BUF:%.*]] = tf_framework.alloc([[CTX]]) : memref<2xf32>
// CHECK:   [[OUT:%.*]] = memref_cast [[BUF]]
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
