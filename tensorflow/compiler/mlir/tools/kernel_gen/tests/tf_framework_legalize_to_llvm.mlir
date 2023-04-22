// RUN: kernel-gen-opt %s -tf-kernel-to-llvm -split-input-file | FileCheck %s

// CHECK: llvm.func @_mlir_ciface_tf_alloc
// CHECK-SAME:  (!llvm.ptr<i8>, i64, i64, i32, i32, !llvm.ptr<i32>) -> !llvm.ptr<i8>

// CHECK-LABEL: llvm.func @alloc(
// CHECK-SAME:    [[TF_CTX:%.*]]: !llvm.ptr<i8>,
// CHECK-SAME:    [[SIZE_0:%.*]]: i64,
// CHECK-SAME:    [[SIZE_2:%.*]]: i64) -> [[DESC_TY:!.*]] {
func @alloc(%ctx: !tf_framework.op_kernel_context,
                %size_0 : index , %size_2 : index) -> memref<?x10x?xf32> {
  %buf = tf_framework.alloc(%ctx, %size_0, %size_2) : memref<?x10x?xf32>
  std.return %buf : memref<?x10x?xf32>
}
// Compute number of elements.
// CHECK: [[SIZE_1:%.*]] = llvm.mlir.constant(10 : index) : i64
// CHECK: [[NUM_ELEM_0:%.*]] = llvm.mul [[SIZE_0]], [[SIZE_1]] : i64
// CHECK: [[NUM_ELEMS:%.*]] = llvm.mul [[NUM_ELEM_0]], [[SIZE_2]] : i64

// Compute the size of an individual element.
// CHECK: [[NULL:%.*]] = llvm.mlir.null : !llvm.ptr<f32>
// CHECK: [[C1:%.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK: [[GEP:%.*]] = llvm.getelementptr [[NULL]]{{\[}}[[C1]]]
// CHECK-SAME:            (!llvm.ptr<f32>, i64) -> !llvm.ptr<f32>
// CHECK: [[SIZE_OF_FLOAT:%.*]] = llvm.ptrtoint [[GEP]]
// CHECK-SAME:            !llvm.ptr<f32> to i64

// Compute output index (-1) and candidate indices (0, NULL).
// CHECK: [[OUTPUT_INDEX:%.*]] = llvm.mlir.constant(-1 : i32) : i32
// CHECK-NEXT: [[NUM_CANDIDATES:%.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT: [[CANDIDATES_PTR:%.*]] = llvm.mlir.null : !llvm.ptr<i32>

// Allocate memory.
// CHECK: [[BYTES_PTR:%.*]] = llvm.call @{{.*}}([[TF_CTX]], [[NUM_ELEMS]],
// CHECK-SAME: [[SIZE_OF_FLOAT]], [[OUTPUT_INDEX]], [[NUM_CANDIDATES]],
// CHECK-SAME: [[CANDIDATES_PTR]])

// Build memref descriptor.
// CHECK: [[DESC_0:%.*]] = llvm.mlir.undef : [[DESC_TY]]

// Set pointers and offset.
// CHECK: [[FLOAT_PTR:%.*]] = llvm.bitcast [[BYTES_PTR]]
// CHECK-SAME:                  !llvm.ptr<i8> to !llvm.ptr<f32>
// CHECK: [[DESC_1:%.*]] = llvm.insertvalue [[FLOAT_PTR]], [[DESC_0]][0]
// CHECK: [[DESC_2:%.*]] = llvm.insertvalue [[FLOAT_PTR]], [[DESC_1]][1]
// CHECK: [[C0:%.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK: [[DESC_3:%.*]] = llvm.insertvalue [[C0]], [[DESC_2]][2] : [[DESC_TY]]

// Set sizes and strides.
// CHECK: [[STRIDE_2:%.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK: [[DESC_4:%.*]] = llvm.insertvalue [[SIZE_2]], [[DESC_3]][3, 2]
// CHECK: [[DESC_5:%.*]] = llvm.insertvalue [[STRIDE_2]], [[DESC_4]][4, 2]
// CHECK: [[STRIDE_1:%.*]] = llvm.mul [[STRIDE_2]], [[SIZE_2]] : i64
// CHECK: [[DESC_6:%.*]] = llvm.insertvalue [[SIZE_1]], [[DESC_5]][3, 1]
// CHECK: [[DESC_7:%.*]] = llvm.insertvalue [[STRIDE_1]], [[DESC_6]][4, 1]
// CHECK: [[STRIDE_0:%.*]] = llvm.mul [[STRIDE_1]], [[SIZE_1]] : i64
// CHECK: [[DESC_8:%.*]] = llvm.insertvalue [[SIZE_0]], [[DESC_7]][3, 0]
// CHECK: [[DESC_9:%.*]] = llvm.insertvalue [[STRIDE_0]], [[DESC_8]][4, 0]
// CHECK: llvm.return [[DESC_9]] : [[DESC_TY]]

// -----

// CHECK: llvm.func @_mlir_ciface_tf_dealloc(!llvm.ptr<i8>, !llvm.ptr<i8>)

// CHECK-LABEL: llvm.func @dealloc(
// CHECK-SAME:    [[TF_CTX:%.*]]: !llvm.ptr<i8>,
func @dealloc(%ctx: !tf_framework.op_kernel_context,
                  %memref : memref<?x10xf32>) {
  tf_framework.dealloc(%ctx, %memref) : memref<?x10xf32>
  return
}
// Extract allocated ptr from the memref descriptor.
// CHECK: %{{.*}} = llvm.mlir.undef : [[DESC_TY:!.*]]
// CHECK: [[FLOAT_PTR:%.*]] = llvm.extractvalue %{{.*}}[0] : [[DESC_TY]]
// CHECK-NEXT: [[VOID_PTR:%.*]] = llvm.bitcast [[FLOAT_PTR]]
// CHECK-SAME:                   !llvm.ptr<f32> to !llvm.ptr<i8>

// Deallocate.
// CHECK: llvm.call @_mlir_ciface_tf_dealloc(
// CHECK-SAME: [[TF_CTX]], [[VOID_PTR]]) : (!llvm.ptr<i8>, !llvm.ptr<i8>) -> ()

// -----

// CHECK-LABEL: llvm.func @_mlir_ciface_tf_report_error(!llvm.ptr<i8>, i32, !llvm.ptr<i8>)
// CHECK: llvm.mlir.global internal constant [[MSG_CONST:@error_message_[0-9]+]]("Everything is awesome")

func @report_error(%ctx: !tf_framework.op_kernel_context) {
  tf_framework.report_error %ctx, "INVALID_ARGUMENT", "Everything is awesome" loc(unknown)
  return
}
// CHECK:     llvm.func @report_error([[CTX:%.*]]: !llvm.ptr<i8>)
// CHECK-NEXT:  [[ADDR:%.*]] = llvm.mlir.addressof [[MSG_CONST]]
// CHECK:       [[MSG:%.*]] = llvm.getelementptr [[ADDR]]
// CHECK:       [[CODE:%.*]] = llvm.mlir.constant({{.*}}) : i32
// CHECK:       llvm.call @{{.*}}_tf_report_error([[CTX]], [[CODE]], [[MSG]])

// ----

// CHECK-LABEL: llvm.func @unranked_null_memref()
func @unranked_null_memref() {
  %null = tf_framework.null_memref : memref<*xf32>
  return
}
// CHECK: [[C0:%.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK: [[DESC_0:%.*]] = llvm.mlir.undef : !llvm.struct<(i64, ptr<i8>)>
// CHECK: [[DESC_1:%.*]] = llvm.insertvalue [[C0]], [[DESC_0]][0]
// CHECK: [[PTR:%.*]] = llvm.alloca {{.*}} x i8
// CHECK: [[DESC_2:%.*]] = llvm.insertvalue [[PTR]], [[DESC_1]][1]

// ----

// CHECK-LABEL: llvm.func @ranked_null_memref()
func @ranked_null_memref() {
  %null = tf_framework.null_memref : memref<2x?xf32>
  return
}
// CHECK: %[[C0:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT: %[[C1:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT: %[[C2:.*]] = llvm.mlir.constant(2 : index) : i64
// CHECK-NEXT: %[[C1_:.*]] = llvm.mlir.constant(1 : index) : i64

// CHECK: llvm.mlir.null
// CHECK: %[[NULL:.*]] = llvm.mlir.null : !llvm.ptr<f32>
// CHECK-NEXT: %[[DESC_0:.*]] = llvm.mlir.undef :
// CHECK-SAME:   !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[DESC_1:.*]] = llvm.insertvalue %[[NULL]], %[[DESC_0]][0]
// CHECK-NEXT: %[[DESC_2:.*]] = llvm.insertvalue %[[NULL]], %[[DESC_1]][1]
// CHECK-NEXT: %[[DESC_3:.*]] = llvm.insertvalue %[[C0]], %[[DESC_2]][2]
// CHECK-NEXT: %[[DESC_4:.*]] = llvm.insertvalue %[[C2]], %[[DESC_3]][3, 0]
// CHECK-NEXT: %[[DESC_5:.*]] = llvm.insertvalue %[[C1]], %[[DESC_4]][4, 0]
// CHECK-NEXT: %[[DESC_6:.*]] = llvm.insertvalue %[[C1]], %[[DESC_5]][3, 1]
// CHECK-NEXT: %[[DESC_7:.*]] = llvm.insertvalue %[[C1_]], %[[DESC_6]][4, 1]

// ----

// CHECK-LABEL: llvm.func @is_valid_memref
func @is_valid_memref(%buf: memref<?xf32>) -> i1 {
  %pred = tf_framework.is_valid_memref(%buf) : memref<?xf32> -> i1
  return %pred : i1
}
// CHECK: %[[MEMREF:.*]] = llvm.insertvalue %{{.*}}, %{{.*}}[4, 0]

// CHECK-NEXT: %[[IS_EMPTY:.*]] = llvm.mlir.constant(false) : i1
// CHECK-NEXT: %[[C0:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT: %[[SIZE:.*]] = llvm.extractvalue %[[MEMREF]][3, 0]
// CHECK-NEXT: %[[IS_ZERO:.*]] = llvm.icmp "eq" %[[SIZE]], %[[C0]] : i64
// CHECK-NEXT: %[[IS_EMPTY_:.*]] =  llvm.or %[[IS_EMPTY]], %[[IS_ZERO]] : i1

// CHECK-NEXT: %[[PTR_F32:.*]] = llvm.extractvalue %[[MEMREF]][0]
// CHECK-NEXT: %[[VOID_PTR:.*]] = llvm.bitcast %[[PTR_F32]] : !llvm.ptr<f32> to !llvm.ptr<i8>
// CHECK-NEXT: %[[NULL_PTR:.*]] = llvm.mlir.null : !llvm.ptr<i8>
// CHECK-NEXT: %[[NOT_NULL:.*]] = llvm.icmp "ne" %[[VOID_PTR]], %[[NULL_PTR]]

// CHECK-NEXT: %[[PRED:.*]] = llvm.or %[[NOT_NULL]], %[[IS_EMPTY_]]  : i1
// CHECK-NEXT: llvm.return %[[PRED]]

// -----

// CHECK-LABEL: llvm.func @_mlir_ciface_tf_jit_compile(!llvm.ptr<i8>, !llvm.ptr<i8>) -> !llvm.ptr<i8>
// CHECK: llvm.mlir.global internal constant @[[CODE:jit_module_code_[0-9]+]]("placeholder")

// CHECK: @jit_compile_from_str(%[[CTX:.*]]: !llvm.ptr<i8>)
func @jit_compile_from_str(%ctx: !tf_framework.op_kernel_context)
    -> !tf_framework.jit_callable {
  // CHECK: %[[ADDR:.*]] = llvm.mlir.addressof @[[CODE]]
  // CHECK: %[[C0:.*]] = llvm.mlir.constant(0 : index)
  // CHECK: %[[CODE_PTR:.*]] = llvm.getelementptr %[[ADDR]][%[[C0]], %[[C0]]]
  // CHECK: %[[RES:.*]] = llvm.call @_mlir_ciface_tf_jit_compile(%[[CTX]], %[[CODE_PTR]])
  // CHECK: llvm.return %[[RES]]
  %0 = tf_framework.jit_compile_from_str %ctx, "placeholder"
  return %0 : !tf_framework.jit_callable
}

// -----

// CHECK-LABEL: llvm.func @_mlir_ciface_tf_jit_execute(!llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.ptr<i8>, i64, !llvm.ptr<i8>)

// CHECK:      @jit_execute
// CHECK-SAME: (%[[CTX:.*]]: !llvm.ptr<i8>, %[[RES:.*]]: !llvm.ptr<i8>, %[[RANK:.*]]: i64, %[[ARG_DESCR:.*]]: !llvm.ptr<i8>)
func @jit_execute(%ctx: !tf_framework.op_kernel_context,
    %callable : !tf_framework.jit_callable, %arg : memref<*xf32>)
    -> memref<*xf32> {
  // CHECK: %[[T0:.*]] = llvm.mlir.undef
  // CHECK: %[[T1:.*]] = llvm.insertvalue %arg2, %[[T0]][0]
  // CHECK: %[[ARG:.*]] = llvm.insertvalue %arg3, %[[T1]][1]
  // CHECK: %[[C1:.*]] = llvm.mlir.constant(1 : index)
  // CHECK: %[[RESULT_PTR:.*]] = llvm.alloca %[[C1]] x !llvm.struct<(i64, ptr<i8>)>
  // CHECK: %[[RESULT_PTR_:.*]] = llvm.bitcast %[[RESULT_PTR]]
  // CHECK: %[[RANK:.*]] = llvm.extractvalue %[[ARG]][0]
  // CHECK: %[[ARG_DESCR:.*]] = llvm.extractvalue %[[ARG]][1]
  // CHECK: llvm.call @_mlir_ciface_tf_jit_execute(%arg0, %arg1, %[[RESULT_PTR_]], %[[RANK]], %[[ARG_DESCR]])
  // CHECK: %[[RESULT:.*]] = llvm.load %[[RESULT_PTR]]

  // Copy unranked memref descriptor to stack-allocated memory.
  // ...
  // CHECK: %[[RESULT_DESCR_SIZE:.*]] = llvm.add %13, %17
  // CHECK: %[[FALSE:.*]] = llvm.mlir.constant(false)
  // CHECK: %[[STACK_RESULT_DESCR:.*]] = llvm.alloca %[[RESULT_DESCR_SIZE]] x i8
  // CHECK: %[[RESULT_DESCR:.*]] = llvm.extractvalue %[[RESULT]][1]
  // CHECK: "llvm.intr.memcpy"(%[[STACK_RESULT_DESCR]], %[[RESULT_DESCR]], %[[RESULT_DESCR_SIZE]], %[[FALSE]])
  // CHECK: llvm.call @free(%[[RESULT_DESCR]])
  // CHECK: %[[T0:.*]] = llvm.mlir.undef
  // CHECK: %[[RANK:.*]] = llvm.extractvalue %[[RESULT]][0]
  // CHECK: %[[T1:.*]] = llvm.insertvalue %[[RANK]], %[[T0]][0]
  // CHECK: %[[RESULT:.*]] = llvm.insertvalue %[[STACK_RESULT_DESCR]], %[[T1]][1]

  // Copy unranked memref descriptor to heap-allocated memory for return.
  // ...
  // CHECK: %[[RESULT_DESCR_SIZE:.*]] = llvm.add %30, %34
  // CHECK: %[[FALSE:.*]] = llvm.mlir.constant(false)
  // CHECK: %[[HEAP_RESUKT_DESCR:.*]] = llvm.call @malloc(%[[RESULT_DESCR_SIZE]])
  // CHECK: %[[STACK_RESULT_DESCR:.*]] = llvm.extractvalue %[[RESULT]][1]
  // CHECK: "llvm.intr.memcpy"(%[[HEAP_RESUKT_DESCR]], %[[STACK_RESULT_DESCR]], %[[RESULT_DESCR_SIZE]], %[[FALSE]])
  // CHECK: %[[T0:.*]] = llvm.mlir.undef
  // CHECK: %[[RANK:.*]] = llvm.extractvalue %[[RESULT]][0]
  // CHECK: %[[T1:.*]] = llvm.insertvalue %[[RANK]], %[[T0]][0]
  // CHECK: %[[RESULT:.*]] = llvm.insertvalue %[[HEAP_RESUKT_DESCR]], %[[T1]][1]
  // CHECK: llvm.return %[[RESULT]]
  %0 = tf_framework.jit_execute ctx = %ctx, %callable(%arg)
      : memref<*xf32> -> memref<*xf32>
  return %0 : memref<*xf32>
}
