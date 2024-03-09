// RUN: kernel-gen-opt %s -tf-kernel-to-llvm -split-input-file | FileCheck %s

// CHECK: llvm.func @_mlir_ciface_tf_alloc
// CHECK-SAME:  (!llvm.ptr, i64, i64, i32, i32, !llvm.ptr) -> !llvm.ptr

// CHECK-LABEL: llvm.func @alloc(
// CHECK-SAME:    [[TF_CTX:%.*]]: !llvm.ptr,
// CHECK-SAME:    [[SIZE_0:%.*]]: i64,
// CHECK-SAME:    [[SIZE_2:%.*]]: i64) -> [[DESC_TY:!.*]] {
func.func @alloc(%ctx: !tf_framework.op_kernel_context,
                %size_0 : index , %size_2 : index) -> memref<?x10x?xf32> {
  %buf = tf_framework.alloc(%ctx, %size_0, %size_2) : memref<?x10x?xf32>
  func.return %buf : memref<?x10x?xf32>
}
// Compute number of elements.
// CHECK: [[SIZE_1A:%.*]] = llvm.mlir.constant(10 : index) : i64
// CHECK: [[SIZE_1B:%.*]] = llvm.mlir.constant(10 : index) : i64
// CHECK: [[NUM_ELEM_0:%.*]] = llvm.mul [[SIZE_0]], [[SIZE_1B]] : i64
// CHECK: [[NUM_ELEMS:%.*]] = llvm.mul [[NUM_ELEM_0]], [[SIZE_2]] : i64

// Compute the size of an individual element.
// CHECK: [[NULL:%.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK: [[GEP:%.*]] = llvm.getelementptr [[NULL]]{{\[}}1]
// CHECK-SAME:            (!llvm.ptr) -> !llvm.ptr, f32
// CHECK: [[SIZE_OF_FLOAT:%.*]] = llvm.ptrtoint [[GEP]]
// CHECK-SAME:            !llvm.ptr to i64

// Compute output index (-1) and candidate indices (0, NULL).
// CHECK: [[OUTPUT_INDEX:%.*]] = llvm.mlir.constant(-1 : i32) : i32
// CHECK-NEXT: [[NUM_CANDIDATES:%.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT: [[CANDIDATES_PTR:%.*]] = llvm.mlir.zero : !llvm.ptr

// Allocate memory.
// CHECK: [[BYTES_PTR:%.*]] = llvm.call @{{.*}}([[TF_CTX]], [[NUM_ELEMS]],
// CHECK-SAME: [[SIZE_OF_FLOAT]], [[OUTPUT_INDEX]], [[NUM_CANDIDATES]],
// CHECK-SAME: [[CANDIDATES_PTR]])

// Build memref descriptor.
// CHECK: [[DESC_0:%.*]] = llvm.mlir.undef : [[DESC_TY]]

// Set pointers and offset.
// CHECK: [[DESC_1:%.*]] = llvm.insertvalue [[BYTES_PTR]], [[DESC_0]][0]
// CHECK: [[DESC_2:%.*]] = llvm.insertvalue [[BYTES_PTR]], [[DESC_1]][1]
// CHECK: [[C0:%.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK: [[DESC_3:%.*]] = llvm.insertvalue [[C0]], [[DESC_2]][2] : [[DESC_TY]]

// Set sizes and strides.
// CHECK: [[STRIDE_2:%.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK: [[DESC_4:%.*]] = llvm.insertvalue [[SIZE_2]], [[DESC_3]][3, 2]
// CHECK: [[DESC_5:%.*]] = llvm.insertvalue [[STRIDE_2]], [[DESC_4]][4, 2]
// CHECK: [[STRIDE_1:%.*]] = llvm.mul [[STRIDE_2]], [[SIZE_2]] : i64
// CHECK: [[DESC_6:%.*]] = llvm.insertvalue [[SIZE_1A]], [[DESC_5]][3, 1]
// CHECK: [[DESC_7:%.*]] = llvm.insertvalue [[STRIDE_1]], [[DESC_6]][4, 1]
// CHECK: [[STRIDE_0:%.*]] = llvm.mul [[STRIDE_1]], [[SIZE_1A]] : i64
// CHECK: [[DESC_8:%.*]] = llvm.insertvalue [[SIZE_0]], [[DESC_7]][3, 0]
// CHECK: [[DESC_9:%.*]] = llvm.insertvalue [[STRIDE_0]], [[DESC_8]][4, 0]
// CHECK: llvm.return [[DESC_9]] : [[DESC_TY]]

// -----

// CHECK: llvm.func @_mlir_ciface_tf_dealloc(!llvm.ptr, !llvm.ptr)

// CHECK-LABEL: llvm.func @dealloc(
// CHECK-SAME:    [[TF_CTX:%[a-z0-9]*]]: !llvm.ptr
func.func @dealloc(%ctx: !tf_framework.op_kernel_context,
                  %memref : memref<?x10xf32>) {
  tf_framework.dealloc(%ctx, %memref) : memref<?x10xf32>
  func.return
}
// Extract allocated ptr from the memref descriptor.
// CHECK: %{{.*}} = llvm.mlir.undef : [[DESC_TY:!.*]]
// CHECK: [[FLOAT_PTR:%.*]] = llvm.extractvalue %{{.*}}[0] : [[DESC_TY]]

// Deallocate.
// CHECK: llvm.call @_mlir_ciface_tf_dealloc(
// CHECK-SAME: [[TF_CTX]], [[FLOAT_PTR]]) : (!llvm.ptr, !llvm.ptr) -> ()

// -----

// CHECK-LABEL: llvm.func @_mlir_ciface_tf_report_error(!llvm.ptr, i32, !llvm.ptr)
// CHECK: llvm.mlir.global internal constant [[MSG_CONST:@error_message_[0-9]+]]("Everything is awesome\00")

func.func @report_error(%ctx: !tf_framework.op_kernel_context) {
  tf_framework.report_error %ctx, "INVALID_ARGUMENT", "Everything is awesome" loc(unknown)
  func.return
}
// CHECK:     llvm.func @report_error([[CTX:%.*]]: !llvm.ptr)
// CHECK-NEXT:  [[ADDR:%.*]] = llvm.mlir.addressof [[MSG_CONST]]
// CHECK:       [[MSG:%.*]] = llvm.getelementptr [[ADDR]]
// CHECK:       [[CODE:%.*]] = llvm.mlir.constant({{.*}}) : i32
// CHECK:       llvm.call @{{.*}}_tf_report_error([[CTX]], [[CODE]], [[MSG]])

// -----

// CHECK-LABEL: llvm.func @unranked_null_memref()
func.func @unranked_null_memref() {
  %null = tf_framework.null_memref : memref<*xf32>
  func.return
}
// CHECK: [[C0:%.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK: [[DESC_0:%.*]] = llvm.mlir.undef : !llvm.struct<(i64, ptr)>
// CHECK: [[DESC_1:%.*]] = llvm.insertvalue [[C0]], [[DESC_0]][0]
// CHECK: [[PTR:%.*]] = llvm.alloca {{.*}} x i8
// CHECK: [[DESC_2:%.*]] = llvm.insertvalue [[PTR]], [[DESC_1]][1]

// -----

// CHECK-LABEL: llvm.func @ranked_null_memref()
func.func @ranked_null_memref() {
  %null = tf_framework.null_memref : memref<2x?xf32>
  func.return
}
// CHECK: %[[C0:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT: %[[C1:.*]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT: %[[C2:.*]] = llvm.mlir.constant(2 : index) : i64
// CHECK-NEXT: %[[C1_:.*]] = llvm.mlir.constant(1 : index) : i64

// CHECK: llvm.mlir.zero
// CHECK: %[[NULL:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT: %[[DESC_0:.*]] = llvm.mlir.undef :
// CHECK-SAME:   !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-NEXT: %[[DESC_1:.*]] = llvm.insertvalue %[[NULL]], %[[DESC_0]][0]
// CHECK-NEXT: %[[DESC_2:.*]] = llvm.insertvalue %[[NULL]], %[[DESC_1]][1]
// CHECK-NEXT: %[[DESC_3:.*]] = llvm.insertvalue %[[C0]], %[[DESC_2]][2]
// CHECK-NEXT: %[[DESC_4:.*]] = llvm.insertvalue %[[C2]], %[[DESC_3]][3, 0]
// CHECK-NEXT: %[[DESC_5:.*]] = llvm.insertvalue %[[C1]], %[[DESC_4]][4, 0]
// CHECK-NEXT: %[[DESC_6:.*]] = llvm.insertvalue %[[C1]], %[[DESC_5]][3, 1]
// CHECK-NEXT: %[[DESC_7:.*]] = llvm.insertvalue %[[C1_]], %[[DESC_6]][4, 1]

// -----

// CHECK-LABEL: llvm.func @is_valid_memref
func.func @is_valid_memref(%buf: memref<?xf32>) -> i1 {
  %pred = tf_framework.is_valid_memref(%buf) : memref<?xf32> -> i1
  func.return %pred : i1
}
// CHECK: %[[MEMREF:.*]] = llvm.insertvalue %{{.*}}, %{{.*}}[4, 0]

// CHECK-NEXT: %[[IS_EMPTY:.*]] = llvm.mlir.constant(false) : i1
// CHECK-NEXT: %[[C0:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT: %[[SIZE:.*]] = llvm.extractvalue %[[MEMREF]][3, 0]
// CHECK-NEXT: %[[IS_ZERO:.*]] = llvm.icmp "eq" %[[SIZE]], %[[C0]] : i64
// CHECK-NEXT: %[[IS_EMPTY_:.*]] =  llvm.or %[[IS_EMPTY]], %[[IS_ZERO]] : i1

// CHECK-NEXT: %[[PTR_F32:.*]] = llvm.extractvalue %[[MEMREF]][0]
// CHECK-NEXT: %[[NULL_PTR:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK-NEXT: %[[NOT_NULL:.*]] = llvm.icmp "ne" %[[PTR_F32]], %[[NULL_PTR]]

// CHECK-NEXT: %[[PRED:.*]] = llvm.or %[[NOT_NULL]], %[[IS_EMPTY_]]  : i1
// CHECK-NEXT: llvm.return %[[PRED]]

// -----

// CHECK-LABEL: llvm.func @_mlir_ciface_tf_jit_compile(!llvm.ptr, !llvm.ptr, i64, !llvm.ptr, i64, !llvm.ptr, i1, i1, i1) -> !llvm.ptr
// CHECK: llvm.mlir.global internal constant @[[CODE:jit_module_code_[0-9]+]]("placeholder\00")

// CHECK: @jit_compile_from_str(%[[CTX:.*]]: !llvm.ptr)
func.func @jit_compile_from_str(%ctx: !tf_framework.op_kernel_context)
    -> !tf_framework.jit_callable {
  // CHECK: %[[ADDR:.*]] = llvm.mlir.addressof @[[CODE]]
  // CHECK: %[[CODE_PTR:.*]] = llvm.getelementptr %[[ADDR]][0, 0]

  // Create stack-allocated array for the tile sizes.
  // CHECK: %[[NUM_TILE_SIZES:.*]] = llvm.mlir.constant(3 : i64)
  // CHECK: %[[TILE_SIZES:.*]] = llvm.alloca %[[NUM_TILE_SIZES]] x i64
  // CHECK: %[[C0:.*]] = llvm.mlir.constant(0 : i64)
  // CHECK: %[[PTR:.*]] = llvm.getelementptr %[[TILE_SIZES]][%[[C0]]]
  // CHECK: %[[C1:.*]] = llvm.mlir.constant(1 : i64)
  // CHECK: llvm.store %[[C1]], %[[PTR]]
  // CHECK: %[[C1:.*]] = llvm.mlir.constant(1 : i64)
  // CHECK: %[[PTR:.*]] = llvm.getelementptr %[[TILE_SIZES]][%[[C1]]]
  // CHECK: %[[C2:.*]] = llvm.mlir.constant(2 : i64)
  // CHECK: llvm.store %[[C2]], %[[PTR]]
  // CHECK: %[[C2:.*]] = llvm.mlir.constant(2 : i64)
  // CHECK: %[[PTR:.*]] = llvm.getelementptr %[[TILE_SIZES]][%[[C2]]]
  // CHECK: %[[C3:.*]] = llvm.mlir.constant(3 : i64)
  // CHECK: llvm.store %[[C3]], %[[PTR]]

  // Create stack-allocated array for the unroll factors.
  // CHECK: %[[NUM_UNROLL_FACTORS:.*]] = llvm.mlir.constant(1 : i64) : i64
  // CHECK: %[[UNROLL_FACTORS:.*]] = llvm.alloca %[[NUM_UNROLL_FACTORS]] x i64
  // CHECK: %[[C0:.*]] = llvm.mlir.constant(0 : i64)
  // CHECK: %[[PTR:.*]] = llvm.getelementptr %[[UNROLL_FACTORS]][%[[C0]]]
  // CHECK: %[[C4:.*]] = llvm.mlir.constant(4 : i64)
  // CHECK: llvm.store %[[C4]], %[[PTR]]

  // CHECK-DAG: %[[ENABLE_FTZ:.*]] = llvm.mlir.constant(false)
  // CHECK-DAG: %[[CPU_CODEGEN:.*]] = llvm.mlir.constant(false)
  // CHECK: %[[RES:.*]] = llvm.call @_mlir_ciface_tf_jit_compile
  // CHECK-SAME: %[[CTX]], %[[CODE_PTR]],
  // CHECK-SAME: %[[NUM_TILE_SIZES]], %[[TILE_SIZES]],
  // CHECK-SAME: %[[NUM_UNROLL_FACTORS]], %[[UNROLL_FACTORS]],
  // CHECK-SAME: %[[ENABLE_FTZ]], %[[CPU_CODEGEN]]
  // CHECK: llvm.return %[[RES]]
  %0 = tf_framework.jit_compile_from_str %ctx, "placeholder" {
      tileSizes = [1, 2, 3], unrollFactors = [4],
      enableFtz = false, index64Bit = false, cpuCodegen = false }
  func.return %0 : !tf_framework.jit_callable
}

// -----

// CHECK-LABEL: llvm.func @_mlir_ciface_tf_jit_execute(!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, !llvm.ptr)

// CHECK:      @jit_execute
// CHECK-SAME: (%[[CTX:.*]]: !llvm.ptr, %[[CALLABLE:.*]]: !llvm.ptr, %[[RANK:.*]]: i64, %[[ARG_DESCR:.*]]: !llvm.ptr)
func.func @jit_execute(%ctx: !tf_framework.op_kernel_context,
    %callable : !tf_framework.jit_callable, %arg : memref<*xf32>)
    -> memref<*xf32> {
  // CHECK: %[[T0:.*]] = llvm.mlir.undef
  // CHECK: %[[T1:.*]] = llvm.insertvalue %[[RANK]], %[[T0]][0]
  // CHECK: %[[ARG:.*]] = llvm.insertvalue %[[ARG_DESCR]], %[[T1]][1]
  // CHECK: %[[C1:.*]] = llvm.mlir.constant(1 : i64)
  // CHECK: %[[RESULT_PTR:.*]] = llvm.alloca %[[C1]] x !llvm.struct<(i64, ptr)>

  // Copy argument(s) to stack-allocated buffer.
  // CHECK: %[[NUM_ARGS:.*]] = llvm.mlir.constant(1 : i64)
  // CHECK: %[[ARGS_PTR:.*]] = llvm.alloca %[[NUM_ARGS]] x !llvm.struct<(i64, ptr)>
  // CHECK: %[[C0:.*]] = llvm.mlir.constant(0 : i64)
  // CHECK: %[[ARGS0_PTR:.*]] = llvm.getelementptr %[[ARGS_PTR]][%[[C0]]]
  // CHECK: llvm.store %[[ARG]], %[[ARGS0_PTR]]
  // CHECK: llvm.call @_mlir_ciface_tf_jit_execute(%[[CTX]], %[[CALLABLE]], %[[RESULT_PTR]], %[[NUM_ARGS]], %[[ARGS_PTR]])
  // CHECK: %[[RESULT:.*]] = llvm.load %[[RESULT_PTR]]

  // Copy unranked memref descriptor to stack-allocated memory.
  // ...
  // CHECK: %[[STACK_RESULT_DESCR:.*]] = llvm.alloca %[[RESULT_DESCR_SIZE:[0-9]*]] x i8
  // CHECK: %[[RESULT_DESCR:.*]] = llvm.extractvalue %[[RESULT]][1]
  // CHECK: "llvm.intr.memcpy"(%[[STACK_RESULT_DESCR]], %[[RESULT_DESCR]], %[[RESULT_DESCR_SIZE]]) <{isVolatile = false}>
  // CHECK: llvm.call @free(%[[RESULT_DESCR]])
  // CHECK: %[[T0:.*]] = llvm.mlir.undef
  // CHECK: %[[RANK:.*]] = llvm.extractvalue %[[RESULT]][0]
  // CHECK: %[[T1:.*]] = llvm.insertvalue %[[RANK]], %[[T0]][0]
  // CHECK: %[[RESULT:.*]] = llvm.insertvalue %[[STACK_RESULT_DESCR]], %[[T1]][1]

  // Copy unranked memref descriptor to heap-allocated memory for return.
  // ...
  // CHECK: %[[HEAP_RESULT_DESCR:.*]] = llvm.call @malloc(%[[RESULT_DESCR_SIZE:[0-9]*]])
  // CHECK: %[[STACK_RESULT_DESCR:.*]] = llvm.extractvalue %[[RESULT]][1]
  // CHECK: "llvm.intr.memcpy"(%[[HEAP_RESULT_DESCR]], %[[STACK_RESULT_DESCR]], %[[RESULT_DESCR_SIZE]]) <{isVolatile = false}>
  // CHECK: %[[T0:.*]] = llvm.mlir.undef
  // CHECK: %[[RANK:.*]] = llvm.extractvalue %[[RESULT]][0]
  // CHECK: %[[T1:.*]] = llvm.insertvalue %[[RANK]], %[[T0]][0]
  // CHECK: %[[RESULT:.*]] = llvm.insertvalue %[[HEAP_RESULT_DESCR]], %[[T1]][1]
  // CHECK: llvm.return %[[RESULT]]
  %0 = tf_framework.jit_execute ctx(%ctx) %callable(%arg)
      : memref<*xf32> -> memref<*xf32>
  func.return %0 : memref<*xf32>
}
