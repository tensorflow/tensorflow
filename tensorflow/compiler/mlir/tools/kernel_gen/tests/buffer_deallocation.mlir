// RUN: kernel-gen-opt %s --buffer-deallocation | FileCheck %s

// CHECK-LABEL: @jit_execute_allocation
// CHECK-SAME:  %[[CTX:.*]]: !tf_framework.op_kernel_context, %[[ARG:.*]]: memref<*xf32>, %[[PRED:.*]]: i1, %[[CALLABLE:.*]]: !tf_framework.jit_callable, %[[SIZE:.*]]: index, %[[SHAPE:.*]]: memref<?xindex>
func.func @jit_execute_allocation(%ctx: !tf_framework.op_kernel_context,
    %arg: memref<*xf32>, %pred: i1, %callable: !tf_framework.jit_callable,
    %size: index, %shape: memref<?xindex>) -> memref<*xf32> {
  // CHECK:   %[[RES:.*]] = scf.if %[[PRED]]
  // CHECK:     %[[JIT_EXECUTE:.*]] = tf_framework.jit_execute ctx(%[[CTX]]) %[[CALLABLE]](%[[ARG]])
  // CHECK:     %[[INNER_RES:.*]] = bufferization.clone %[[JIT_EXECUTE]]
  // CHECK:     tf_framework.dealloc(%[[CTX]], %[[JIT_EXECUTE]])
  // CHECK:     scf.yield %[[INNER_RES]]
  // CHECK:   else
  // CHECK:     %[[ALLOC:.*]] = tf_framework.alloc(%[[CTX]], %[[SIZE]])
  // CHECK:     %[[RESHAPE:.*]] = memref.reshape %[[ALLOC]](%[[SHAPE]])
  // CHECK:     %[[INNER_RES:.*]] = bufferization.clone %[[RESHAPE]]
  // CHECK:     tf_framework.dealloc(%[[CTX]], %[[ALLOC]])
  // CHECK:     scf.yield %[[INNER_RES]]
  // CHECK:   return %[[RES]]
  %res = scf.if %pred -> (memref<*xf32>) {
    %inner_res = tf_framework.jit_execute ctx(%ctx) %callable(%arg)
        : memref<*xf32> -> memref<*xf32>
    scf.yield %inner_res : memref<*xf32>
  } else {
    %alloc = tf_framework.alloc(%ctx, %size) : memref<?xf32>
    %inner_res = memref.reshape %alloc(%shape)
        : (memref<?xf32>, memref<?xindex>) -> memref<*xf32>
    scf.yield %inner_res : memref<*xf32>
  }
  return %res : memref<*xf32>
}
