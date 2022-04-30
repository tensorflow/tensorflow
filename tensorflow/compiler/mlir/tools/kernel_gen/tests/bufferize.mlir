// RUN: kernel-gen-opt %s --split-input-file  --mlir-print-op-generic |\
// RUN: mlir-hlo-opt --allow-unregistered-dialect \
// RUN:   --computeop-and-func-bufferize |\
// RUN: kernel-gen-opt \
// RUN:    --kernelgen-final-bufferize | FileCheck %s
// RUN: kernel-gen-opt %s --split-input-file  --mlir-print-op-generic |\
// RUN: mlir-hlo-opt --allow-unregistered-dialect \
// RUN:   --computeop-and-func-bufferize |\
// RUN: kernel-gen-opt \
// RUN:    --kernelgen-final-bufferize --promote-buffers-to-stack | FileCheck %s

// CHECK-LABEL: @jit_execute
// CHECK-SAME: (%[[F:.*]]: !tf_framework.jit_callable, %[[ARG:.*]]: memref<*xf32>) -> memref<*xf32>
func.func @jit_execute(%f : !tf_framework.jit_callable, %arg : tensor<*xf32>)
    -> tensor<*xf32> {
  // CHECK: %[[RES:.*]] = tf_framework.jit_execute %[[F]](%[[ARG]]) : memref<*xf32> -> memref<*xf32>
  // CHECK: return %[[RES]] : memref<*xf32>
  %0 = tf_framework.jit_execute %f(%arg) : tensor<*xf32> -> tensor<*xf32>
  func.return %0 : tensor<*xf32>
}
