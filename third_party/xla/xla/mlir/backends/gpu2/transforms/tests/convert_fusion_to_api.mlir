// RUN: export MSAN_OPTIONS=intercept_strpbrk=0
// RUN: xla-gpu2-opt %s --xla-gpu2-convert-to-runtime=backend=streamexecutor   \
// RUN:                 --split-input-file                                     \
// RUN:   | FileCheck %s

func.func @fusion(
    %arg0: memref<12xi8>, %arg1: memref<12xi8>,
    %arg2: memref<12xi8> {lmhlo.output_index = dense<> : tensor<0xi64>}
) {
  %c0 = arith.constant 0 : index
  %view0 = memref.view %arg0[%c0][] : memref<12xi8> to memref<3xf32>
  %view1 = memref.view %arg1[%c0][] : memref<12xi8> to memref<3xf32>
  %view2 = memref.view %arg2[%c0][] : memref<12xi8> to memref<3xf32>
  "lmhlo.fusion"() ({
    %0 = bufferization.to_tensor %view0 : memref<3xf32>
    %1 = bufferization.to_tensor %view1 : memref<3xf32>
    %2 = mhlo.add %0, %1 : tensor<3xf32>
    memref.tensor_store %2, %view2 : memref<3xf32>
    "lmhlo.terminator"() : () -> ()
  }) : () -> ()
  "lmhlo.terminator"() : () -> ()
}

// CHECK-LABEL: func @fusion(
// CHECK:   %[[CTX:.*]]: !xla_gpu.execution_context,
// CHECK:   %[[ARG0:.*]]: tensor<12xi8>, %[[ARG1:.*]]: tensor<12xi8>,
// CHECK:   %[[ARG2:.*]]: tensor<12xi8> {lmhlo.output_index = {{.*}}}
// CHECK: ) {

// CHECK-DAG: %[[BUFFER0:.*]] = iree_input.tensor.export %[[ARG0]]
// CHECK-DAG: %[[BUFFER1:.*]] = iree_input.tensor.export %[[ARG1]]
// CHECK-DAG: %[[BUFFER2:.*]] = iree_input.tensor.export %[[ARG2]]
// CHECK-DAG: %[[TENSOR0:.*]] = iree_input.tensor.import %[[BUFFER0]]
// CHECK-DAG: %[[TENSOR1:.*]] = iree_input.tensor.import %[[BUFFER1]]
// CHECK-DAG: %[[TENSOR2:.*]] = iree_input.tensor.import %[[BUFFER2]]

// CHECK:   %[[KERNEL:.*]] = iree_input.global.load @[[KERNEL_NAME:.*]] :

// CHECK:   %[[C3:.*]] = arith.constant 3 : index
// CHECK:   %[[ARGS:.*]] = iree_input.list.create %[[C3]]
// CHECK-SAME: !iree_input.list<!iree_input.buffer_view>
// CHECK:   iree_input.list.set
// CHECK:   iree_input.list.set
// CHECK:   iree_input.list.set

// CHECK:   call @xla_gpu.kernel.dispatch(%[[CTX]], %[[KERNEL]], %[[ARGS]]
// CHECK-SAME:   (!xla_gpu.execution_context, !xla_gpu.kernel,
// CHECK-SAME:    !iree_input.list<!iree_input.buffer_view>, i32, i32, i32,
// CHECK-SAME:    i32, i32, i32) -> ()
// CHECK: }

// CHECK: func private @[[KERNEL_NAME]].initializer() -> !xla_gpu.kernel {
// CHECK:   %[[CREATED:.*]] = call @xla_gpu.kernel.create
// CHECK:   return %[[CREATED]] : !xla_gpu.kernel
// CHECK: }

// CHECK: iree_input.global private @[[KERNEL_NAME]]
// CHECK-SAME:  initializer(@[[KERNEL_NAME]].initializer)
