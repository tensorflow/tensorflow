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
  xla_gpu.graph.region {
    // Read: [view0] / Write: [view0]
    "lmhlo.fusion"() ({
      %0 = bufferization.to_tensor %view0 : memref<3xf32>
      %1 = bufferization.to_tensor %view0 : memref<3xf32>
      %2 = mhlo.add %0, %1 : tensor<3xf32>
      memref.tensor_store %2, %view0 : memref<3xf32>
      "lmhlo.terminator"() : () -> ()
    }) : () -> ()
    // Read: [view1] / Write: [view1]
    "lmhlo.fusion"() ({
      %0 = bufferization.to_tensor %view1 : memref<3xf32>
      %1 = bufferization.to_tensor %view1 : memref<3xf32>
      %2 = mhlo.add %0, %1 : tensor<3xf32>
      memref.tensor_store %2, %view1 : memref<3xf32>
      "lmhlo.terminator"() : () -> ()
    }) : () -> ()
    // Read: [view0, view1] / Write: [view2]
    "lmhlo.fusion"() ({
      %0 = bufferization.to_tensor %view0 : memref<3xf32>
      %1 = bufferization.to_tensor %view1 : memref<3xf32>
      %2 = mhlo.add %0, %1 : tensor<3xf32>
      memref.tensor_store %2, %view2 : memref<3xf32>
      "lmhlo.terminator"() : () -> ()
    }) : () -> ()
  }
  "lmhlo.terminator"() : () -> ()
}

// CHECK-LABEL: func @fusion(
// CHECK:   %[[CTX:.*]]: !xla_gpu.execution_context,
// CHECK:   %[[ARG0:.*]]: tensor<12xi8>, %[[ARG1:.*]]: tensor<12xi8>,
// CHECK:   %[[ARG2:.*]]: tensor<12xi8> {lmhlo.output_index = {{.*}}}
// CHECK: )

// CHECK: xla_gpu.graph.dispatch graph(%[[GRAPH:.*]]: !xla_gpu.graph) {

// CHECK:   iree_input.global.load @__xla_gpu_kernel.unknown.0
// CHECK:   iree_input.list.create {{.*}} !iree_input.list<!xla_gpu.graph.node>
// CHECK-NEXT: %[[N0:.*]] = func.call @xla_gpu.graph.kernel_node.create

// CHECK:   iree_input.global.load @__xla_gpu_kernel.unknown.1
// CHECK:   iree_input.list.create {{.*}} !iree_input.list<!xla_gpu.graph.node>
// CHECK-NEXT: %[[N1:.*]] = func.call @xla_gpu.graph.kernel_node.create

// CHECK:   iree_input.global.load @__xla_gpu_kernel.unknown.2
// CHECK:   iree_input.list.create {{.*}} !iree_input.list<!xla_gpu.graph.node>
// CHECK:   iree_input.list.set {{.*}}, %[[N0]]
// CHECK:   iree_input.list.set {{.*}}, %[[N1]]
// CHECK: %[[N0:.*]] = func.call @xla_gpu.graph.kernel_node.create

// CHECK: }