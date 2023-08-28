// RUN: export MSAN_OPTIONS=intercept_strpbrk=0
// RUN: xla-gpu2-opt %s --xla-gpu2-convert-to-runtime --split-input-file       \
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

// CHECK:   %[[RES:.*]] = iree_input.dispatch @xla.module.ptx
// CHECK:     (%[[TENSOR0]], %[[TENSOR1]], %[[TENSOR2]]) {{.*}} -> %[[TENSOR2]]
// CHECK:   iree_input.optimization_barrier %[[RES]] : tensor<3xf32>
// CHECK: }

// CHECK: iree_input.executable.source private @xla.module.ptx
// CHECK:   iree_input.executable.export public {{.*}} ordinal(0)
// CHECK:     layout(<push_constants = 0,
// CHECK:             sets = [<0, bindings = [<0, storage_buffer, ReadOnly>,
// CHECK:                                     <1, storage_buffer, ReadOnly>,
// CHECK:                                     <2, storage_buffer>]>]>)
// CHECK:   attributes {workgroup_size = [1 : index, 1 : index, 1 : index]}

// -----

func.func @fusions(
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

  "lmhlo.fusion"() ({
    %0 = bufferization.to_tensor %view0 : memref<3xf32>
    %1 = bufferization.to_tensor %view1 : memref<3xf32>
    %2 = mhlo.add %0, %1 : tensor<3xf32>
    memref.tensor_store %2, %view2 : memref<3xf32>
    "lmhlo.terminator"() : () -> ()
  }) : () -> ()

  "lmhlo.terminator"() : () -> ()
}

// Check that we chain together multiple fusions writing to the same memref
// with tied operands.

// CHECK-LABEL: func @fusions(
// CHECK:   %[[CTX:.*]]: !xla_gpu.execution_context,
// CHECK:   %[[ARG0:.*]]: tensor<12xi8>, %[[ARG1:.*]]: tensor<12xi8>,
// CHECK:   %[[ARG2:.*]]: tensor<12xi8> {lmhlo.output_index = {{.*}}}
// CHECK: ) {

// CHECK-DAG: %[[BUFFER0:.*]] = iree_input.tensor.export %[[ARG0]]
// CHECK-DAG: %[[BUFFER1:.*]] = iree_input.tensor.export %[[ARG1]]
// CHECK-DAG: %[[BUFFER2:.*]] = iree_input.tensor.export %[[ARG2]]
// CHECK-DAG: %[[TENSOR1:.*]] = iree_input.tensor.import %[[BUFFER1]]
// CHECK-DAG: %[[TENSOR0:.*]] = iree_input.tensor.import %[[BUFFER0]]
// CHECK-DAG: %[[TENSOR2:.*]] = iree_input.tensor.import %[[BUFFER2]]

// CHECK:   %[[RES0:.*]] = iree_input.dispatch @xla.module.ptx
// CHECK:     (%[[TENSOR0]], %[[TENSOR1]], %[[TENSOR2]]) {{.*}} -> %[[TENSOR2]]
// CHECK:   %[[RES1:.*]] = iree_input.dispatch @xla.module.ptx
// CHECK:     (%[[TENSOR0]], %[[TENSOR1]], %[[RES0]]) {{.*}} -> %[[RES0]]
// CHECK:   iree_input.optimization_barrier %[[RES1]] : tensor<3xf32>
// CHECK: }

// CHECK: iree_input.executable.source private @xla.module.ptx
// CHECK:   iree_input.executable.export public {{.*}} ordinal(0)
// CHECK:   iree_input.executable.export public {{.*}} ordinal(1)

// -----

#map = affine_map<(d0, d1, d2, d3) -> (d0 * 33280 + d1 * 8320 + d2 + d3 * 65)>

func.func @reinterpret_cast(
    %arg0: memref<66560xi8> {lmhlo.output_index = dense<> : tensor<0xi64>}
) {
  %c0 = arith.constant 0 : index
  %view = memref.view %arg0[%c0][] : memref<66560xi8> to memref<1x4x128x65xbf16>
  %cast = memref.reinterpret_cast %view to
            offset: [0], sizes: [1, 4, 65, 128], strides: [33280, 8320, 1, 65]
          : memref<1x4x128x65xbf16> to memref<1x4x65x128xbf16, #map>
  "lmhlo.fusion"() ({
    %0 = bufferization.to_tensor %cast : memref<1x4x65x128xbf16, #map>
    %1 = mhlo.sqrt %0 : tensor<1x4x65x128xbf16>
    memref.tensor_store %1, %cast : memref<1x4x65x128xbf16, #map>
    "lmhlo.terminator"() : () -> ()
  }) : () -> ()
  "lmhlo.terminator"() : () -> ()
}

// Buffer layout is hard coded into the kernel implementation, so we can drop
// it when lowering fustions to dispatches.

// CHECK-LABEL: func @reinterpret_cast(
// CHECK:   %[[CTX:.*]]: !xla_gpu.execution_context,
// CHECK:   %[[ARG0:.*]]: tensor<66560xi8> {lmhlo.output_index = {{.*}}}
// CHECK: ) {
// CHECK:   %[[T:.*]] = iree_input.tensor.import {{.*}} tensor<1x4x128x65xbf16>
// CHECK:   iree_input.dispatch @xla.module.ptx
// CHECK:     (%[[T]], %[[T]]) {{.*}} -> %[[T]]
// CHECK: }
