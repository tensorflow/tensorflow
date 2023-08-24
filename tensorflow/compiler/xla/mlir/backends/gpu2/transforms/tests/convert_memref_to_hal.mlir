// RUN: export MSAN_OPTIONS=intercept_strpbrk=0
// RUN: xla-gpu2-opt %s --xla-gpu2-convert-to-runtime --split-input-file       \
// RUN:   | FileCheck %s

func.func @main(%arg0: memref<12xi8>) {
  %c0 = arith.constant 0 : index
  %view = memref.view %arg0[%c0][] : memref<12xi8> to memref<3xf32>
  return
}

// CHECK-LABEL: func @main(
// CHECK:   %[[CTX:.*]]: !xla_gpu.execution_context,
// CHECK:   %[[ARG0:.*]]: tensor<12xi8>
// CHECK: ) {
// CHECK:   %[[C0:.*]] = arith.constant 0 : index
// CHECK:   %[[BUF:.*]] = iree_input.tensor.export %[[ARG0]]
// CHECK:   %[[C3:.*]] = arith.constant 3 : index
// CHECK:   %[[C12:.*]] = arith.constant 12 : index
// CHECK:   %[[TYPE:.*]] = arith.constant 553648160 : i32
// CHECK:   %[[ENCODING:.*]] = arith.constant 1 : i32
// CHECK:   %[[VIEW:.*]] = iree_input.buffer_view.create
// CHECK:       buffer(%[[BUF]] : !iree_input.buffer)[%[[C0]], %[[C12]]]
// CHECK:       shape([%[[C3]]])
// CHECK:       type(%[[TYPE]])
// CHECK:       encoding(%[[ENCODING]]) : !iree_input.buffer_view
// CHECK:   iree_input.tensor.import %[[VIEW]] {{.*}} -> tensor<3xf32>
// CHECK: }

// -----

func.func @main(%arg0: memref<12xi8>) {
  %c8 = arith.constant 8 : index
  %view = memref.view %arg0[%c8][] : memref<12xi8> to memref<f32>
  return
}

// CHECK-LABEL: func @main(
// CHECK:   %[[CTX:.*]]: !xla_gpu.execution_context,
// CHECK:   %[[ARG0:.*]]: tensor<12xi8>
// CHECK: ) {
// CHECK:   %[[C8:.*]] = arith.constant 8 : index
// CHECK:   %[[BUF:.*]] = iree_input.tensor.export %[[ARG0]]
// CHECK:   %[[C1:.*]] = arith.constant 1 : index
// CHECK:   %[[C4:.*]] = arith.constant 4 : index
// CHECK:   %[[TYPE:.*]] = arith.constant 553648160 : i32
// CHECK:   %[[ENCODING:.*]] = arith.constant 1 : i32
// CHECK:   %[[VIEW:.*]] = iree_input.buffer_view.create
// CHECK:       buffer(%[[BUF]] : !iree_input.buffer)[%[[C8]], %[[C4]]]
// CHECK:       shape([%[[C1]]])
// CHECK:       type(%[[TYPE]])
// CHECK:       encoding(%[[ENCODING]]) : !iree_input.buffer_view
// CHECK:   iree_input.tensor.import %[[VIEW]] {{.*}} -> tensor<1xf32>
// CHECK: }

// -----

memref.global "private" constant @cst : memref<i64> = dense<1>

func.func @main(%arg0: memref<8xi8> {lmhlo.constant_name = "cst"}) {
  %0 = memref.get_global @cst : memref<i64>
  return
}

// Load from a global memref correspoding to constant argument replaced
// with an argument itself.

// CHECK-LABEL: func @main(
// CHECK:   %[[CTX:.*]]: !xla_gpu.execution_context,
// CHECK:   %[[ARG0:.*]]: tensor<8xi8>
// CHECK: ) {
// CHECK:   %[[BUF:.*]] = iree_input.tensor.export %[[ARG0]]
// CHECK:   %[[VIEW:.*]] = iree_input.buffer_view.create
// CHECK:       buffer(%[[BUF]] : !iree_input.buffer)
// CHECK:   iree_input.tensor.import %[[VIEW]] {{.*}} -> tensor<1xi64>
// CHECK: }

// -----

#map = affine_map<(d0, d1, d2, d3) -> (d0 * 33280 + d1 * 8320 + d2 + d3 * 65)>

func.func @main(%arg0: memref<66560xi8>) {
  %c0 = arith.constant 0 : index
  %view = memref.view %arg0[%c0][] : memref<66560xi8> to memref<1x4x128x65xbf16>
  %cast = memref.reinterpret_cast %view to offset: [0],
                                           sizes: [1, 4, 65, 128],
                                           strides: [33280, 8320, 1, 65]
          : memref<1x4x128x65xbf16> to memref<1x4x65x128xbf16, #map>
  return
}

// We currently do not support strided memrefs, and always represent them as
// a row-major tensor (buffer view). We'll be working on adding strides to
// either the buffer view itself, or as a separate metadata object.

// CHECK-LABEL: func @main(
// CHECK:   %[[CTX:.*]]: !xla_gpu.execution_context,
// CHECK:   %[[ARG0:.*]]: tensor<66560xi8>
// CHECK: ) {
// CHECK:   %[[C0:.*]] = arith.constant 0 : index
// CHECK:   %[[BUF:.*]] = iree_input.tensor.export %[[ARG0]]
// CHECK:   %[[C1:.*]] = arith.constant 1 : index
// CHECK:   %[[C4:.*]] = arith.constant 4 : index
// CHECK:   %[[C128:.*]] = arith.constant 128 : index
// CHECK:   %[[C65:.*]] = arith.constant 65 : index
// CHECK:   %[[C66560:.*]] = arith.constant 66560 : index
// CHECK:   %[[VIEW:.*]] = iree_input.buffer_view.create
// CHECK:     buffer(%[[BUF]] : !iree_input.buffer)[%c0, %[[C66560]]]
// CHECK:     shape([%[[C1]], %[[C4]], %[[C128]], %[[C65]]])
// CHECK: }
