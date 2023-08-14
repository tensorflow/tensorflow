// RUN: export MSAN_OPTIONS=intercept_strpbrk=0
// RUN: xla-gpu2-opt %s --xla-gpu2-create-graph-regions --split-input-file     \
// RUN:   | FileCheck %s

func.func @fusion(%arg0: memref<12xi8>, %arg1: memref<12xi8>,
                  %arg2: memref<12xi8> ) {
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

// CHECK-LABEL: func @fusion
// CHECK: memref.view
// CHECK: memref.view
// CHECK: memref.view
// CHECK: xla_gpu.graph.region {
// CHECK:   lmhlo.fusion
// CHECK: }

// -----

func.func @fusions(%arg0: memref<12xi8>, %arg1: memref<12xi8>,
                  %arg2: memref<12xi8> ) {
  %c0 = arith.constant 0 : index
  %view0 = memref.view %arg0[%c0][] : memref<12xi8> to memref<3xf32>
  %view1 = memref.view %arg1[%c0][] : memref<12xi8> to memref<3xf32>
  "lmhlo.fusion"() ({
    %0 = bufferization.to_tensor %view0 : memref<3xf32>
    %1 = bufferization.to_tensor %view0 : memref<3xf32>
    %2 = mhlo.add %0, %1 : tensor<3xf32>
    memref.tensor_store %2, %view1 : memref<3xf32>
    "lmhlo.terminator"() : () -> ()
  }) : () -> ()
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

// CHECK-LABEL: func @fusion
// CHECK: memref.view
// CHECK: memref.view
// CHECK: memref.view
// CHECK: xla_gpu.graph.region {
// CHECK:   lmhlo.fusion
// CHECK:   lmhlo.fusion
// CHECK: }
