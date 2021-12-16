// RUN: mlir-hlo-opt %s -input-inline-fusion -split-input-file | FileCheck %s

// CHECK-LABEL: @inline_fusion_fusion_order
// CHECK-SAME: (%[[INPUT1:.*]]: memref<?xf32>, %[[INPUT2:.*]]: memref<3xi32>, %[[INPUT3:.*]]: memref<?x?x?xf32>, %[[INPUT4:.*]]: memref<?x?x?xf32>, %[[TMP_BUF1:.*]]: memref<?x?x?xf32>, %[[TMP_BUF2:.*]]: memref<?x?x?xf32>, %[[OUT:.*]]: memref<?x?x?xf32>) -> memref<?x?x?xf32>
func @inline_fusion_fusion_order(%arg0: memref<?xf32>, %arg1: memref<3xi32>, %arg2: memref<?x?x?xf32>, %arg3: memref<?x?x?xf32>, %arg4: memref<?x?x?xf32>, %arg5: memref<?x?x?xf32>, %arg6: memref<?x?x?xf32>) -> memref<?x?x?xf32> {
  %c2 = arith.constant 2 : index
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  // CHECK: "lmhlo.fusion"() ( {
  "lmhlo.fusion"() ( {
    // CHECK-NOT: lmhlo.dynamic_broadcast_in_dim
    // CHECK-NOT: lmhlo.add
    "lmhlo.dynamic_broadcast_in_dim"(%arg0, %arg1, %arg4) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (memref<?xf32>, memref<3xi32>, memref<?x?x?xf32>) -> ()
    "lmhlo.add"(%arg2, %arg4, %arg5) : (memref<?x?x?xf32>, memref<?x?x?xf32>, memref<?x?x?xf32>) -> ()
    %0 = memref.dim %arg6, %c0 : memref<?x?x?xf32>
    %1 = memref.dim %arg6, %c1 : memref<?x?x?xf32>
    %2 = arith.muli %0, %1 : index
    %3 = memref.dim %arg6, %c2 : memref<?x?x?xf32>
    %4 = arith.muli %2, %3 : index
    // CHECK: scf.parallel
    scf.parallel (%arg7) = (%c0) to (%4) step (%c1) {
      %5 = memref.dim %arg3, %c1 : memref<?x?x?xf32>
      %6 = memref.dim %arg3, %c2 : memref<?x?x?xf32>
      %7 = arith.muli %6, %5 : index
      %8 = arith.divui %arg7, %7 : index
      %9 = arith.remui %arg7, %7 : index
      %10 = arith.divui %9, %6 : index
      %11 = arith.remui %9, %6 : index
      %12 = memref.load %arg3[%8, %10, %11] : memref<?x?x?xf32>
      %13 = memref.load %arg5[%8, %10, %11] : memref<?x?x?xf32>
      %14 = arith.mulf %12, %13 : f32
      %15 = memref.reinterpret_cast %arg6 to offset: [%c0], sizes: [%4], strides: [%c1] : memref<?x?x?xf32> to memref<?xf32>
      memref.store %14, %15[%arg7] : memref<?xf32>
      scf.yield
    }
    // CHECK: "lmhlo.terminator"() : () -> ()
    "lmhlo.terminator"() : () -> ()
  }) : () -> ()
  // CHECK: return %[[OUT]] : memref<?x?x?xf32>
  return %arg6 : memref<?x?x?xf32>
}

// CHECK-LABEL: @multioutput_loop_fusion_with_dependency
// CHECK-SAME: (%[[INPUT1:.*]]: memref<?xf32>, %[[INPUT2:.*]]: memref<3xi32>, %[[INPUT3:.*]]: memref<?x?x?xf32>, %[[TMP_BUF:.*]]: memref<?x?x?xf32>, %[[OUT1:.*]]: memref<?x?x?xf32>, %[[OUT2:.*]]: memref<?x?x?xf32>) -> (memref<?x?x?xf32>, memref<?x?x?xf32>)
func @multioutput_loop_fusion_with_dependency(%arg0: memref<?xf32>, %arg1: memref<3xi32>, %arg2: memref<?x?x?xf32>, %arg3: memref<?x?x?xf32>, %arg4: memref<?x?x?xf32>, %arg5: memref<?x?x?xf32>) -> (memref<?x?x?xf32>, memref<?x?x?xf32>) {
  %c2 = arith.constant 2 : index
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  // CHECK: "lmhlo.fusion"() ( {
  "lmhlo.fusion"() ( {
    // CHECK-NOT: lmhlo.dynamic_broadcast_in_dim
    // CHECK-NOT: lmhlo.add
    "lmhlo.dynamic_broadcast_in_dim"(%arg0, %arg1, %arg3) {broadcast_dimensions = dense<2> : tensor<1xi64>} : (memref<?xf32>, memref<3xi32>, memref<?x?x?xf32>) -> ()
    "lmhlo.add"(%arg2, %arg3, %arg4) : (memref<?x?x?xf32>, memref<?x?x?xf32>, memref<?x?x?xf32>) -> ()
    %0 = memref.dim %arg5, %c0 : memref<?x?x?xf32>
    %1 = memref.dim %arg5, %c1 : memref<?x?x?xf32>
    %2 = arith.muli %0, %1 : index
    %3 = memref.dim %arg5, %c2 : memref<?x?x?xf32>
    %4 = arith.muli %2, %3 : index
    // CHECK: scf.parallel
    scf.parallel (%arg6) = (%c0) to (%4) step (%c1) {
      %5 = memref.dim %arg2, %c1 : memref<?x?x?xf32>
      %6 = memref.dim %arg2, %c2 : memref<?x?x?xf32>
      %7 = arith.muli %6, %5 : index
      %8 = arith.divui %arg6, %7 : index
      %9 = arith.remui %arg6, %7 : index
      %10 = arith.divui %9, %6 : index
      %11 = arith.remui %9, %6 : index
      %12 = memref.load %arg2[%8, %10, %11] : memref<?x?x?xf32>
      %13 = memref.load %arg3[%8, %10, %11] : memref<?x?x?xf32>
      %14 = arith.addf %12, %13 : f32
      %15 = memref.dim %arg4, %c0 : memref<?x?x?xf32>
      %16 = memref.dim %arg4, %c1 : memref<?x?x?xf32>
      %17 = arith.muli %15, %16 : index
      %18 = memref.dim %arg4, %c2 : memref<?x?x?xf32>
      %19 = arith.muli %17, %18 : index
      %20 = memref.reinterpret_cast %arg4 to offset: [%c0], sizes: [%19], strides: [%c1] : memref<?x?x?xf32> to memref<?xf32>
      memref.store %14, %20[%arg6] : memref<?xf32>
      %21 = memref.load %arg2[%8, %10, %11] : memref<?x?x?xf32>
      %22 = memref.load %arg4[%8, %10, %11] : memref<?x?x?xf32>
      %23 = arith.mulf %21, %22 : f32
      %24 = memref.reinterpret_cast %arg5 to offset: [%c0], sizes: [%4], strides: [%c1] : memref<?x?x?xf32> to memref<?xf32>
      memref.store %23, %24[%arg6] : memref<?xf32>
      scf.yield
    }
    // CHECK: "lmhlo.terminator"() : () -> ()
    "lmhlo.terminator"() : () -> ()
  }) : () -> ()
  // CHECK: return %[[OUT1]], %[[OUT2]] : memref<?x?x?xf32>, memref<?x?x?xf32>
  return %arg4, %arg5 : memref<?x?x?xf32>, memref<?x?x?xf32>
}
