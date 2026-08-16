// RUN: emitters_opt %s --expand-xtile-complex-ops | FileCheck %s

xtile.entry_func @wrapped_fusion(%arg0: memref<128x256xcomplex<f32>>, %arg1: memref<128x256xcomplex<f32>>, %arg2: memref<128x256xcomplex<f32>>, %arg3: index) attributes {num_opaque_args = 0 : i32} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %lhs = xtile.extract %arg0[%c0, %c0] [1, 4] [1, 1] : memref<128x256xcomplex<f32>> -> tensor<1x4xcomplex<f32>>
  %rhs = xtile.extract %arg1[%c1, %c1] [1, 4] [1, 1] : memref<128x256xcomplex<f32>> -> tensor<1x4xcomplex<f32>>
  %add = stablehlo.add %lhs, %rhs : tensor<1x4xcomplex<f32>>
  xtile.insert %add into %arg2[%c0, %c1] [1, 4] [1, 1] : tensor<1x4xcomplex<f32>> -> memref<128x256xcomplex<f32>>
  xtile.return
}

// CHECK-LABEL: xtile.entry_func @wrapped_fusion
// CHECK-SAME: (%arg0: memref<128x256x2xf32>, %arg1: memref<128x256x2xf32>, %arg2: memref<128x256x2xf32>, %arg3: index)
// CHECK-DAG:    %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:    %[[C1:.*]] = arith.constant 1 : index
// CHECK:        %[[LHS:.*]] = xtile.extract %arg0[%[[C0]], %[[C0]], %[[C0]]] [1, 4, 2] [1, 1, 1] : memref<128x256x2xf32> -> tensor<1x4x2xf32>
// CHECK:        %[[RHS:.*]] = xtile.extract %arg1[%[[C1]], %[[C1]], %[[C0]]] [1, 4, 2] [1, 1, 1] : memref<128x256x2xf32> -> tensor<1x4x2xf32>
// CHECK:        %[[ADD:.*]] = stablehlo.add %[[LHS]], %[[RHS]] : tensor<1x4x2xf32>
// CHECK:        xtile.insert %[[ADD]] into %arg2[%[[C0]], %[[C1]], %[[C0]]] [1, 4, 2] [1, 1, 1] : tensor<1x4x2xf32> -> memref<128x256x2xf32>
// CHECK:        xtile.return
