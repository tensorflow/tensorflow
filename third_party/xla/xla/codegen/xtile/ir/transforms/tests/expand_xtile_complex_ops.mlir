// RUN: emitters_opt %s --expand-xtile-complex-ops -split-input-file | FileCheck %s

xtile.entry_func @add(%arg0: memref<128x256xcomplex<f32>>, %arg1: memref<128x256xcomplex<f32>>, %arg2: memref<128x256xcomplex<f32>>, %arg3: index) attributes {num_opaque_args = 0 : i32} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %lhs = xtile.extract %arg0[%c0, %c0] [1, 4] [1, 1] : memref<128x256xcomplex<f32>> -> tensor<1x4xcomplex<f32>>
  %rhs = xtile.extract %arg1[%c1, %c1] [1, 4] [1, 1] : memref<128x256xcomplex<f32>> -> tensor<1x4xcomplex<f32>>
  %add = stablehlo.add %lhs, %rhs : tensor<1x4xcomplex<f32>>
  xtile.insert %add into %arg2[%c0, %c1] [1, 4] [1, 1] : tensor<1x4xcomplex<f32>> -> memref<128x256xcomplex<f32>>
  xtile.return
}

// CHECK-LABEL: xtile.entry_func @add
// CHECK-SAME: (%arg0: memref<128x256x2xf32>, %arg1: memref<128x256x2xf32>, %arg2: memref<128x256x2xf32>, %arg3: index)
// CHECK-DAG:    %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:    %[[C1:.*]] = arith.constant 1 : index
// CHECK:        %[[LHS:.*]] = xtile.extract %arg0[%[[C0]], %[[C0]], %[[C0]]] [1, 4, 2] [1, 1, 1] : memref<128x256x2xf32> -> tensor<1x4x2xf32>
// CHECK:        %[[RHS:.*]] = xtile.extract %arg1[%[[C1]], %[[C1]], %[[C0]]] [1, 4, 2] [1, 1, 1] : memref<128x256x2xf32> -> tensor<1x4x2xf32>
// CHECK:        %[[ADD:.*]] = stablehlo.add %[[LHS]], %[[RHS]] : tensor<1x4x2xf32>
// CHECK:        xtile.insert %[[ADD]] into %arg2[%[[C0]], %[[C1]], %[[C0]]] [1, 4, 2] [1, 1, 1] : tensor<1x4x2xf32> -> memref<128x256x2xf32>
// CHECK:        xtile.return

// -----

xtile.entry_func @subtract(%arg0: memref<1xcomplex<f32>>, %arg1: memref<1xcomplex<f32>>, %arg2: memref<1xcomplex<f32>>, %arg3: index) attributes {num_opaque_args = 0 : i32} {
  %c0 = arith.constant 0 : index
  %lhs = xtile.extract %arg0[%c0] [1] [1] : memref<1xcomplex<f32>> -> tensor<1xcomplex<f32>>
  %rhs = xtile.extract %arg1[%c0] [1] [1] : memref<1xcomplex<f32>> -> tensor<1xcomplex<f32>>
  %sub = stablehlo.subtract %lhs, %rhs : tensor<1xcomplex<f32>>
  xtile.insert %sub into %arg2[%c0] [1] [1] : tensor<1xcomplex<f32>> -> memref<1xcomplex<f32>>
  xtile.return
}

// CHECK-LABEL: xtile.entry_func @subtract
// CHECK-SAME: (%arg0: memref<1x2xf32>, %arg1: memref<1x2xf32>, %arg2: memref<1x2xf32>, %arg3: index)
// CHECK:        %[[LHS:.*]] = xtile.extract %arg0[{{.*}}] [1, 2] [1, 1] : memref<1x2xf32> -> tensor<1x2xf32>
// CHECK:        %[[RHS:.*]] = xtile.extract %arg1[{{.*}}] [1, 2] [1, 1] : memref<1x2xf32> -> tensor<1x2xf32>
// CHECK:        %[[SUB:.*]] = stablehlo.subtract %[[LHS]], %[[RHS]] : tensor<1x2xf32>
// CHECK:        xtile.insert %[[SUB]] into %arg2[{{.*}}] [1, 2] [1, 1] : tensor<1x2xf32> -> memref<1x2xf32>

// -----

xtile.entry_func @real(%arg0: memref<1xcomplex<f32>>, %arg1: memref<1xf32>, %arg2: index) attributes {num_opaque_args = 0 : i32} {
  %c0 = arith.constant 0 : index
  %lhs = xtile.extract %arg0[%c0] [1] [1] : memref<1xcomplex<f32>> -> tensor<1xcomplex<f32>>
  %real = stablehlo.real %lhs : (tensor<1xcomplex<f32>>) -> tensor<1xf32>
  xtile.insert %real into %arg1[%c0] [1] [1] : tensor<1xf32> -> memref<1xf32>
  xtile.return
}

// CHECK-LABEL: xtile.entry_func @real
// CHECK-SAME: (%arg0: memref<1x2xf32>, %arg1: memref<1xf32>, %arg2: index)
// CHECK:        %[[SLICE:.*]] = stablehlo.slice %{{.*}} [0:1, 0:1] : (tensor<1x2xf32>) -> tensor<1x1xf32>
// CHECK:        %[[REAL:.*]] = stablehlo.reshape %[[SLICE]] : (tensor<1x1xf32>) -> tensor<1xf32>
// CHECK:        xtile.insert %[[REAL]] into %arg1[{{.*}}] [1] [1] : tensor<1xf32> -> memref<1xf32>

// -----

xtile.entry_func @imag(%arg0: memref<1xcomplex<f32>>, %arg1: memref<1xf32>, %arg2: index) attributes {num_opaque_args = 0 : i32} {
  %c0 = arith.constant 0 : index
  %lhs = xtile.extract %arg0[%c0] [1] [1] : memref<1xcomplex<f32>> -> tensor<1xcomplex<f32>>
  %imag = stablehlo.imag %lhs : (tensor<1xcomplex<f32>>) -> tensor<1xf32>
  xtile.insert %imag into %arg1[%c0] [1] [1] : tensor<1xf32> -> memref<1xf32>
  xtile.return
}

// CHECK-LABEL: xtile.entry_func @imag
// CHECK-SAME: (%arg0: memref<1x2xf32>, %arg1: memref<1xf32>, %arg2: index)
// CHECK:        %[[SLICE:.*]] = stablehlo.slice %{{.*}} [0:1, 1:2] : (tensor<1x2xf32>) -> tensor<1x1xf32>
// CHECK:        %[[IMAG:.*]] = stablehlo.reshape %[[SLICE]] : (tensor<1x1xf32>) -> tensor<1xf32>
// CHECK:        xtile.insert %[[IMAG]] into %arg1[{{.*}}] [1] [1] : tensor<1xf32> -> memref<1xf32>

// -----

xtile.entry_func @complex(%arg0: memref<1xf32>, %arg1: memref<1xf32>, %arg2: memref<1xcomplex<f32>>, %arg3: index) attributes {num_opaque_args = 0 : i32} {
  %c0 = arith.constant 0 : index
  %real = xtile.extract %arg0[%c0] [1] [1] : memref<1xf32> -> tensor<1xf32>
  %imag = xtile.extract %arg1[%c0] [1] [1] : memref<1xf32> -> tensor<1xf32>
  %cplx = stablehlo.complex %real, %imag : tensor<1xcomplex<f32>>
  xtile.insert %cplx into %arg2[%c0] [1] [1] : tensor<1xcomplex<f32>> -> memref<1xcomplex<f32>>
  xtile.return
}

// CHECK-LABEL: xtile.entry_func @complex
// CHECK-SAME: (%arg0: memref<1xf32>, %arg1: memref<1xf32>, %arg2: memref<1x2xf32>, %arg3: index)
// CHECK:        %[[R_RESHAPE:.*]] = stablehlo.reshape %{{.*}} : (tensor<1xf32>) -> tensor<1x1xf32>
// CHECK:        %[[I_RESHAPE:.*]] = stablehlo.reshape %{{.*}} : (tensor<1xf32>) -> tensor<1x1xf32>
// CHECK:        %[[COMB:.*]] = stablehlo.concatenate %[[R_RESHAPE]], %[[I_RESHAPE]], dim = 1 : (tensor<1x1xf32>, tensor<1x1xf32>) -> tensor<1x2xf32>
// CHECK:        xtile.insert %[[COMB]] into %arg2[{{.*}}] [1, 2] [1, 1] : tensor<1x2xf32> -> memref<1x2xf32>

// -----

xtile.entry_func @constants(%arg0: memref<1xcomplex<f32>>, %arg1: index) attributes {num_opaque_args = 0 : i32} {
  %c0 = arith.constant 0 : index
  %c_arith = arith.constant dense<(1.0, 2.0)> : tensor<1xcomplex<f32>>
  xtile.insert %c_arith into %arg0[%c0] [1] [1] : tensor<1xcomplex<f32>> -> memref<1xcomplex<f32>>
  xtile.return
}

// CHECK-LABEL: xtile.entry_func @constants
// CHECK-SAME: (%arg0: memref<1x2xf32>, %arg1: index)
// CHECK:        %[[C_ARITH:.*]] = arith.constant dense<[{{\[}}1.000000e+00, 2.000000e+00{{\]}}]> : tensor<1x2xf32>
// CHECK:        xtile.insert %[[C_ARITH]] into %arg0[{{.*}}] [1, 2] [1, 1] : tensor<1x2xf32> -> memref<1x2xf32>
