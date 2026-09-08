// RUN: emitters_opt %s --expand-xtile-complex-ops -split-input-file | FileCheck %s

xtile.entry_func @add(%arg0: memref<128x256xcomplex<f32>>,
                      %arg1: memref<128x256xcomplex<f32>>,
                      %arg2: memref<128x256xcomplex<f32>>,
                      %arg3: index) attributes {num_opaque_args = 0 : i32} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %lhs = xtile.extract %arg0[%c0, %c0] [1, 4] [1, 1]
    : memref<128x256xcomplex<f32>> -> tensor<1x4xcomplex<f32>>
  %rhs = xtile.extract %arg1[%c1, %c1] [1, 4] [1, 1]
    : memref<128x256xcomplex<f32>> -> tensor<1x4xcomplex<f32>>
  %add = stablehlo.add %lhs, %rhs : tensor<1x4xcomplex<f32>>
  xtile.insert %add into %arg2[%c0, %c1] [1, 4] [1, 1]
    : tensor<1x4xcomplex<f32>> -> memref<128x256xcomplex<f32>>
  xtile.return
}
// CHECK-LABEL: xtile.entry_func @add
// CHECK-SAME: (%[[ARG0:[a-zA-Z0-9_]*]]: memref<128x256x2xf32>, %[[ARG1:[a-zA-Z0-9_]*]]: memref<128x256x2xf32>, %[[ARG2:[a-zA-Z0-9_]*]]: memref<128x256x2xf32>, %[[ARG3:[a-zA-Z0-9_]*]]: index)

// CHECK-DAG:    %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:    %[[C1:.*]] = arith.constant 1 : index

// CHECK:        %[[LHS:.*]] = xtile.extract %[[ARG0]][%[[C0]], %[[C0]], %[[C0]]] [1, 4, 2] [1, 1, 1] : memref<128x256x2xf32> -> tensor<1x4x2xf32>
// CHECK:        %[[RHS:.*]] = xtile.extract %[[ARG1]][%[[C1]], %[[C1]], %[[C0]]] [1, 4, 2] [1, 1, 1] : memref<128x256x2xf32> -> tensor<1x4x2xf32>
// CHECK:        %[[ADD:.*]] = stablehlo.add %[[LHS]], %[[RHS]] : tensor<1x4x2xf32>
// CHECK:        xtile.insert %[[ADD]] into %[[ARG2]][%[[C0]], %[[C1]], %[[C0]]] [1, 4, 2] [1, 1, 1] : tensor<1x4x2xf32> -> memref<128x256x2xf32>

// -----

xtile.entry_func @subtract(%arg0: memref<1xcomplex<f32>>,
                           %arg1: memref<1xcomplex<f32>>,
                           %arg2: memref<1xcomplex<f32>>,
                           %arg3: index) attributes {num_opaque_args = 0 : i32} {
  %c0 = arith.constant 0 : index
  %lhs = xtile.extract %arg0[%c0] [1] [1]
    : memref<1xcomplex<f32>> -> tensor<1xcomplex<f32>>
  %rhs = xtile.extract %arg1[%c0] [1] [1]
    : memref<1xcomplex<f32>> -> tensor<1xcomplex<f32>>
  %sub = stablehlo.subtract %lhs, %rhs : tensor<1xcomplex<f32>>
  xtile.insert %sub into %arg2[%c0] [1] [1]
    : tensor<1xcomplex<f32>> -> memref<1xcomplex<f32>>
  xtile.return
}
// CHECK-LABEL: xtile.entry_func @subtract
// CHECK-SAME: (%[[ARG0:[a-zA-Z0-9_]*]]: memref<1x2xf32>, %[[ARG1:[a-zA-Z0-9_]*]]: memref<1x2xf32>, %[[ARG2:[a-zA-Z0-9_]*]]: memref<1x2xf32>, %[[ARG3:[a-zA-Z0-9_]*]]: index)

// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK: %[[LHS:.*]] = xtile.extract %[[ARG0]][%[[C0]], %[[C0]]] [1, 2] [1, 1] : memref<1x2xf32> -> tensor<1x2xf32>
// CHECK: %[[RHS:.*]] = xtile.extract %[[ARG1]][%[[C0]], %[[C0]]] [1, 2] [1, 1] : memref<1x2xf32> -> tensor<1x2xf32>
// CHECK: %[[SUB:.*]] = stablehlo.subtract %[[LHS]], %[[RHS]] : tensor<1x2xf32>
// CHECK: xtile.insert %[[SUB]] into %[[ARG2]][%[[C0]], %[[C0]]] [1, 2] [1, 1] : tensor<1x2xf32> -> memref<1x2xf32>

// -----

xtile.entry_func @real(%arg0: memref<1xcomplex<f32>>, %arg1: memref<1xf32>,
                       %arg2: index) attributes {num_opaque_args = 0 : i32} {
  %c0 = arith.constant 0 : index
  %lhs = xtile.extract %arg0[%c0] [1] [1]
    : memref<1xcomplex<f32>> -> tensor<1xcomplex<f32>>
  %real = stablehlo.real %lhs : (tensor<1xcomplex<f32>>) -> tensor<1xf32>
  xtile.insert %real into %arg1[%c0] [1] [1] : tensor<1xf32> -> memref<1xf32>
  xtile.return
}

// CHECK-LABEL: xtile.entry_func @real
// CHECK-SAME: (%[[ARG0:[a-zA-Z0-9_]*]]: memref<1x2xf32>, %[[ARG1:[a-zA-Z0-9_]*]]: memref<1xf32>, %[[ARG2:[a-zA-Z0-9_]*]]: index)

// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK: %[[LHS:.*]] = xtile.extract %[[ARG0]][%[[C0]], %[[C0]]] [1, 2] [1, 1] : memref<1x2xf32> -> tensor<1x2xf32>
// CHECK: %[[SLICE:.*]] = stablehlo.slice %[[LHS]] [0:1, 0:1] : (tensor<1x2xf32>) -> tensor<1x1xf32>
// CHECK: %[[REAL:.*]] = stablehlo.reshape %[[SLICE]] : (tensor<1x1xf32>) -> tensor<1xf32>
// CHECK: xtile.insert %[[REAL]] into %[[ARG1]][%[[C0]]] [1] [1] : tensor<1xf32> -> memref<1xf32>

// -----

xtile.entry_func @imag(%arg0: memref<1xcomplex<f32>>, %arg1: memref<1xf32>,
                       %arg2: index) attributes {num_opaque_args = 0 : i32} {
  %c0 = arith.constant 0 : index
  %lhs = xtile.extract %arg0[%c0] [1] [1]
    : memref<1xcomplex<f32>> -> tensor<1xcomplex<f32>>
  %imag = stablehlo.imag %lhs : (tensor<1xcomplex<f32>>) -> tensor<1xf32>
  xtile.insert %imag into %arg1[%c0] [1] [1] : tensor<1xf32> -> memref<1xf32>
  xtile.return
}

// CHECK-LABEL: xtile.entry_func @imag
// CHECK-SAME: (%[[ARG0:[a-zA-Z0-9_]*]]: memref<1x2xf32>, %[[ARG1:[a-zA-Z0-9_]*]]: memref<1xf32>, %[[ARG2:[a-zA-Z0-9_]*]]: index)

// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK: %[[LHS:.*]] = xtile.extract %[[ARG0]][%[[C0]], %[[C0]]] [1, 2] [1, 1] : memref<1x2xf32> -> tensor<1x2xf32>
// CHECK: %[[SLICE:.*]] = stablehlo.slice %[[LHS]] [0:1, 1:2] : (tensor<1x2xf32>) -> tensor<1x1xf32>
// CHECK: %[[IMAG:.*]] = stablehlo.reshape %[[SLICE]] : (tensor<1x1xf32>) -> tensor<1xf32>
// CHECK: xtile.insert %[[IMAG]] into %[[ARG1]][%[[C0]]] [1] [1] : tensor<1xf32> -> memref<1xf32>

// -----

xtile.entry_func @complex(%arg0: memref<1xf32>, %arg1: memref<1xf32>,
                          %arg2: memref<1xcomplex<f32>>,
                          %arg3: index) attributes {num_opaque_args = 0 : i32} {
  %c0 = arith.constant 0 : index
  %real = xtile.extract %arg0[%c0] [1] [1] : memref<1xf32> -> tensor<1xf32>
  %imag = xtile.extract %arg1[%c0] [1] [1] : memref<1xf32> -> tensor<1xf32>
  %cplx = stablehlo.complex %real, %imag : tensor<1xcomplex<f32>>
  xtile.insert %cplx into %arg2[%c0] [1] [1]
    : tensor<1xcomplex<f32>> -> memref<1xcomplex<f32>>
  xtile.return
}
// CHECK-LABEL: xtile.entry_func @complex
// CHECK-SAME: (%[[ARG0:[a-zA-Z0-9_]*]]: memref<1xf32>, %[[ARG1:[a-zA-Z0-9_]*]]: memref<1xf32>, %[[ARG2:[a-zA-Z0-9_]*]]: memref<1x2xf32>, %[[ARG3:[a-zA-Z0-9_]*]]: index)

// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK: %[[REAL:.*]] = xtile.extract %[[ARG0]][%[[C0]]] [1] [1] : memref<1xf32> -> tensor<1xf32>
// CHECK: %[[IMAG:.*]] = xtile.extract %[[ARG1]][%[[C0]]] [1] [1] : memref<1xf32> -> tensor<1xf32>
// CHECK: %[[R_RESHAPE:.*]] = stablehlo.reshape %[[REAL]] : (tensor<1xf32>) -> tensor<1x1xf32>
// CHECK: %[[I_RESHAPE:.*]] = stablehlo.reshape %[[IMAG]] : (tensor<1xf32>) -> tensor<1x1xf32>
// CHECK: %[[COMB:.*]] = stablehlo.concatenate %[[R_RESHAPE]], %[[I_RESHAPE]], dim = 1 : (tensor<1x1xf32>, tensor<1x1xf32>) -> tensor<1x2xf32>
// CHECK: xtile.insert %[[COMB]] into %[[ARG2]][%[[C0]], %[[C0]]] [1, 2] [1, 1] : tensor<1x2xf32> -> memref<1x2xf32>

// -----

xtile.entry_func @constants(%arg0: memref<1xcomplex<f32>>, %arg1: index)
    attributes {num_opaque_args = 0 : i32} {
  %c0 = arith.constant 0 : index
  %c_arith = arith.constant dense<(1.0, 2.0)> : tensor<1xcomplex<f32>>
  xtile.insert %c_arith into %arg0[%c0] [1] [1]
    : tensor<1xcomplex<f32>> -> memref<1xcomplex<f32>>
  xtile.return
}
// CHECK-LABEL: xtile.entry_func @constants
// CHECK-SAME: (%[[ARG0:[a-zA-Z0-9_]*]]: memref<1x2xf32>, %[[ARG1:[a-zA-Z0-9_]*]]: index)

// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[C_ARITH:.*]] = arith.constant dense<{{.*}}> : tensor<1x2xf32>
// CHECK: xtile.insert %[[C_ARITH]] into %[[ARG0]][%[[C0]], %[[C0]]] [1, 2] [1, 1] : tensor<1x2xf32> -> memref<1x2xf32>

// -----

xtile.entry_func @reshape(%arg0: memref<128x256xcomplex<f32>>,
                          %arg1: memref<32768xcomplex<f32>>,
                          %arg2: index) attributes {num_opaque_args = 0 : i32} {
  %c0 = arith.constant 0 : index
  %lhs = xtile.extract %arg0[%c0, %c0] [1, 4] [1, 1]
    : memref<128x256xcomplex<f32>> -> tensor<1x4xcomplex<f32>>
  %reshape = stablehlo.reshape %lhs
    : (tensor<1x4xcomplex<f32>>) -> tensor<4xcomplex<f32>>
  xtile.insert %reshape into %arg1[%c0] [4] [1]
    : tensor<4xcomplex<f32>> -> memref<32768xcomplex<f32>>
  xtile.return
}

// CHECK-LABEL: xtile.entry_func @reshape
// CHECK-SAME: (%[[ARG0:[a-zA-Z0-9_]*]]: memref<128x256x2xf32>, %[[ARG1:[a-zA-Z0-9_]*]]: memref<32768x2xf32>, %[[ARG2:[a-zA-Z0-9_]*]]: index)

// CHECK-DAG:    %[[C0:.*]] = arith.constant 0 : index

// CHECK: %[[LHS:.*]] = xtile.extract %[[ARG0]][%[[C0]], %[[C0]], %[[C0]]] [1, 4, 2] [1, 1, 1] : memref<128x256x2xf32> -> tensor<1x4x2xf32>
// CHECK: %[[RESHAPE:.*]] = stablehlo.reshape %[[LHS]] : (tensor<1x4x2xf32>) -> tensor<4x2xf32>
// CHECK: xtile.insert %[[RESHAPE]] into %[[ARG1]][%[[C0]], %[[C0]]] [4, 2] [1, 1] : tensor<4x2xf32> -> memref<32768x2xf32>

// -----

xtile.entry_func @transpose(%arg0: memref<128x256xcomplex<f32>>,
                            %arg1: memref<128x256xcomplex<f32>>,
                            %arg2: index) attributes {num_opaque_args = 0 : i32} {
  %c0 = arith.constant 0 : index
  %lhs = xtile.extract %arg0[%c0, %c0] [1, 4] [1, 1]
    : memref<128x256xcomplex<f32>> -> tensor<1x4xcomplex<f32>>
  %transpose = stablehlo.transpose %lhs, dims = [1, 0]
    : (tensor<1x4xcomplex<f32>>) -> tensor<4x1xcomplex<f32>>
  xtile.insert %transpose into %arg1[%c0, %c0] [4, 1] [1, 1]
    : tensor<4x1xcomplex<f32>> -> memref<128x256xcomplex<f32>>
  xtile.return
}

// CHECK-LABEL: xtile.entry_func @transpose
// CHECK-SAME: (%[[ARG0:[a-zA-Z0-9_]*]]: memref<128x256x2xf32>, %[[ARG1:[a-zA-Z0-9_]*]]: memref<128x256x2xf32>, %[[ARG2:[a-zA-Z0-9_]*]]: index)

// CHECK-DAG:    %[[C0:.*]] = arith.constant 0 : index

// CHECK: %[[LHS:.*]] = xtile.extract %[[ARG0]][%[[C0]], %[[C0]], %[[C0]]] [1, 4, 2] [1, 1, 1] : memref<128x256x2xf32> -> tensor<1x4x2xf32>
// CHECK: %[[TRANSPOSE:.*]] = stablehlo.transpose %[[LHS]], dims = [1, 0, 2] : (tensor<1x4x2xf32>) -> tensor<4x1x2xf32>
// CHECK: xtile.insert %[[TRANSPOSE]] into %[[ARG1]][%[[C0]], %[[C0]], %[[C0]]] [4, 1, 2] [1, 1, 1] : tensor<4x1x2xf32> -> memref<128x256x2xf32>

// -----

xtile.entry_func @convert_real_to_complex(%arg0: memref<1xi32>,
    %arg1: memref<1xcomplex<f32>>,
    %arg2: index) attributes {num_opaque_args = 0 : i32} {
  %c0 = arith.constant 0 : index
  %lhs = xtile.extract %arg0[%c0] [1] [1] : memref<1xi32> -> tensor<1xi32>
  %conv = stablehlo.convert %lhs : (tensor<1xi32>) -> tensor<1xcomplex<f32>>
  xtile.insert %conv into %arg1[%c0] [1] [1]
    : tensor<1xcomplex<f32>> -> memref<1xcomplex<f32>>
  xtile.return
}

// CHECK-LABEL: xtile.entry_func @convert_real_to_complex
// CHECK-SAME:   (%[[ARG0:[a-zA-Z0-9_]*]]: memref<1xi32>,
// CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]*]]: memref<1x2xf32>,
// CHECK-SAME:   %[[ARG2:[a-zA-Z0-9_]*]]: index)

// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[ZERO:.*]] = arith.constant dense<0.000000e+00> : tensor<1xf32>

// CHECK: %[[LHS:.*]] = xtile.extract %[[ARG0]][%[[C0]]] [1] [1] : memref<1xi32> -> tensor<1xi32>
// CHECK: %[[REAL:.*]] = stablehlo.convert %[[LHS]] : (tensor<1xi32>) -> tensor<1xf32>

// CHECK:      %[[R_RESHAPE:.*]] = stablehlo.reshape %[[REAL]]
// CHECK-SAME:   : (tensor<1xf32>) -> tensor<1x1xf32>
// CHECK:      %[[I_RESHAPE:.*]] = stablehlo.reshape %[[ZERO]]
// CHECK-SAME:   : (tensor<1xf32>) -> tensor<1x1xf32>

// CHECK: %[[COMB:.*]] = stablehlo.concatenate %[[R_RESHAPE]], %[[I_RESHAPE]], dim = 1 : (tensor<1x1xf32>, tensor<1x1xf32>) -> tensor<1x2xf32>
// CHECK: xtile.insert %[[COMB]] into %[[ARG1]][%[[C0]], %[[C0]]] [1, 2] [1, 1] : tensor<1x2xf32> -> memref<1x2xf32>

// -----

xtile.entry_func @convert_complex_to_complex(%arg0: memref<1xcomplex<f32>>,
      %arg1: memref<1xcomplex<f64>>,
      %arg2: index) attributes {num_opaque_args = 0 : i32} {
  %c0 = arith.constant 0 : index
  %lhs = xtile.extract %arg0[%c0] [1] [1]
    : memref<1xcomplex<f32>> -> tensor<1xcomplex<f32>>
  %conv = stablehlo.convert %lhs
    : (tensor<1xcomplex<f32>>) -> tensor<1xcomplex<f64>>
  xtile.insert %conv into %arg1[%c0] [1] [1]
    : tensor<1xcomplex<f64>> -> memref<1xcomplex<f64>>
  xtile.return
}
// CHECK-LABEL: xtile.entry_func @convert_complex_to_complex
// CHECK-SAME:   (%[[ARG0:[a-zA-Z0-9_]*]]: memref<1x2xf32>,
// CHECK-SAME:   %[[ARG1:[a-zA-Z0-9_]*]]: memref<1x2xf64>,
// CHECK-SAME:   %[[ARG2:[a-zA-Z0-9_]*]]: index)

// CHECK: %[[C0:.*]] = arith.constant 0 : index
// CHECK: %[[LHS:.*]] = xtile.extract %[[ARG0]][%[[C0]], %[[C0]]] [1, 2] [1, 1] : memref<1x2xf32> -> tensor<1x2xf32>
// CHECK: %[[CONV:.*]] = stablehlo.convert %[[LHS]] : (tensor<1x2xf32>) -> tensor<1x2xf64>
// CHECK: xtile.insert %[[CONV]] into %[[ARG1]][%[[C0]], %[[C0]]] [1, 2] [1, 1] : tensor<1x2xf64> -> memref<1x2xf64>


// -----

xtile.entry_func @transpose_with_layout(
    %arg0: memref<128x256xcomplex<f32>, #xtile.layout<[0, 1]>>,
     %arg1: memref<128x256xcomplex<f32>>,
     %arg2: index) attributes {num_opaque_args = 0 : i32} {
  %c0 = arith.constant 0 : index
  %lhs = xtile.extract %arg0[%c0, %c0] [1, 4] [1, 1]
    : memref<128x256xcomplex<f32>, #xtile.layout<[0, 1]>> -> tensor<1x4xcomplex<f32>>
  %transpose = stablehlo.transpose %lhs, dims = [1, 0]
    : (tensor<1x4xcomplex<f32>>) -> tensor<4x1xcomplex<f32>>
  xtile.insert %transpose into %arg1[%c0, %c0] [4, 1] [1, 1]
    : tensor<4x1xcomplex<f32>> -> memref<128x256xcomplex<f32>>
  xtile.return
}

// CHECK-LABEL: xtile.entry_func @transpose_with_layout
// CHECK-SAME: (%[[ARG0:[a-zA-Z0-9_]*]]: memref<128x256x2xf32, #xtile.layout<[2, 0, 1]>>,
// CHECK-SAME: %[[ARG1:[a-zA-Z0-9_]*]]: memref<128x256x2xf32>,
// CHECK-SAME: %[[ARG2:[a-zA-Z0-9_]*]]: index)
