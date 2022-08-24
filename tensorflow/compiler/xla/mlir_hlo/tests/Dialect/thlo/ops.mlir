// RUN: mlir-hlo-opt %s --split-input-file --allow-unregistered-dialect | \
// RUN: mlir-hlo-opt --verify-diagnostics --split-input-file \
// RUN:     --allow-unregistered-dialect | \
// RUN: FileCheck %s

func.func @dynamic_broadcast_in_dim(%arg: tensor<?x?xf32>,
                                    %dst: tensor<?x?x?xf32>) {
  %bcast = thlo.dynamic_broadcast_in_dim
      ins(%arg: tensor<?x?xf32>)
      outs(%dst: tensor<?x?x?xf32>)
      broadcast_dimensions = [0, 2]
  func.return
}
// CHECK-LABEL: func @dynamic_broadcast_in_dim

// -----

func.func @gather(%arg: tensor<100xf32>,
                  %indices: tensor<42x1xi64>,
                  %dst: tensor<42xf32>) -> tensor<42xf32> {
  %gather = thlo.gather
      ins(%arg: tensor<100xf32>, %indices: tensor<42x1xi64>)
      outs(%dst: tensor<42xf32>)
  func.return %gather : tensor<42xf32>
}
// CHECK-LABEL: func @gather

// -----

func.func @scatter(%indices: tensor<2x2xi64>,
                   %updates: tensor<3xf32>,
                   %dst: tensor<3x3xf32>) -> tensor<3x3xf32> {
  %scatter = thlo.scatter
      ins(%indices: tensor<2x2xi64>, %updates: tensor<3xf32>)
      outs(%dst: tensor<3x3xf32>)
  func.return %scatter : tensor<3x3xf32>
}
// CHECK-LABEL: func @scatter

// -----

func.func @transpose(%input: tensor<16x32x64xf32>,
                     %init: tensor<32x64x16xf32>) -> tensor<32x64x16xf32> {
  %transpose = thlo.transpose
      ins(%input:tensor<16x32x64xf32>)
      outs(%init:tensor<32x64x16xf32>)
      permutation = [1, 2, 0]
  func.return %transpose : tensor<32x64x16xf32>
}
// CHECK-LABEL: func @transpose

// -----

func.func @transpose_unknown_dimentions(%input: tensor<16x?xf32>,
                                         %init: tensor<64x?xf32>) -> tensor<64x?xf32> {
  %transpose = thlo.transpose
      ins(%input:tensor<16x?xf32>)
      outs(%init:tensor<64x?xf32>)
      permutation = [1, 0]
  func.return %transpose : tensor<64x?xf32>
}
// CHECK-LABEL: func @transpose_unknown_dimentions

// -----

func.func @reduction(%input: tensor<16x32x64xf32>,
                     %init: tensor<16x64xf32>)  -> tensor<16x64xf32> {
  %reduction = thlo.reduction
      ins(%input:tensor<16x32x64xf32>)
      outs(%init:tensor<16x64xf32>)
      dimensions = [1]
      (%in: f32, %out: f32) {
        %0 = arith.addf %in, %out: f32
        thlo.yield %0: f32
      }
  func.return %reduction : tensor<16x64xf32>
}
// CHECK-LABEL: func @reduction
