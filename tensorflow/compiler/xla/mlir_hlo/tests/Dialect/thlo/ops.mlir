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

func.func @scatter(%indices: tensor<3x2xi64>,
                   %updates: tensor<3xf32>,
                   %dst: tensor<3x3xf32>) -> tensor<3x3xf32> {
  %scatter = thlo.scatter
      ins(%indices: tensor<3x2xi64>, %updates: tensor<3xf32>)
      outs(%dst: tensor<3x3xf32>)
      (%in: f32, %out: f32) {
        %0 = arith.addf %in, %out: f32
        thlo.yield %0: f32
      }
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
                     %init: tensor<16x64xf32>) -> tensor<16x64xf32> {
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

// -----

func.func @variadic_reduction(%input1: tensor<16x32x64xf32>,
    %init1: tensor<16x64xf32>, %input2: tensor<16x32x64xi64>,
    %init2: tensor<16x64xi64>)  -> (tensor<16x64xf32>, tensor<16x64xi64>) {
  %reduction, %reduction2 = thlo.reduction
      ins(%input1:tensor<16x32x64xf32>, %input2:tensor<16x32x64xi64>)
      outs(%init1:tensor<16x64xf32>, %init2:tensor<16x64xi64>)
      dimensions = [1]
      (%in1: f32, %in2: i64, %out1: f32, %out2: i64) {
        %0 = arith.addf %in1, %out1: f32
        %1 = arith.addi %in2, %out2: i64
        thlo.yield %0, %1: f32, i64
      }
  func.return %reduction, %reduction2 : tensor<16x64xf32>, tensor<16x64xi64>
}
// CHECK-LABEL: func @variadic_reduction

// -----

func.func @map_binary(%lhs: tensor<64xf32>, %rhs: tensor<64xf32>,
                      %init: tensor<64xf32>) -> tensor<64xf32> {
   %add = thlo.map
          ins(%lhs:tensor<64xf32>, %rhs:tensor<64xf32>)
          outs(%init:tensor<64xf32>)
          (%lhs_elem: f32, %rhs_elem: f32) {
            %0 = arith.addf %lhs_elem, %rhs_elem: f32
            thlo.yield %0: f32
          }
  func.return %add : tensor<64xf32>
}
// CHECK-LABEL: func @map_binary

// -----

func.func @map_unary(%input: tensor<64xf32>,
                     %init: tensor<64xf32>) -> tensor<64xf32> {
   %abs = thlo.map
          ins(%input:tensor<64xf32>)
          outs(%init:tensor<64xf32>)
          (%input_elem: f32) {
            %0 = math.absf %input_elem: f32
            thlo.yield %0: f32
          }
  func.return %abs : tensor<64xf32>
}
// CHECK-LABEL: func @map_unary
