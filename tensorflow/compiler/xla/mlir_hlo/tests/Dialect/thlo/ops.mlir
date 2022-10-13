// RUN: mlir-hlo-opt %s --split-input-file --allow-unregistered-dialect | \
// RUN: mlir-hlo-opt --verify-diagnostics --split-input-file \
// RUN:     --allow-unregistered-dialect | \
// RUN: FileCheck %s

func.func @concatenate(%arg1: tensor<?x?xf32>,
                       %arg2: tensor<?x?xf32>,
                       %dst: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %cat = thlo.concatenate
      ins(%arg1: tensor<?x?xf32>, %arg2: tensor<?x?xf32>)
      outs(%dst: tensor<?x?xf32>)
      { dimension = 0 : i64 }
  func.return %cat : tensor<?x?xf32>
}
// CHECK-LABEL: func @concatenate

// -----

func.func @concatenate_memref(%arg1: memref<?x?xf32>,
                              %arg2: memref<?x?xf32>,
                              %dst: memref<?x?xf32>) {
  thlo.concatenate
      ins(%arg1: memref<?x?xf32>, %arg2: memref<?x?xf32>)
      outs(%dst: memref<?x?xf32>)
      { dimension = 0 : i64 }
  func.return
}
// CHECK-LABEL: func @concatenate_memref

// -----

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

func.func @dynamic_broadcast_in_dim_memref(%arg: memref<?x?xf32>,
                                           %dst: memref<?x?x?xf32>) {
  thlo.dynamic_broadcast_in_dim
      ins(%arg: memref<?x?xf32>)
      outs(%dst: memref<?x?x?xf32>)
      broadcast_dimensions = [0, 2]
  func.return
}
// CHECK-LABEL: func @dynamic_broadcast_in_dim_memref

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

func.func @gather_memref(%arg: memref<100xf32>,
                         %indices: memref<42x1xi64>,
                         %dst: memref<42xf32>) {
  thlo.gather
      ins(%arg: memref<100xf32>, %indices: memref<42x1xi64>)
      outs(%dst: memref<42xf32>)
  func.return
}
// CHECK-LABEL: func @gather_memref

// -----

func.func @scatter(%indices: tensor<2x2xi32>, %updates: tensor<2x1x3xf32>,
    %init: tensor<3x3xf32>) -> tensor<3x3xf32> {
  %0 = thlo.scatter ins(%indices : tensor<2x2xi32>, %updates : tensor<2x1x3xf32>)
                    outs(%init : tensor<3x3xf32>)
                    (%in: f32, %out: f32) {
    %sum = arith.addf %in, %out : f32
    thlo.yield %sum : f32
  }
  return %0 : tensor<3x3xf32>
}
// CHECK-LABEL: func @scatter

// -----

func.func @scatter_memref(%indices: memref<2x2xi32>,
    %updates: memref<2x1x3xf32>, %init: memref<3x3xf32>) {
  thlo.scatter ins(%indices : memref<2x2xi32>, %updates : memref<2x1x3xf32>)
               outs(%init : memref<3x3xf32>)
               (%in: f32, %out: f32) {
    %sum = arith.addf %in, %out : f32
    thlo.yield %sum : f32
  }
  func.return
}
// CHECK-LABEL: func @scatter_memref

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

func.func @transpose_memref(%input: memref<16x32x64xf32>,
                            %init: memref<32x64x16xf32>) {
  thlo.transpose
      ins(%input:memref<16x32x64xf32>)
      outs(%init:memref<32x64x16xf32>)
      permutation = [1, 2, 0]
  func.return
}
// CHECK-LABEL: func @transpose_memref

// -----

func.func @transpose_unknown_dimensions(%input: tensor<16x?xf32>,
    %init: tensor<64x?xf32>) -> tensor<64x?xf32> {
  %transpose = thlo.transpose
      ins(%input:tensor<16x?xf32>)
      outs(%init:tensor<64x?xf32>)
      permutation = [1, 0]
  func.return %transpose : tensor<64x?xf32>
}
// CHECK-LABEL: func @transpose_unknown_dimensions

// -----

func.func @transpose_unknown_dimensions_memref(%input: memref<16x?xf32>,
                                               %init: memref<64x?xf32>) {
  thlo.transpose
      ins(%input:memref<16x?xf32>)
      outs(%init:memref<64x?xf32>)
      permutation = [1, 0]
  func.return
}
// CHECK-LABEL: func @transpose_unknown_dimensions_memref

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

func.func @reduction_memref(%input: memref<16x32x64xf32>,
                            %init: memref<16x64xf32>) {
  thlo.reduction
      ins(%input:memref<16x32x64xf32>)
      outs(%init:memref<16x64xf32>)
      dimensions = [1]
      (%in: f32, %out: f32) {
        %0 = arith.addf %in, %out: f32
        thlo.yield %0: f32
      }
  func.return
}
// CHECK-LABEL: func @reduction_memref

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

func.func @variadic_reduction_memref(%input1: memref<16x32x64xf32>,
    %init1: memref<16x64xf32>, %input2: memref<16x32x64xi64>,
    %init2: memref<16x64xi64>) {
  thlo.reduction
      ins(%input1:memref<16x32x64xf32>, %input2:memref<16x32x64xi64>)
      outs(%init1:memref<16x64xf32>, %init2:memref<16x64xi64>)
      dimensions = [1]
      (%in1: f32, %in2: i64, %out1: f32, %out2: i64) {
        %0 = arith.addf %in1, %out1: f32
        %1 = arith.addi %in2, %out2: i64
        thlo.yield %0, %1: f32, i64
      }
  func.return
}
// CHECK-LABEL: func @variadic_reduction_memref

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

func.func @map_binary_memref(%lhs: memref<64xf32>, %rhs: memref<64xf32>,
                      %init: memref<64xf32>) {
   thlo.map
      ins(%lhs:memref<64xf32>, %rhs:memref<64xf32>)
      outs(%init:memref<64xf32>)
      (%lhs_elem: f32, %rhs_elem: f32) {
        %0 = arith.addf %lhs_elem, %rhs_elem: f32
        thlo.yield %0: f32
      }
  func.return
}
// CHECK-LABEL: func @map_binary_memref

// -----

func.func @map_unary(%input: tensor<64xf32>, %init: tensor<64xf32>) -> tensor<64xf32> {
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

// -----

func.func @map_unary_memref(%input: memref<64xf32>, %init: memref<64xf32>) {
   thlo.map
      ins(%input:memref<64xf32>)
      outs(%init:memref<64xf32>)
      (%input_elem: f32) {
        %0 = math.absf %input_elem: f32
        thlo.yield %0: f32
      }
  func.return
}
// CHECK-LABEL: func @map_unary_memref

// -----

func.func @sort(%input1: tensor<?x?xf32>, %input2: tensor<?x?xi32>,
                %init1: tensor<?x?xf32>, %init2: tensor<?x?xi32>)
    -> (tensor<?x?xf32>, tensor<?x?xi32>) {
  %sorted1, %sorted2 = thlo.sort
      ins(%input1: tensor<?x?xf32>, %input2: tensor<?x?xi32>)
      outs(%init1: tensor<?x?xf32>, %init2: tensor<?x?xi32>)
      { dimension = 0 : i64, is_stable = true }
      (%e11: f32, %e12: f32, %e21: i32, %e22: i32) {
        %gt = arith.cmpf ogt, %e11, %e12: f32
        thlo.yield %gt : i1
      }
  func.return %sorted1, %sorted2 : tensor<?x?xf32>, tensor<?x?xi32>
}
// CHECK-LABEL: func @sort

// -----

func.func @sort_memref(%input1: memref<?x?xf32>, %input2: memref<?x?xi32>,
                       %init1: memref<?x?xf32>, %init2: memref<?x?xi32>) {
  thlo.sort
      ins(%input1: memref<?x?xf32>, %input2: memref<?x?xi32>)
      outs(%init1: memref<?x?xf32>, %init2: memref<?x?xi32>)
      { dimension = 0 : i64, is_stable = true }
      (%e11: f32, %e12: f32, %e21: i32, %e22: i32) {
        %gt = arith.cmpf ogt, %e11, %e12: f32
        thlo.yield %gt : i1
      }
  func.return
}
// CHECK-LABEL: func @sort_memref