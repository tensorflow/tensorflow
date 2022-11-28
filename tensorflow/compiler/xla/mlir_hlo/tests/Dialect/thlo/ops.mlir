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
      dimension = 0
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
      dimension = 0
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
                  %indices: tensor<42x1xindex>,
                  %dst: tensor<42xf32>) -> tensor<42xf32> {
  %gather = thlo.gather
      ins(%arg: tensor<100xf32>, %indices: tensor<42x1xindex>)
      outs(%dst: tensor<42xf32>)
  func.return %gather : tensor<42xf32>
}
// CHECK-LABEL: func @gather

// -----

func.func @gather_memref(%arg: memref<100xf32>,
                         %indices: memref<42x1xindex>,
                         %dst: memref<42xf32>) {
  thlo.gather
      ins(%arg: memref<100xf32>, %indices: memref<42x1xindex>)
      outs(%dst: memref<42xf32>)
  func.return
}
// CHECK-LABEL: func @gather_memref

// -----

func.func @scatter(%indices: tensor<2x2xindex>, %updates: tensor<2x1x3xf32>,
    %init: tensor<3x3xf32>) -> tensor<3x3xf32> {
  %0 = thlo.scatter ins(%indices : tensor<2x2xindex>,
                        %updates : tensor<2x1x3xf32>)
                    outs(%init : tensor<3x3xf32>)
                    (%in: f32, %out: f32) {
    %sum = arith.addf %in, %out : f32
    thlo.yield %sum : f32
  }
  return %0 : tensor<3x3xf32>
}
// CHECK-LABEL: func @scatter

// -----

func.func @scatter_memref(%indices: memref<2x2xindex>,
    %updates: memref<2x1x3xf32>, %init: memref<3x3xf32>) {
  thlo.scatter ins(%indices : memref<2x2xindex>, %updates : memref<2x1x3xf32>)
               outs(%init : memref<3x3xf32>)
               (%in: f32, %out: f32) {
    %sum = arith.addf %in, %out : f32
    thlo.yield %sum : f32
  }
  func.return
}
// CHECK-LABEL: func @scatter_memref

// -----

func.func @sort(%input1: tensor<?x?xf32>, %input2: tensor<?x?xi32>,
                %init1: tensor<?x?xf32>, %init2: tensor<?x?xi32>)
    -> (tensor<?x?xf32>, tensor<?x?xi32>) {
  %sorted1, %sorted2 = thlo.sort
      ins(%input1: tensor<?x?xf32>, %input2: tensor<?x?xi32>)
      outs(%init1: tensor<?x?xf32>, %init2: tensor<?x?xi32>)
      dimension = 0
      is_stable = true
      (%e11: f32, %e12: f32, %e21: i32, %e22: i32) {
        %gt = arith.cmpf ogt, %e11, %e12: f32
        thlo.yield %gt : i1
      }
  func.return %sorted1, %sorted2 : tensor<?x?xf32>, tensor<?x?xi32>
}
// CHECK-LABEL: func @sort
// CHECK:         %[[RES1:sorted0]], %[[RES2:sorted1]] = thlo.sort
// CHECK:         %[[LHS0:lhs0: f32]], %[[RHS0:rhs0: f32]],
// CHECK-SAME:    %[[LHS1:lhs1: i32]], %[[RHS1:rhs1: i32]]

// -----

func.func @sort_memref(%input1: memref<?x?xf32>, %input2: memref<?x?xi32>,
                       %init1: memref<?x?xf32>, %init2: memref<?x?xi32>) {
  thlo.sort
      ins(%input1: memref<?x?xf32>, %input2: memref<?x?xi32>)
      outs(%init1: memref<?x?xf32>, %init2: memref<?x?xi32>)
      dimension = 0
      is_stable = true
      (%e11: f32, %e12: f32, %e21: i32, %e22: i32) {
        %gt = arith.cmpf ogt, %e11, %e12: f32
        thlo.yield %gt : i1
      }
  func.return
}
// CHECK-LABEL: func @sort_memref
