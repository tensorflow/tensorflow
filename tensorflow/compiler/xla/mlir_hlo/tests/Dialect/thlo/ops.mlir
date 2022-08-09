// RUN: mlir-hlo-opt %s --split-input-file --allow-unregistered-dialect | \
// RUN: mlir-hlo-opt --verify-diagnostics --split-input-file \
// RUN:     --allow-unregistered-dialect | \
// RUN: FileCheck %s

func.func @dynamic_broadcast_in_dim(%arg: tensor<?x?xf32>,
                                    %dst: tensor<?x?x?xf32>) {
  %bcast = thlo.dynamic_broadcast_in_dim
      ins(%arg: tensor<?x?xf32>)
      outs(%dst: tensor<?x?x?xf32>) {
        broadcast_dimensions = [:i64 0, 2]
      }
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
