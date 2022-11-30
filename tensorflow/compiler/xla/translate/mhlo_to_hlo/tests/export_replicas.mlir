// RUN: xla-translate -split-input-file -mlir-hlo-to-hlo-text %s | FileCheck %s

// Tests that the exported HLO module keeps parameter replication annotation.

// CHECK:  HloModule
func.func @main(%arg0: tensor<16x16xf32>, %arg1: tensor<16x16xf32> {mhlo.is_same_data_across_replicas}) -> tensor<16x16xf32> {
  %0 = "mhlo.add"(%arg0, %arg1) : (tensor<16x16xf32>, tensor<16x16xf32>) -> tensor<16x16xf32>
  func.return %0 : tensor<16x16xf32>
}

// CHECK:  ENTRY
// CHECK:  %[[ARG0:.*]] = f32[16,16] parameter(0)
// CHECK-NOT: parameter_replication={true}
// CHECK:  %[[ARG1:.*]] = f32[16,16] parameter(1), parameter_replication={true}
// CHECK:  ROOT %[[RESULT:.*]] = f32[16,16] add(f32[16,16] %[[ARG0]], f32[16,16] %[[ARG1]])
