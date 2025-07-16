// RUN: stablehlo-quant-opt %s -split-input-file -tf-stablehlo-unfuse-mhlo-batch-norm | FileCheck %s

// CHECK-LABEL: @unfuse_batch_norm
// CHECK-SAME: %[[X:[^:[:space:]]+]]
// CHECK-SAME: %[[SCALE:[^:[:space:]]+]]
// CHECK-SAME: %[[OFFSET:[^:[:space:]]+]]
// CHECK-SAME: %[[MEAN:[^:[:space:]]+]]
// CHECK-SAME: %[[VARIANCE:[^:[:space:]]+]]
func.func @unfuse_batch_norm(
    %x: tensor<4x256xf32>, %scale: tensor<256xf32>, %offset: tensor<256xf32>,
    %mean: tensor<256xf32>, %variance: tensor<256xf32>)
    -> (tensor<4x256xf32>) {
  // CHECK-DAG: %[[EPS_BCAST:.+]] = mhlo.constant dense<1.001000e-05> : tensor<256xf32>
  // CHECK-DAG: %[[VARIANCE_EPS:.+]] = mhlo.add %[[VARIANCE]], %[[EPS_BCAST]] : tensor<256xf32>
  // CHECK-DAG: %[[STDDEV:.+]] = mhlo.sqrt %[[VARIANCE_EPS]] : tensor<256xf32>
  // CHECK-DAG: %[[STDDEV_BCAST:.+]] = "mhlo.broadcast_in_dim"(%[[STDDEV]]) <{broadcast_dimensions = dense<1> : tensor<1xi64>}> : (tensor<256xf32>) -> tensor<4x256xf32>
  // CHECK-DAG: %[[SCALE_BCAST:.+]] = "mhlo.broadcast_in_dim"(%[[SCALE]]) <{broadcast_dimensions = dense<1> : tensor<1xi64>}> : (tensor<256xf32>) -> tensor<4x256xf32>
  // CHECK-DAG: %[[OFFSET_BCAST:.+]] = "mhlo.broadcast_in_dim"(%[[OFFSET]]) <{broadcast_dimensions = dense<1> : tensor<1xi64>}> : (tensor<256xf32>) -> tensor<4x256xf32>
  // CHECK-DAG: %[[MEAN_BCAST:.+]] = "mhlo.broadcast_in_dim"(%[[MEAN]]) <{broadcast_dimensions = dense<1> : tensor<1xi64>}> : (tensor<256xf32>) -> tensor<4x256xf32>
  // CHECK: %[[X_CENTER:.+]] = mhlo.subtract %[[X]], %[[MEAN_BCAST]] : tensor<4x256xf32>
  // CHECK: %[[X_SCALED:.+]] = mhlo.multiply %[[X_CENTER]], %[[SCALE_BCAST]] : tensor<4x256xf32>
  // CHECK: %[[X_NORMED:.+]] = mhlo.divide %[[X_SCALED]], %[[STDDEV_BCAST]] : tensor<4x256xf32>
  // CHECK: %[[RESULT:.+]] = mhlo.add %[[X_NORMED]], %[[OFFSET_BCAST]] : tensor<4x256xf32>
  %0 = "mhlo.batch_norm_inference"(%x, %scale, %offset, %mean, %variance)
      {epsilon = 1.001000e-05 : f32, feature_index = 1 : i64} :
      (tensor<4x256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>,
        tensor<256xf32>) -> tensor<4x256xf32>
  // CHECK-DAG: return %[[RESULT]]
  func.return %0 : tensor<4x256xf32>
}
