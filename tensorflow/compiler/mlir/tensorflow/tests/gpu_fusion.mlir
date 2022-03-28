// RUN: tf-opt %s -tf-gpu-op-fusion | FileCheck %s

// Test the op-fusion pass specific to the GPU target.

// CHECK-LABEL: func @FusedBatchNormRelu
func.func @FusedBatchNormRelu(%arg0: tensor<8x8x8x8xf32>, %arg1: tensor<8xf32>, %arg2: tensor<8xf32>, %arg3: tensor<8xf32>, %arg4: tensor<8xf32>) -> (tensor<8x8x8x8xf32>) {
// CHECK-NEXT: %[[Y:[a-z0-9]*]], {{.*}}_FusedBatchNormEx
// CHECK-NEXT: return %[[Y]]
  %y:6 = "tf.FusedBatchNormV3"(%arg0, %arg1, %arg2, %arg3, %arg4) {T = "tfdtype$DT_FLOAT", data_format = "NHWC", epsilon = 0.001 : f32, is_training = false} : (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>) -> (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>)
  %relu = "tf.Relu"(%y#0) : (tensor<8x8x8x8xf32>) -> tensor<8x8x8x8xf32>
  func.return %relu : tensor<8x8x8x8xf32>
}

// CHECK-LABEL: func @FusedBatchNormAddRelu
func.func @FusedBatchNormAddRelu(%arg0: tensor<8x8x8x8xf32>, %arg1: tensor<8xf32>, %arg2: tensor<8xf32>, %arg3: tensor<8xf32>, %arg4: tensor<8xf32>) -> (tensor<8x8x8x8xf32>) {
// CHECK-NEXT: %[[Y:[a-z0-9]*]], {{.*}}_FusedBatchNormEx
// CHECK-NEXT: return %[[Y]]
  %y:6 = "tf.FusedBatchNormV3"(%arg0, %arg1, %arg2, %arg3, %arg4) {T = "tfdtype$DT_FLOAT", data_format = "NHWC", epsilon = 0.001 : f32, is_training = false} : (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>) -> (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>)
  %add = "tf.AddV2"(%arg0, %y#0) : (tensor<8x8x8x8xf32>, tensor<8x8x8x8xf32>) -> tensor<8x8x8x8xf32>
  %relu = "tf.Relu"(%add) : (tensor<8x8x8x8xf32>) -> tensor<8x8x8x8xf32>
  func.return %relu : tensor<8x8x8x8xf32>
}

// CHECK-LABEL: func @FusedBatchNormAddReluTwoUses
func.func @FusedBatchNormAddReluTwoUses(%arg0: tensor<8x8x8x8xf32>, %arg1: tensor<8xf32>, %arg2: tensor<8xf32>, %arg3: tensor<8xf32>, %arg4: tensor<8xf32>) -> (tensor<8x8x8x8xf32>, tensor<8x8x8x8xf32>) {
// Since the tf.AddV2 op has two uses, we have a _FusedBatchNormEx without the
// Relu activation and we only fuse the add.
// CHECK-NEXT: %[[Y:[a-z0-9]*]], {{.*}}_FusedBatchNormEx
// CHECK-NEXT: %[[relu:[a-z0-9]*]] ={{.*}}Relu"(%[[Y]]
// CHECK-NEXT: return %[[relu]]
  %y:6 = "tf.FusedBatchNormV3"(%arg0, %arg1, %arg2, %arg3, %arg4) {T = "tfdtype$DT_FLOAT", data_format = "NHWC", epsilon = 0.001 : f32, is_training = false} : (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>) -> (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>)
  %add = "tf.AddV2"(%arg0, %y#0) : (tensor<8x8x8x8xf32>, tensor<8x8x8x8xf32>) -> tensor<8x8x8x8xf32>
  %relu = "tf.Relu"(%add) : (tensor<8x8x8x8xf32>) -> tensor<8x8x8x8xf32>
  func.return %relu, %add  : tensor<8x8x8x8xf32>, tensor<8x8x8x8xf32>
}

// CHECK-LABEL: func @TrainingFusedBatchNormRelu
func.func @TrainingFusedBatchNormRelu(%arg0: tensor<8x8x8x8xf32>, %arg1: tensor<8xf32>, %arg2: tensor<8xf32>, %arg3: tensor<8xf32>, %arg4: tensor<8xf32>) -> (tensor<8x8x8x8xf32>) {
  // We don't fuse in training right now
// CHECK-NEXT: %[[Y:[a-z0-9]*]], {{.*}}FusedBatchNorm
// CHECK-NEXT: %[[relu:[a-z0-9]*]] ={{.*}}Relu"(%[[Y]]
// CHECK-NEXT: return %[[relu]]
  %y:6 = "tf.FusedBatchNormV3"(%arg0, %arg1, %arg2, %arg3, %arg4) {T = "tfdtype$DT_FLOAT", data_format = "NHWC", epsilon = 0.001 : f32, is_training = true} : (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>) -> (tensor<8x8x8x8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>, tensor<8xf32>)
  %relu = "tf.Relu"(%y#0) : (tensor<8x8x8x8xf32>) -> tensor<8x8x8x8xf32>
  func.return %relu : tensor<8x8x8x8xf32>
}

