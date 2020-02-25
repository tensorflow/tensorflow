// RUN: tf-opt %s -tf-layout-optimization=force-data-format=NHWC -verify-diagnostics | FileCheck %s --dump-input=always

// CHECK-LABEL: func @transpose_resnet_layer
func @transpose_resnet_layer(%arg0: tensor<?x224x224x3xf32>, // input
                             %arg1: tensor<64xf32>,          // batch_norm args
                             %arg2: tensor<256xf32>,          // batch_norm args
                             %arg3: tensor<7x7x3x64xf32>,    // conv filter #0
                             %arg4: tensor<1x1x64x256xf32>   // conv filter #1
                            ) -> tensor<?x256xf32> {

  // This is a simplified ResNet layer that gets input in NHWC format, converts
  // it to NCHW before padding, and does all computations in NCHW (this is the
  // default setup for ResNet model trained in fp32 on GPU).
  //
  // To be able to use Tensor Cores on latest NVIDIA GPUs this model has to be
  // converted to NHWC data format.

  // Padding in spatial dimension (NCHW)
  %0 = "tf.Const"() {value = dense<[[0, 0], [0, 0], [3, 3], [3, 3]]> : tensor<4x2xi32>} : () -> tensor<4x2xi32>

  // Reduce over spatial dimensions (NCHW)
  %1 = "tf.Const"() {value = dense<[2, 3]> : tensor<2xi32>} : () -> tensor<2xi32>

  // Transpose input: NHWC -> NCHW
  %2 = "tf.Const"() {value = dense<[0, 3, 1, 2]> : tensor<4xi32>} : () -> tensor<4xi32>
  %3 = "tf.Transpose"(%arg0, %2) : (tensor<?x224x224x3xf32>, tensor<4xi32>) -> tensor<?x3x224x224xf32>

  // Pad spatial dimensions.
  %4 = "tf.Pad"(%3, %0) : (tensor<?x3x224x224xf32>, tensor<4x2xi32>) -> tensor<?x3x230x230xf32>

  // Shuffled paddings.
  // CHECK: %[[PADDINGS:[0-9]*]] = "tf.Const"(){{.*}}[0, 0], [3, 3], [3, 3], [0, 0]

  // Pad input with new paddings.
  // CHECK: %[[PAD:[0-9]*]] = "tf.Pad"(%arg0, %[[PADDINGS]])
  // CHECK-SAME: (tensor<?x224x224x3xf32>, tensor<4x2xi32>) -> tensor<?x230x230x3xf32>

  // ------------------------------------------------------------------------ //
  // Convolution layer #0.
  // ------------------------------------------------------------------------ //
  %5 = "tf.Conv2D"(%4, %arg3)
        {
          data_format = "NCHW",
          dilations = [1, 1, 1, 1],
          explicit_paddings = [],
          padding = "VALID",
          strides = [1, 1, 2, 2]
        } : (tensor<?x3x230x230xf32>, tensor<7x7x3x64xf32>) -> tensor<?x64x112x112xf32>

  // CHECK: %[[CONV0:[0-9]*]] = "tf.Conv2D"
  // CHECK-SAME %[[PAD]]
  // CHECK-SAME: data_format = "NHWC"
  // CHECK-SAME: strides = [1, 2, 2, 1]

  %6, %batch_mean, %batch_variance, %reserved_1, %reserved_2, %reserved_3 =
       "tf.FusedBatchNormV3"(%5, %arg1, %arg1, %arg1, %arg1)
       {
         data_format = "NCHW",
         epsilon = 1.001000e-05 : f32,
         is_training = false
       } : (tensor<?x64x112x112xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>)
        -> (tensor<?x64x112x112xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<*xf32>)

  // CHECK: "tf.FusedBatchNormV3"
  // CHECK-SAME: data_format = "NHWC"

  %7 = "tf.Relu"(%6) : (tensor<?x64x112x112xf32>) -> tensor<?x64x112x112xf32>
  %8 = "tf.MaxPool"(%7)
       {
         data_format = "NCHW",
         ksize = [1, 1, 3, 3],
         padding = "SAME",
         strides = [1, 1, 2, 2]
       } : (tensor<?x64x112x112xf32>) -> tensor<?x64x56x56xf32>

  // CHECK: %[[MAX_POOL:[0-9]*]] = "tf.MaxPool"
  // CHECK-SAME: data_format = "NHWC"
  // CHECK-SAME: ksize = [1, 3, 3, 1]
  // CHECK-SAME: strides = [1, 2, 2, 1]

  // ------------------------------------------------------------------------ //
  // Convolution layer #1.
  // ------------------------------------------------------------------------ //
  %9 = "tf.Conv2D"(%8, %arg4)
       {
         data_format = "NCHW",
         dilations = [1, 1, 1, 1],
         explicit_paddings = [],
         padding = "VALID",
         strides = [1, 1, 1, 1]
       } : (tensor<?x64x56x56xf32>, tensor<1x1x64x256xf32>) -> tensor<?x256x56x56xf32>

  // CHECK: %[[CONV1:[0-9]*]] = "tf.Conv2D"(%[[MAX_POOL]], %arg4)
  // CHECK-SAME: data_format = "NHWC"

  %10, %batch_mean_1, %batch_variance_1, %reserved_1_1, %reserved_1_2, %reserved_1_3 =
       "tf.FusedBatchNormV3"(%9, %arg2, %arg2, %arg2, %arg2)
       {
         data_format = "NCHW",
         epsilon = 1.001000e-05 : f32
       } : (tensor<?x256x56x56xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>)
        -> (tensor<?x256x56x56xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<*xf32>)

  // CHECK: %[[BATCH_NORM1:[_a-z0-9]*]], {{.*}} = "tf.FusedBatchNormV3"
  // CHECK-SAME: %[[CONV1]]
  // CHECK-SAME: data_format = "NHWC"

  // ------------------------------------------------------------------------ //
  // Convolution layer #2.
  // ------------------------------------------------------------------------ //
  %11 = "tf.Conv2D"(%8, %arg4)
       {
         data_format = "NCHW",
         dilations = [1, 1, 1, 1],
         explicit_paddings = [],
         padding = "VALID",
         strides = [1, 1, 1, 1]
       } : (tensor<?x64x56x56xf32>, tensor<1x1x64x256xf32>) -> tensor<?x256x56x56xf32>

  // CHECK: %[[CONV2:[0-9]*]] = "tf.Conv2D"(%[[MAX_POOL]], %arg4)
  // CHECK-SAME: data_format = "NHWC"

  %12, %batch_mean_2, %batch_variance_2, %reserved_2_1, %reserved_2_2, %reserved_2_3 =
       "tf.FusedBatchNormV3"(%11, %arg2, %arg2, %arg2, %arg2)
       {
         data_format = "NCHW",
         epsilon = 1.001000e-05 : f32
       } : (tensor<?x256x56x56xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>)
        -> (tensor<?x256x56x56xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<*xf32>)

  // CHECK: %[[BATCH_NORM2:[_a-z0-9]*]], {{.*}} = "tf.FusedBatchNormV3"
  // CHECK-SAME: %[[CONV2]]
  // CHECK-SAME: data_format = "NHWC"

  // ------------------------------------------------------------------------ //
  // Add results of convolution layers #1 and #2.
  // ------------------------------------------------------------------------ //

  %14 = "tf.AddV2"(%10, %12) : (tensor<?x256x56x56xf32>, tensor<?x256x56x56xf32>) -> tensor<?x256x56x56xf32>
  %15 = "tf.Relu"(%14) : (tensor<?x256x56x56xf32>) -> tensor<?x256x56x56xf32>

  // CHECK: %[[ADD:[0-9]*]] = "tf.AddV2"(%[[BATCH_NORM1]], %[[BATCH_NORM2]])
  // CHECK: %[[RELU:[0-9]*]] = "tf.Relu"(%[[ADD]])

  // Reduce spatial dimensions
  %16 = "tf.Mean"(%15, %1) : (tensor<?x256x56x56xf32>, tensor<2xi32>) -> tensor<?x256xf32>

  // Mean should compute reduction over NHWC spatial dimensions.
  // CHECK: %[[MEAN_DIMS:[0-9]*]] = "tf.Const"() {value = dense<[1, 2]> : tensor<2xi32>}
  // CHECK: %[[MEAN:[0-9]*]] = "tf.Mean"(%[[RELU]], %[[MEAN_DIMS]])
  // CHECK-SAME: (tensor<?x56x56x256xf32>, tensor<2xi32>) -> tensor<?x256xf32>
  // CHECK: return %[[MEAN]] : tensor<?x256xf32>

  return %16 : tensor<?x256xf32>
}

