// RUN: tfg-transforms-opt --tfg-remapper=enable-onednn-patterns %s | FileCheck %s

// -----

// CHECK-LABEL: tfg.func @conv2d_test
tfg.func @conv2d_test() {
  // CHECK: %[[PLACEHOLDER:.*]], {{.*}} name("input_tensor")
  %Placeholder, %ctl = Placeholder device("/device:CPU:0") name("input_tensor") {dtype = f32, shape = #tf_type.shape<1x3x3x1>} : () -> (tensor<*xf32>)
  // CHECK: %[[FILTER:.*]], {{.*}} name("Const")
  %Const, %ctl_0 = Const device("/device:CPU:0") name("Const") {dtype = f32, value = dense<[[[[1.11986792, -3.0272491]]]]> : tensor<1x1x1x2xf32>} : () -> (tensor<*xf32>)
  // CHECK: %[[BIAS:.*]], {{.*}} name("Const_1")
  %Const_1, %ctl_2 = Const device("/device:CPU:0") name("Const_1") {dtype = f32, value = dense<[0.531091094, -0.719168067]> : tensor<2xf32>} : () -> (tensor<*xf32>)
  %Conv2D, %ctl_3 = Conv2D(%Placeholder, %Const) device("/device:CPU:0") name("Conv2D") {T = f32, data_format = "NHWC", dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "SAME", strides = [1, 1, 1, 1], use_cudnn_on_gpu = true} : (tensor<*xf32>, tensor<*xf32>) -> (tensor<*xf32>)
  // CHECK: _FusedConv2D(%[[PLACEHOLDER]], %[[FILTER]], %[[BIAS]]) {{.*}} name("BiasAdd") {{.*}} fused_ops = ["BiasAdd"]
  %BiasAdd, %ctl_4 = BiasAdd(%Conv2D, %Const_1) device("/device:CPU:0") name("BiasAdd") {T = f32, data_format = "NHWC"} : (tensor<*xf32>, tensor<*xf32>) -> (tensor<*xf32>)
  // CHECK: _FusedConv2D(%[[PLACEHOLDER]], %[[FILTER]], %[[BIAS]]) {{.*}} name("LeakyRelu") {{.*}} fused_ops = ["BiasAdd", "LeakyRelu"]
  %LeakyRelu, %ctl_5 = LeakyRelu(%BiasAdd) device("/device:CPU:0") name("LeakyRelu") {T = f32, alpha = 3.000000e-01 : f32} : (tensor<*xf32>) -> (tensor<*xf32>)
  %Conv2D_6, %ctl_7 = Conv2D(%Placeholder, %Const) device("/device:CPU:0") name("Conv2D_1") {T = f32, data_format = "NHWC", dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "VALID", strides = [1, 1, 1, 1], use_cudnn_on_gpu = true} : (tensor<*xf32>, tensor<*xf32>) -> (tensor<*xf32>)
  // CHECK: %[[BIAS_ADD:.*]], {{.*}} _FusedConv2D(%[[PLACEHOLDER]], %[[FILTER]], %[[BIAS]]) {{.*}} name("BiasAdd_1")
  %BiasAdd_8, %ctl_9 = BiasAdd(%Conv2D_6, %Const_1) device("/device:CPU:0") name("BiasAdd_1") {T = f32, data_format = "NHWC"} : (tensor<*xf32>, tensor<*xf32>) -> (tensor<*xf32>)
  // CHECK: Relu(%[[BIAS_ADD]]) {{.*}} name("Relu")
  %Relu, %ctl_10 = Relu(%BiasAdd_8) device("/device:CPU:0") name("Relu") {T = f32} : (tensor<*xf32>) -> (tensor<*xf32>)
  // CHECK: Elu(%[[BIAS_ADD]]) {{.*}} name("Elu")
  %Elu, %ctl_11 = Elu(%BiasAdd_8) device("/device:CPU:0") name("Elu") {T = f32} : (tensor<*xf32>) -> (tensor<*xf32>)
  // CHECK: Sigmoid(%[[BIAS_ADD]]) {{.*}} name("Sigmoid")
  %Sigmoid, %ctl_12 = Sigmoid(%BiasAdd_8) device("/device:CPU:0") name("Sigmoid") {T = f32} : (tensor<*xf32>) -> (tensor<*xf32>)
  return
}

// -----

// CHECK-LABEL: tfg.func @conv3d_test
tfg.func @conv3d_test() {
  // CHECK: %[[PLACEHOLDER:.*]], {{.*}} name("input_tensor")
  %Placeholder, %ctl = Placeholder device("/device:CPU:0") name("input_tensor") {dtype = f32, shape = #tf_type.shape<1x1x3x3x1>} : () -> (tensor<*xf32>)
  // CHECK: %[[FILTER:.*]], {{.*}} name("Const")
  %Const, %ctl_0 = Const device("/device:CPU:0") name("Const") {dtype = f32, value = dense<[[[[[2.08520603, 0.201729089]]]]]> : tensor<1x1x1x1x2xf32>} : () -> (tensor<*xf32>)
  // CHECK: %[[BIAS:.*]], {{.*}} name("Const_1")
  %Const_1, %ctl_2 = Const device("/device:CPU:0") name("Const_1") {dtype = f32, value = dense<[0.00119970727, 0.199058324]> : tensor<2xf32>} : () -> (tensor<*xf32>)
  %Conv3D, %ctl_3 = Conv3D(%Placeholder, %Const) device("/device:CPU:0") name("Conv3D") {T = f32, data_format = "NDHWC", dilations = [1, 1, 1, 1, 1], padding = "VALID", strides = [1, 1, 1, 1, 1]} : (tensor<*xf32>, tensor<*xf32>) -> (tensor<*xf32>)
  // CHECK: _FusedConv3D(%[[PLACEHOLDER]], %[[FILTER]], %[[BIAS]]) {{.*}} name("BiasAdd") {{.*}} fused_ops = ["BiasAdd"]
  %BiasAdd, %ctl_4 = BiasAdd(%Conv3D, %Const_1) device("/device:CPU:0") name("BiasAdd") {T = f32, data_format = "NHWC"} : (tensor<*xf32>, tensor<*xf32>) -> (tensor<*xf32>)
  // CHECK: _FusedConv3D(%[[PLACEHOLDER]], %[[FILTER]], %[[BIAS]]) {{.*}} name("Relu") {{.*}} fused_ops = ["BiasAdd", "Relu"]
  %Relu, %ctl_5 = Relu(%BiasAdd) device("/device:CPU:0") name("Relu") {T = f32} : (tensor<*xf32>) -> (tensor<*xf32>)
  return
}

// -----

// CHECK-LABEL: tfg.func @depthwise_conv2d_test
tfg.func @depthwise_conv2d_test() {
  // CHECK: %[[PLACEHOLDER:.*]], {{.*}} name("input_tensor")
  %Placeholder, %ctl = Placeholder device("/device:CPU:0") name("input_tensor") {dtype = f32, shape = #tf_type.shape<1x2x3x2>} : () -> (tensor<*xf32>)
  // CHECK: %[[FILTER:.*]], {{.*}} name("Const")
  %Const, %ctl_0 = Const device("/device:CPU:0") name("Const") {dtype = f32, value = dense<[[[[-0.595111072, -1.49551904], [0.151230454, 0.00520740356]], [[-0.626356721, 0.508778572], [0.209191799, 0.146687463]]], [[[0.0205380265, -2.10263419], [0.10162171, 1.73773324]], [[0.723617255, -0.600047112], [-0.812727749, 0.183114871]]]]> : tensor<2x2x2x2xf32>} : () -> (tensor<*xf32>)
  %Const_1, %ctl_2 = Const device("/device:CPU:0") name("depthwise/Shape") {dtype = i32, value = dense<2> : tensor<4xi32>} : () -> (tensor<*xi32>)
  %Const_3, %ctl_4 = Const device("/device:CPU:0") name("depthwise/dilation_rate") {dtype = i32, value = dense<1> : tensor<2xi32>} : () -> (tensor<*xi32>)
  %DepthwiseConv2dNative, %ctl_5 = DepthwiseConv2dNative(%Placeholder, %Const) device("/device:CPU:0") name("depthwise") {T = f32, data_format = "NHWC", dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "VALID", strides = [1, 1, 1, 1]} : (tensor<*xf32>, tensor<*xf32>) -> (tensor<*xf32>)
  // CHECK: %[[BIAS:.*]], {{.*}} name("Const_1")
  %Const_6, %ctl_7 = Const device("/device:CPU:0") name("Const_1") {dtype = f32, value = dense<[-0.777153254, -0.696344554, 0.0879275202, -1.49649549]> : tensor<4xf32>} : () -> (tensor<*xf32>)
  // CHECK: _FusedDepthwiseConv2dNative(%[[PLACEHOLDER]], %[[FILTER]], %[[BIAS]]) {{.*}} name("BiasAdd") {{.*}} fused_ops = ["BiasAdd"]
  %BiasAdd, %ctl_8 = BiasAdd(%DepthwiseConv2dNative, %Const_6) device("/device:CPU:0") name("BiasAdd") {T = f32, data_format = "NHWC"} : (tensor<*xf32>, tensor<*xf32>) -> (tensor<*xf32>)
  // CHECK: _FusedDepthwiseConv2dNative(%[[PLACEHOLDER]], %[[FILTER]], %[[BIAS]]) {{.*}} name("Relu") {{.*}} fused_ops = ["BiasAdd", "Relu"]
  %Relu, %ctl_9 = Relu(%BiasAdd) device("/device:CPU:0") name("Relu") {T = f32} : (tensor<*xf32>) -> (tensor<*xf32>)
  return
}

// -----

// CHECK-LABEL: tfg.func @matmul_test
tfg.func @matmul_test() {
  // CHECK: %[[PLACEHOLDER:.*]], {{.*}} name("input_tensor")
  %Placeholder, %ctl = Placeholder device("/device:CPU:0") name("input_tensor") {dtype = f32, shape = #tf_type.shape<3x4>} : () -> (tensor<*xf32>)
  // CHECK: %[[WEIGHT:.*]], {{.*}} name("Const")
  %Const, %ctl_0 = Const device("/device:CPU:0") name("Const") {dtype = f32, value = dense<[[1.55347502, 2.01656532, 0.956115126, -0.508335888, 0.0327275023], [-0.287353367, -1.85766768, -0.841522634, 1.74347401, 0.0903762504], [0.0194547977, -1.15673554, -0.00433536153, -0.743040859, -0.340555519], [2.13576722, 0.604285777, -0.409727454, -0.367086798, -0.375431746]]> : tensor<4x5xf32>} : () -> (tensor<*xf32>)
  %MatMul, %ctl_1 = MatMul(%Placeholder, %Const) device("/device:CPU:0") name("MatMul") {T = f32, transpose_a = false, transpose_b = false} : (tensor<*xf32>, tensor<*xf32>) -> (tensor<*xf32>)
  // CHECK: %[[BIAS:.*]], {{.*}} name("Const_1")
  %Const_2, %ctl_3 = Const device("/device:CPU:0") name("Const_1") {dtype = f32, value = dense<[0.521312416, -0.748116672, 0.769415915, -1.03628337, 1.45927799]> : tensor<5xf32>} : () -> (tensor<*xf32>)
  // CHECK: _FusedMatMul(%[[PLACEHOLDER]], %[[WEIGHT]], %[[BIAS]]) {{.*}} name("BiasAdd") {{.*}} fused_ops = ["BiasAdd"]
  %BiasAdd, %ctl_4 = BiasAdd(%MatMul, %Const_2) device("/device:CPU:0") name("BiasAdd") {T = f32, data_format = "NHWC"} : (tensor<*xf32>, tensor<*xf32>) -> (tensor<*xf32>)
  // CHECK: _FusedMatMul(%[[PLACEHOLDER]], %[[WEIGHT]], %[[BIAS]]) {{.*}} name("Tanh") {{.*}} fused_ops = ["BiasAdd", "Tanh"]
  %Tanh, %ctl_5 = Tanh(%BiasAdd) device("/device:CPU:0") name("Tanh") {T = f32} : (tensor<*xf32>) -> (tensor<*xf32>)
  return
}
