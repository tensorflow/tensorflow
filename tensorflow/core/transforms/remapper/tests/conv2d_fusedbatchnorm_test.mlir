// RUN: tfg-transforms-opt --tfg-remapper %s | FileCheck %s

module {
  tfg.graph #tf_type.version<producer = 1133, min_consumer = 0> {
    %Placeholder, %ctl = Placeholder device("/device:CPU:0") name("input") {dtype = f32, shape = #tf_type.shape<2x8x8x24>} : () -> (tensor<*xf32>)
    //CHECK: %[[INPUT:.*]], {{.*}} name("input")
    %Placeholder_0, %ctl_1 = Placeholder device("/device:CPU:0") name("filter") {dtype = f32, shape = #tf_type.shape<3x3x24x16>} : () -> (tensor<*xf32>)
    //CHECK: %[[FILTER:.*]], {{.*}} name("filter")
    %Conv2D, %ctl_2 = Conv2D(%Placeholder, %Placeholder_0) device("/device:CPU:0") name("conv") {T = f32, data_format = "NHWC", dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "SAME", strides = [1, 1, 1, 1], use_cudnn_on_gpu = true} : (tensor<*xf32>, tensor<*xf32>) -> (tensor<*xf32>)
    %Placeholder_3, %ctl_4 = Placeholder device("/device:CPU:0") name("scale") {dtype = f32, shape = #tf_type.shape<*>} : () -> (tensor<*xf32>)
    //CHECK: %[[SCALE:.*]], {{.*}} name("scale")
    %Placeholder_5, %ctl_6 = Placeholder device("/device:CPU:0") name("offset") {dtype = f32, shape = #tf_type.shape<*>} : () -> (tensor<*xf32>)
    //CHECK: %[[OFFSET:.*]], {{.*}} name("offset")
    %Placeholder_7, %ctl_8 = Placeholder device("/device:CPU:0") name("mean") {dtype = f32, shape = #tf_type.shape<*>} : () -> (tensor<*xf32>)
    //CHECK: %[[MEAN:.*]], {{.*}} name("mean")
    %Placeholder_9, %ctl_10 = Placeholder device("/device:CPU:0") name("var") {dtype = f32, shape = #tf_type.shape<*>} : () -> (tensor<*xf32>)
    //CHECK: %[[VAR:.*]], {{.*}} name("var")
    %FusedBatchNormV3:6, %ctl_11 = FusedBatchNormV3(%Conv2D, %Placeholder_3, %Placeholder_5, %Placeholder_7, %Placeholder_9) device("/device:CPU:0") name("fused_batch_norm") {T = f32, U = f32, data_format = "NHWC", epsilon = 1.000000e-01 : f32, exponential_avg_factor = 1.000000e+00 : f32, is_training = false} : (tensor<*xf32>, tensor<*xf32>, tensor<*xf32>, tensor<*xf32>, tensor<*xf32>) -> (tensor<*xf32>, tensor<*xf32>, tensor<*xf32>, tensor<*xf32>, tensor<*xf32>, tensor<*xf32>)
    // CHECK: _FusedConv2D(%[[INPUT]], %[[FILTER]], %[[SCALE]], %[[OFFSET]], %[[MEAN]], %[[VAR]]) {{.*}} name("fused_batch_norm") {{.*}} fused_ops = ["FusedBatchNorm"]
  }
}

module {
  tfg.graph #tf_type.version<producer = 1133, min_consumer = 0> {
    %Placeholder, %ctl = Placeholder device("/device:CPU:0") name("input") {dtype = f32, shape = #tf_type.shape<2x8x8x24>} : () -> (tensor<*xf32>)
    //CHECK: %[[INPUT:.*]], {{.*}} name("input")
    %Placeholder_0, %ctl_1 = Placeholder device("/device:CPU:0") name("filter") {dtype = f32, shape = #tf_type.shape<3x3x24x16>} : () -> (tensor<*xf32>)
    //CHECK: %[[FILTER:.*]], {{.*}} name("filter")
    %Conv2D, %ctl_2 = Conv2D(%Placeholder, %Placeholder_0) device("/device:CPU:0") name("conv") {T = f32, data_format = "NHWC", dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "SAME", strides = [1, 1, 1, 1], use_cudnn_on_gpu = true} : (tensor<*xf32>, tensor<*xf32>) -> (tensor<*xf32>)
    %Placeholder_3, %ctl_4 = Placeholder device("/device:CPU:0") name("scale") {dtype = f32, shape = #tf_type.shape<*>} : () -> (tensor<*xf32>)
    //CHECK: %[[SCALE:.*]], {{.*}} name("scale")
    %Placeholder_5, %ctl_6 = Placeholder device("/device:CPU:0") name("offset") {dtype = f32, shape = #tf_type.shape<*>} : () -> (tensor<*xf32>)
    //CHECK: %[[OFFSET:.*]], {{.*}} name("offset")
    %Placeholder_7, %ctl_8 = Placeholder device("/device:CPU:0") name("mean") {dtype = f32, shape = #tf_type.shape<*>} : () -> (tensor<*xf32>)
    //CHECK: %[[MEAN:.*]], {{.*}} name("mean")
    %Placeholder_9, %ctl_10 = Placeholder device("/device:CPU:0") name("var") {dtype = f32, shape = #tf_type.shape<*>} : () -> (tensor<*xf32>)
    //CHECK: %[[VAR:.*]], {{.*}} name("var")
    %FusedBatchNormV2:5, %ctl_11 = FusedBatchNormV2(%Conv2D, %Placeholder_3, %Placeholder_5, %Placeholder_7, %Placeholder_9) device("/device:CPU:0") name("fused_batch_norm") {T = f32, U = f32, data_format = "NHWC", epsilon = 1.000000e-01 : f32, exponential_avg_factor = 1.000000e+00 : f32, is_training = false} : (tensor<*xf32>, tensor<*xf32>, tensor<*xf32>, tensor<*xf32>, tensor<*xf32>) -> (tensor<*xf32>, tensor<*xf32>, tensor<*xf32>, tensor<*xf32>, tensor<*xf32>)
    %Relu, %ctl_12 = Relu(%FusedBatchNormV2#0) device("/device:CPU:0") name("relu") {T = f32} : (tensor<*xf32>) -> (tensor<*xf32>)
    // CHECK: _FusedConv2D(%[[INPUT]], %[[FILTER]], %[[SCALE]], %[[OFFSET]], %[[MEAN]], %[[VAR]]) {{.*}} name("relu") {{.*}} fused_ops = ["FusedBatchNorm", "Relu"]
  }
}

module {
  tfg.graph #tf_type.version<producer = 1133, min_consumer = 0> {
    %Placeholder, %ctl = Placeholder device("/device:CPU:0") name("input") {dtype = f32, shape = #tf_type.shape<2x8x8x24>} : () -> (tensor<*xf32>)
    //CHECK: %[[INPUT:.*]], {{.*}} name("input")
    %Placeholder_0, %ctl_1 = Placeholder device("/device:CPU:0") name("filter") {dtype = f32, shape = #tf_type.shape<3x3x24x16>} : () -> (tensor<*xf32>)
    //CHECK: %[[FILTER:.*]], {{.*}} name("filter")
    %Conv2D, %ctl_2 = Conv2D(%Placeholder, %Placeholder_0) device("/device:CPU:0") name("conv") {T = f32, data_format = "NHWC", dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "SAME", strides = [1, 1, 1, 1], use_cudnn_on_gpu = true} : (tensor<*xf32>, tensor<*xf32>) -> (tensor<*xf32>)
    %Placeholder_3, %ctl_4 = Placeholder device("/device:CPU:0") name("scale") {dtype = f32, shape = #tf_type.shape<*>} : () -> (tensor<*xf32>)
    //CHECK: %[[SCALE:.*]], {{.*}} name("scale")
    %Placeholder_5, %ctl_6 = Placeholder device("/device:CPU:0") name("offset") {dtype = f32, shape = #tf_type.shape<*>} : () -> (tensor<*xf32>)
    //CHECK: %[[OFFSET:.*]], {{.*}} name("offset")
    %Placeholder_7, %ctl_8 = Placeholder device("/device:CPU:0") name("mean") {dtype = f32, shape = #tf_type.shape<*>} : () -> (tensor<*xf32>)
    //CHECK: %[[MEAN:.*]], {{.*}} name("mean")
    %Placeholder_9, %ctl_10 = Placeholder device("/device:CPU:0") name("var") {dtype = f32, shape = #tf_type.shape<*>} : () -> (tensor<*xf32>)
    //CHECK: %[[VAR:.*]], {{.*}} name("var")
    %FusedBatchNormV3:6, %ctl_11 = FusedBatchNormV3(%Conv2D, %Placeholder_3, %Placeholder_5, %Placeholder_7, %Placeholder_9) device("/device:CPU:0") name("fused_batch_norm") {T = f32, U = f32, data_format = "NHWC", epsilon = 1.000000e-01 : f32, exponential_avg_factor = 1.000000e+00 : f32, is_training = false} : (tensor<*xf32>, tensor<*xf32>, tensor<*xf32>, tensor<*xf32>, tensor<*xf32>) -> (tensor<*xf32>, tensor<*xf32>, tensor<*xf32>, tensor<*xf32>, tensor<*xf32>, tensor<*xf32>)
    %Relu, %ctl_12 = Relu6(%FusedBatchNormV3#0) device("/device:CPU:0") name("relu6") {T = f32} : (tensor<*xf32>) -> (tensor<*xf32>)
    // CHECK: _FusedConv2D(%[[INPUT]], %[[FILTER]], %[[SCALE]], %[[OFFSET]], %[[MEAN]], %[[VAR]]) {{.*}} name("relu6") {{.*}} fused_ops = ["FusedBatchNorm", "Relu6"]
  }
}

module {
  tfg.graph #tf_type.version<producer = 1133, min_consumer = 0> {
    %Placeholder, %ctl = Placeholder device("/device:CPU:0") name("input") {dtype = f32, shape = #tf_type.shape<2x8x8x24>} : () -> (tensor<*xf32>)
    //CHECK: %[[INPUT:.*]], {{.*}} name("input")
    %Placeholder_0, %ctl_1 = Placeholder device("/device:CPU:0") name("filter") {dtype = f32, shape = #tf_type.shape<3x3x24x16>} : () -> (tensor<*xf32>)
    //CHECK: %[[FILTER:.*]], {{.*}} name("filter")
    %Conv2D, %ctl_2 = Conv2D(%Placeholder, %Placeholder_0) device("/device:CPU:0") name("conv") {T = f32, data_format = "NHWC", dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "SAME", strides = [1, 1, 1, 1], use_cudnn_on_gpu = true} : (tensor<*xf32>, tensor<*xf32>) -> (tensor<*xf32>)
    %Placeholder_3, %ctl_4 = Placeholder device("/device:CPU:0") name("scale") {dtype = f32, shape = #tf_type.shape<*>} : () -> (tensor<*xf32>)
    //CHECK: %[[SCALE:.*]], {{.*}} name("scale")
    %Placeholder_5, %ctl_6 = Placeholder device("/device:CPU:0") name("offset") {dtype = f32, shape = #tf_type.shape<*>} : () -> (tensor<*xf32>)
    //CHECK: %[[OFFSET:.*]], {{.*}} name("offset")
    %Placeholder_7, %ctl_8 = Placeholder device("/device:CPU:0") name("mean") {dtype = f32, shape = #tf_type.shape<*>} : () -> (tensor<*xf32>)
    //CHECK: %[[MEAN:.*]], {{.*}} name("mean")
    %Placeholder_9, %ctl_10 = Placeholder device("/device:CPU:0") name("var") {dtype = f32, shape = #tf_type.shape<*>} : () -> (tensor<*xf32>)
    //CHECK: %[[VAR:.*]], {{.*}} name("var")
    %FusedBatchNormV3:6, %ctl_11 = FusedBatchNormV3(%Conv2D, %Placeholder_3, %Placeholder_5, %Placeholder_7, %Placeholder_9) device("/device:CPU:0") name("fused_batch_norm") {T = f32, U = f32, data_format = "NHWC", epsilon = 1.000000e-01 : f32, exponential_avg_factor = 1.000000e+00 : f32, is_training = false} : (tensor<*xf32>, tensor<*xf32>, tensor<*xf32>, tensor<*xf32>, tensor<*xf32>) -> (tensor<*xf32>, tensor<*xf32>, tensor<*xf32>, tensor<*xf32>, tensor<*xf32>, tensor<*xf32>)
    %LeakyRelu, %ctl_12 = LeakyRelu(%FusedBatchNormV3#0) device("/device:CPU:0") name("leakyrelu") {T = f32, alpha = 0.1 : f32} : (tensor<*xf32>) -> (tensor<*xf32>)
    // CHECK: _FusedConv2D(%[[INPUT]], %[[FILTER]], %[[SCALE]], %[[OFFSET]], %[[MEAN]], %[[VAR]]) {{.*}} name("leakyrelu") {{.*}} fused_ops = ["FusedBatchNorm", "LeakyRelu"]
  }
}
