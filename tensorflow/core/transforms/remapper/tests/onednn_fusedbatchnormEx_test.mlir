
// RUN: tfg-transforms-opt --pass-pipeline='builtin.module(tfg-shape-inference,tfg-remapper{enable-onednn-patterns=true})' %s | FileCheck %s

// CHECK-LABEL: tfg.func @fusedbatchnorm_relu
tfg.func @fusedbatchnorm_relu() {
    %Placeholder, %ctl = Placeholder device("/device:CPU:0") name("input") {dtype = f32, shape = #tf_type.shape<2x8x8x24>} : () -> (tensor<*xf32>)
    //CHECK: %[[INPUT:.*]], {{.*}} name("input")
    %Placeholder_3, %ctl_4 = Placeholder device("/device:CPU:0") name("scale") {dtype = f32, shape = #tf_type.shape<*>} : () -> (tensor<*xf32>)
    //CHECK: %[[SCALE:.*]], {{.*}} name("scale")
    %Placeholder_5, %ctl_6 = Placeholder device("/device:CPU:0") name("offset") {dtype = f32, shape = #tf_type.shape<*>} : () -> (tensor<*xf32>)
    //CHECK: %[[OFFSET:.*]], {{.*}} name("offset")
    %Placeholder_7, %ctl_8 = Placeholder device("/device:CPU:0") name("mean") {dtype = f32, shape = #tf_type.shape<*>} : () -> (tensor<*xf32>)
    //CHECK: %[[MEAN:.*]], {{.*}} name("mean")
    %Placeholder_9, %ctl_10 = Placeholder device("/device:CPU:0") name("var") {dtype = f32, shape = #tf_type.shape<*>} : () -> (tensor<*xf32>)
    //CHECK: %[[VAR:.*]], {{.*}} name("var")
    %FusedBatchNormV3:6, %ctl_11 = FusedBatchNormV3(%Placeholder, %Placeholder_3, %Placeholder_5, %Placeholder_7, %Placeholder_9) device("/device:CPU:0") name("fused_batch_norm") {T = f32, U = f32, data_format = "NHWC", epsilon = 1.000000e-01 : f32, exponential_avg_factor = 1.000000e+00 : f32, is_training = false} : (tensor<*xf32>, tensor<*xf32>, tensor<*xf32>, tensor<*xf32>, tensor<*xf32>) -> (tensor<*xf32>, tensor<*xf32>, tensor<*xf32>, tensor<*xf32>, tensor<*xf32>, tensor<*xf32>)
    // CHECK: _FusedBatchNormEx(%[[INPUT]], %[[SCALE]], %[[OFFSET]], %[[MEAN]], %[[VAR]]) {{.*}} name("fused_batch_norm") {{.*}} activation_mode = "Relu", {{.*}} num_side_inputs = 0
    %Relu, %ctl_12 = Relu(%FusedBatchNormV3#0) device("/device:CPU:0") name("relu") {T = f32} : (tensor<*xf32>) -> (tensor<*xf32>)
    // CHECK: Identity(%_FusedBatchNormEx#0) {{.*}} name("relu")
    return
}
