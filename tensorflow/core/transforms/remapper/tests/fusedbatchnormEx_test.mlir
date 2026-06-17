// RUN: tfg-transforms-opt --pass-pipeline='builtin.module(tfg-shape-inference,tfg-remapper)' %s | FileCheck %s

// CHECK-LABEL: tfg.func @fusedbatchnorm_sideinput_relu_gpu
tfg.func @fusedbatchnorm_sideinput_relu_gpu() {
    //CHECK: %[[INPUT:.*]], {{.*}} name("input")
    %Placeholder_0, %ctl = Placeholder device("/device:GPU:0") name("input") {dtype = f32, shape = #tf_type.shape<2x8x8x24>} : () -> (tensor<*xf32>)
    //CHECK: %[[SCALE:.*]], {{.*}} name("scale")    
    %Placeholder_1, %ctl_1 = Placeholder device("/device:GPU:0") name("scale") {dtype = f32, shape = #tf_type.shape<*>} : () -> (tensor<*xf32>)
    //CHECK: %[[OFFSET:.*]], {{.*}} name("offset")
    %Placeholder_2, %ctl_2 = Placeholder device("/device:GPU:0") name("offset") {dtype = f32, shape = #tf_type.shape<*>} : () -> (tensor<*xf32>)
    //CHECK: %[[MEAN:.*]], {{.*}} name("mean")
    %Placeholder_3, %ctl_3 = Placeholder device("/device:GPU:0") name("mean") {dtype = f32, shape = #tf_type.shape<*>} : () -> (tensor<*xf32>)
    //CHECK: %[[VAR:.*]], {{.*}} name("var")
    %Placeholder_4, %ctl_4 = Placeholder device("/device:GPU:0") name("var") {dtype = f32, shape = #tf_type.shape<*>} : () -> (tensor<*xf32>)
    //CHECK: %[[SIDE_IN:.*]], {{.*}} name("input_side")    
    %Placeholder_5, %ctl_6 = Placeholder device("/device:GPU:0") name("input_side") {dtype = f32, shape = #tf_type.shape<2x8x8x24>} : () -> (tensor<*xf32>)
    %FusedBatchNormV3:6, %ctl_5 = FusedBatchNormV3(%Placeholder_0, %Placeholder_1, %Placeholder_2, %Placeholder_3, %Placeholder_4) device("/device:GPU:0") name("fused_batch_norm") {T = f32, U = f32, data_format = "NHWC", epsilon = 1.000000e-01 : f32, exponential_avg_factor = 1.000000e+00 : f32, is_training = false} : (tensor<*xf32>, tensor<*xf32>, tensor<*xf32>, tensor<*xf32>, tensor<*xf32>) -> (tensor<*xf32>, tensor<*xf32>, tensor<*xf32>, tensor<*xf32>, tensor<*xf32>, tensor<*xf32>)
    %AddV2, %ctl_7 = AddV2(%FusedBatchNormV3#0, %Placeholder_5) : (tensor<*xf32>, tensor<*xf32>) -> (tensor<*xf32>)
    // CHECK: %[[FBN_Ex:.*]], {{.*}}_FusedBatchNormEx(%[[INPUT]], %[[SCALE]], %[[OFFSET]], %[[MEAN]], %[[VAR]], %[[SIDE_IN]]) {{.*}} name("fused_batch_norm") {{.*}} activation_mode = "Relu", {{.*}} num_side_inputs = 1
    %Relu, %ctl_8 = Relu(%AddV2) device("/device:GPU:0") name("relu") {T = f32} : (tensor<*xf32>) -> (tensor<*xf32>)
    // CHECK: Identity(%_FusedBatchNormEx#0) {{.*}} name("relu")

    return
}
