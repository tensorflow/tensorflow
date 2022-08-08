// RUN: tf-opt -quant-raise-flex-fallback %s | FileCheck %s

// CHECK-LABEL: bias_add
func.func @bias_add(%arg0: tensor<1x10x10x32xf32>, %arg1: tensor<32xf32>) -> tensor<1x10x10x32xf32> {
  %0 = "tf.BiasAdd"(%arg0, %arg1) {T = "tfdtype$DT_FLOAT", data_format = "NHWC"} : (tensor<1x10x10x32xf32>, tensor<32xf32>) -> tensor<1x10x10x32xf32>
  func.return %0 : tensor<1x10x10x32xf32>
// CHECK: %[[BIASADD_0:.*]] = "tf.BiasAdd"(%arg0, %arg1) {T = "tfdtype$DT_FLOAT", data_format = "NHWC"} : (tensor<1x10x10x32xf32>, tensor<32xf32>) -> tensor<1x10x10x32xf32>
// CHECK: return %[[BIASADD_0]] : tensor<1x10x10x32xf32>
}

// CHECK-LABEL: add
func.func @add(%arg0: tensor<1xf32>, %arg1: tensor<1xf32>) -> tensor<1xf32> {
  %0 = "tf.Add"(%arg0, %arg1) : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
  func.return %0: tensor<1xf32>
// CHECK: %[[CUSTOM_0:.*]] = "tfl.custom"(%arg0, %arg1) {custom_code = "FlexAdd", custom_option = #tfl<const_bytes : "0x03416464001412034164641A001A002A070A015412023001320000021B171414042801">} : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
// CHECK: return %[[CUSTOM_0]] : tensor<1xf32>
}

// CHECK-LABEL: softmax
func.func @softmax(%arg0: tensor<8x16xf32>) -> tensor<8x16xf32> {
  %0 = "tf.Softmax"(%arg0) : (tensor<8x16xf32>) -> tensor<8x16xf32>
  func.return %0 : tensor<8x16xf32>
// CHECK: %[[CUSTOM_0:.*]] = "tfl.custom"(%arg0) {custom_code = "FlexSoftmax", custom_option = #tfl<const_bytes : "0x07536F66746D617800161207536F66746D61781A002A070A0154120230013200000221191414042801">} : (tensor<8x16xf32>) -> tensor<8x16xf32>
// CHECK: return %[[CUSTOM_0]] : tensor<8x16xf32>
}

// CHECK-LABEL: conv2d_backprop_input_with_add
func.func @conv2d_backprop_input_with_add(%arg0: tensor<4xi32>, %arg1: tensor<3x3x1x32xf32>, %arg2: tensor<15x14x14x32xf32>) -> tensor<15x28x28x1xf32> {
  %0 = "tf.Conv2DBackpropInput"(%arg0, %arg1, %arg2) {strides = [1, 2, 2, 1], padding="SAME", dilations=[1, 1, 1, 1]}: (tensor<4xi32>, tensor<3x3x1x32xf32>, tensor<15x14x14x32xf32>) -> tensor<15x28x28x1xf32>
  %1 = "tf.Const"() {value = dense<0.000000e+00> : tensor<1xf32>} : () -> tensor<1xf32>
  %2 = "tf.AddV2"(%0, %1): (tensor<15x28x28x1xf32>, tensor<1xf32>) -> tensor<15x28x28x1xf32>
  func.return %2 : tensor<15x28x28x1xf32>
// CHECK: %[[CONST_0:.*]] = "tf.Const"() {value = dense<0.000000e+00> : tensor<1xf32>} : () -> tensor<1xf32>
// CHECK: %[[CONV2DBACKPROPINPUT_0:.*]] = "tf.Conv2DBackpropInput"(%arg0, %arg1, %arg2) {dilations = [1, 1, 1, 1], padding = "SAME", strides = [1, 2, 2, 1]} : (tensor<4xi32>, tensor<3x3x1x32xf32>, tensor<15x14x14x32xf32>) -> tensor<15x28x28x1xf32>
// CHECK: %[[ADDV2_0:.*]] = "tf.AddV2"(%[[CONV2DBACKPROPINPUT_0]], %[[CONST_0]]) : (tensor<15x28x28x1xf32>, tensor<1xf32>) -> tensor<15x28x28x1xf32>
// CHECK: return %[[ADDV2_0]] : tensor<15x28x28x1xf32>
}

// CHECK-LABEL: conv2d_backprop_input_with_sub
func.func @conv2d_backprop_input_with_sub(%arg0: tensor<4xi32>, %arg1: tensor<3x3x1x32xf32>, %arg2: tensor<15x14x14x32xf32>) -> tensor<15x28x28x1xf32> {
  %0 = "tf.Conv2DBackpropInput"(%arg0, %arg1, %arg2) {strides = [1, 2, 2, 1], padding="SAME", dilations=[1, 1, 1, 1]}: (tensor<4xi32>, tensor<3x3x1x32xf32>, tensor<15x14x14x32xf32>) -> tensor<15x28x28x1xf32>
  %1 = "tf.Const"() {value = dense<0.000000e+00> : tensor<1xf32>} : () -> tensor<1xf32>
  %2 = "tf.Sub"(%0, %1): (tensor<15x28x28x1xf32>, tensor<1xf32>) -> tensor<15x28x28x1xf32>
  func.return %2 : tensor<15x28x28x1xf32>
// CHECK: %[[CONST_0:.*]] = "tf.Const"() {value = dense<0.000000e+00> : tensor<1xf32>} : () -> tensor<1xf32>
// CHECK: %[[CONV2DBACKPROPINPUT_0:.*]] = "tf.Conv2DBackpropInput"(%arg0, %arg1, %arg2) {dilations = [1, 1, 1, 1], padding = "SAME", strides = [1, 2, 2, 1]} : (tensor<4xi32>, tensor<3x3x1x32xf32>, tensor<15x14x14x32xf32>) -> tensor<15x28x28x1xf32>
// CHECK: %[[SUB_0:.*]] = "tf.Sub"(%[[CONV2DBACKPROPINPUT_0]], %[[CONST_0]]) : (tensor<15x28x28x1xf32>, tensor<1xf32>) -> tensor<15x28x28x1xf32>
// CHECK: return %[[SUB_0]] : tensor<15x28x28x1xf32>
}

// CHECK-LABEL: depth_to_space
func.func @depth_to_space(%arg0: tensor<1x1x1x4xf32>) -> tensor<1x2x2x1xf32> {
  %0 = "tf.DepthToSpace"(%arg0) {block_size = 2: i64,  data_format = "NHWC"}: (tensor<1x1x1x4xf32>) -> tensor<1x2x2x1xf32>
  func.return %0 : tensor<1x2x2x1xf32>
// CHECK: %[[CUSTOM_0:.*]] = "tfl.custom"(%arg0) {custom_code = "FlexDepthToSpace", custom_option = #tfl<const_bytes : "{{.*}}">} : (tensor<1x1x1x4xf32>) -> tensor<1x2x2x1xf32>
// CHECK: return %[[CUSTOM_0]] : tensor<1x2x2x1xf32>
}

// CHECK-LABEL: floor_mod
func.func @floor_mod(%arg0: tensor<5xf32>, %arg1: tensor<5xf32>) -> tensor<5xf32> {
  %0 = "tf.FloorMod"(%arg0, %arg1) : (tensor<5xf32>, tensor<5xf32>) -> tensor<5xf32>
  func.return %0 : tensor<5xf32>
// CHECK: %[[CUSTOM_0:.*]] = "tfl.custom"(%arg0, %arg1) {custom_code = "FlexFloorMod", custom_option = #tfl<const_bytes : "{{.*}}">} : (tensor<5xf32>, tensor<5xf32>) -> tensor<5xf32>
// CHECK: return %[[CUSTOM_0]] : tensor<5xf32>
}

// CHECK-LABEL: identity_with_const
func.func @identity_with_const() -> tensor<*xf32> {
  %cst = "tf.Const"() {device = "", value = dense<[2.167590e-01, 2.89403105]> : tensor<2xf32>} : () -> tensor<2xf32>
  %cst_1 = "tf.Const"() {device = "", value = dense<1.000000e-03> : tensor<f32>} : () -> tensor<f32>
  %0 = "tf.Identity"(%cst) {device = ""} : (tensor<2xf32>) -> tensor<*xf32>
  %1 = "tf.AddV2"(%0, %cst_1) {device = ""} : (tensor<*xf32>, tensor<f32>) -> tensor<*xf32>
  func.return %1 : tensor<*xf32>
// CHECK: %[[CONST_0:.*]] = "tf.Const"() {value = dense<[2.177590e-01, 2.89503098]> : tensor<2xf32>} : () -> tensor<*xf32>
// CHECK: return %[[CONST_0]] : tensor<*xf32>
}

func.func @identity(%arg0: tensor<2xf32>) -> tensor<*xf32> {
  %cst_1 = "tf.Const"() {device = "", value = dense<1.000000e-03> : tensor<f32>} : () -> tensor<f32>
  %0 = "tf.Identity"(%arg0) {device = ""} : (tensor<2xf32>) -> tensor<*xf32>
  %1 = "tf.AddV2"(%0, %cst_1) {device = ""} : (tensor<*xf32>, tensor<f32>) -> tensor<*xf32>
  func.return %1 : tensor<*xf32>
// CHECK: %[[CONST_0:.*]] = "tf.Const"() {device = "", value = dense<1.000000e-03> : tensor<f32>} : () -> tensor<f32>
// CHECK: %[[IDENTITY_0:.*]] = "tf.Identity"(%arg0) {device = ""} : (tensor<2xf32>) -> tensor<*xf32>
// CHECK: %[[ADDV2_0:.*]] = "tfl.custom"(%0, %cst) {custom_code = "FlexAddV2", custom_option = #tfl<const_bytes : "0x0541646456320016120541646456321A001A002A070A015412023001320000021F191414042801">} : (tensor<*xf32>, tensor<f32>) -> tensor<*xf32>
// CHECK: return %[[ADDV2_0]] : tensor<*xf32>
}

// CHECK-LABEL: bias_add_with_identity
func.func @bias_add_with_identity(%arg0: tensor<4xi32>, %arg1: tensor<3x3x1x32xf32>, %arg2: tensor<15x14x14x32xf32>) -> tensor<15x28x28x1xf32> {
  %cst = "tf.Const"() {device = "", value = dense<[2.167590e-01]> : tensor<1xf32>} : () -> tensor<1xf32>
  %cst_1 = "tf.Const"() {device = "", value = dense<1.000000e-03> : tensor<f32>} : () -> tensor<f32>
  %0 = "tf.Identity"(%cst) {device = ""} : (tensor<1xf32>) -> tensor<1xf32>
  %1 = "tf.AddV2"(%0, %cst_1) {device = ""} : (tensor<1xf32>, tensor<f32>) -> tensor<1xf32>
  %2 = "tf.Conv2DBackpropInput"(%arg0, %arg1, %arg2) {strides = [1, 2, 2, 1], padding="SAME", dilations=[1, 1, 1, 1]}: (tensor<4xi32>, tensor<3x3x1x32xf32>, tensor<15x14x14x32xf32>) -> tensor<15x28x28x1xf32>
  %3 = "tf.AddV2"(%2, %1): (tensor<15x28x28x1xf32>, tensor<1xf32>) -> tensor<15x28x28x1xf32>
  func.return %2 : tensor<15x28x28x1xf32>
// CHECK: %[[CONV2DBACKPROPINPUT_0:.*]] = "tf.Conv2DBackpropInput"(%arg0, %arg1, %arg2) {dilations = [1, 1, 1, 1], padding = "SAME", strides = [1, 2, 2, 1]} : (tensor<4xi32>, tensor<3x3x1x32xf32>, tensor<15x14x14x32xf32>) -> tensor<15x28x28x1xf32>
// CHECK: return %[[CONV2DBACKPROPINPUT_0]] : tensor<15x28x28x1xf32>
}

// CHECK-LABEL: conv_with_relu1_pattern1
func.func @conv_with_relu1_pattern1(%arg0: tensor<1x3x4x3xf32>) -> (tensor<1x3x4x2xf32>) {
  %cst = "tf.Const"() {value = dense<[[[[-8.69931221, 6.44628429], [-9.18393421, 1.53671741], [8.68561744, -3.581774]]]]> : tensor<1x1x3x2xf32>} : () -> tensor<1x1x3x2xf32>
  %cst_0 = "tf.Const"() {value = dense<-1.000000e+00> : tensor<f32>} : () -> tensor<f32>
  %cst_1 = "tf.Const"() {value = dense<1.000000e+00> : tensor<f32>} : () -> tensor<f32>
  %0 = "tf.Conv2D"(%arg0, %cst) {data_format = "NHWC", dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "SAME", strides = [1, 1, 1, 1]} : (tensor<1x3x4x3xf32>, tensor<1x1x3x2xf32>) -> tensor<1x3x4x2xf32>
  %1 = "tf.Maximum"(%0, %cst_0) : (tensor<1x3x4x2xf32>, tensor<f32>) -> tensor<1x3x4x2xf32>
  %2 = "tf.Minimum"(%1, %cst_1) : (tensor<1x3x4x2xf32>, tensor<f32>) -> tensor<1x3x4x2xf32>
  func.return %2 : tensor<1x3x4x2xf32>
// CHECK-DAG: %[[CONST_0:.*]] = "tf.Const"() {value = dense<{{.*}}> : tensor<1x1x3x2xf32>} : () -> tensor<1x1x3x2xf32>
// CHECK-DAG: %[[CONST_1:.*]] = "tf.Const"() {value = dense<-1.000000e+00> : tensor<f32>} : () -> tensor<f32>
// CHECK-DAG: %[[CONST_2:.*]] = "tf.Const"() {value = dense<1.000000e+00> : tensor<f32>} : () -> tensor<f32>
// CHECK: %[[CONV2D_0:.*]] = "tf.Conv2D"(%arg0, %[[CONST_0]]) {data_format = "NHWC", dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "SAME", strides = [1, 1, 1, 1]} : (tensor<1x3x4x3xf32>, tensor<1x1x3x2xf32>) -> tensor<1x3x4x2xf32>
// CHECK: %[[MAXIMUM_0:.*]] = "tf.Maximum"(%[[CONV2D_0]], %[[CONST_1]]) : (tensor<1x3x4x2xf32>, tensor<f32>) -> tensor<1x3x4x2xf32>
// CHECK: %[[MINIMUM_0:.*]] = "tf.Minimum"(%[[MAXIMUM_0]], %[[CONST_2]]) : (tensor<1x3x4x2xf32>, tensor<f32>) -> tensor<1x3x4x2xf32>
// CHECK: return %[[MINIMUM_0]] : tensor<1x3x4x2xf32>
}

// CHECK-LABEL: conv_with_relu1_pattern2
func.func @conv_with_relu1_pattern2(%arg0: tensor<1x3x4x3xf32>) -> (tensor<1x3x4x2xf32>) {
  %cst = "tf.Const"() {value = dense<[[[[-8.69931221, 6.44628429], [-9.18393421, 1.53671741], [8.68561744, -3.581774]]]]> : tensor<1x1x3x2xf32>} : () -> tensor<1x1x3x2xf32>
  %cst_0 = "tf.Const"() {value = dense<-1.000000e+00> : tensor<f32>} : () -> tensor<f32>
  %cst_1 = "tf.Const"() {value = dense<1.000000e+00> : tensor<f32>} : () -> tensor<f32>
  %0 = "tf.Conv2D"(%arg0, %cst) {data_format = "NHWC", dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "SAME", strides = [1, 1, 1, 1]} : (tensor<1x3x4x3xf32>, tensor<1x1x3x2xf32>) -> tensor<1x3x4x2xf32>
  %1 = "tf.Minimum"(%0, %cst_1) : (tensor<1x3x4x2xf32>, tensor<f32>) -> tensor<1x3x4x2xf32>
  %2 = "tf.Maximum"(%1, %cst_0) : (tensor<1x3x4x2xf32>, tensor<f32>) -> tensor<1x3x4x2xf32>
  func.return %2 : tensor<1x3x4x2xf32>
// CHECK-DAG: %[[CONST_0:.*]] = "tf.Const"() {value = dense<{{.*}}> : tensor<1x1x3x2xf32>} : () -> tensor<1x1x3x2xf32>
// CHECK-DAG: %[[CONST_1:.*]] = "tf.Const"() {value = dense<-1.000000e+00> : tensor<f32>} : () -> tensor<f32>
// CHECK-DAG: %[[CONST_2:.*]] = "tf.Const"() {value = dense<1.000000e+00> : tensor<f32>} : () -> tensor<f32>
// CHECK: %[[CONV2D_0:.*]] = "tf.Conv2D"(%arg0, %[[CONST_0]]) {data_format = "NHWC", dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "SAME", strides = [1, 1, 1, 1]} : (tensor<1x3x4x3xf32>, tensor<1x1x3x2xf32>) -> tensor<1x3x4x2xf32>
// CHECK: %[[MINIMUM_0:.*]] = "tf.Minimum"(%[[CONV2D_0]], %[[CONST_2]]) : (tensor<1x3x4x2xf32>, tensor<f32>) -> tensor<1x3x4x2xf32>
// CHECK: %[[MAXIMUM_0:.*]] = "tf.Maximum"(%[[MINIMUM_0]], %[[CONST_1]]) : (tensor<1x3x4x2xf32>, tensor<f32>) -> tensor<1x3x4x2xf32>
// CHECK: return %[[MAXIMUM_0]] : tensor<1x3x4x2xf32>
}

// CHECK-LABEL: conv_with_relu1_invalid_pattern
func.func @conv_with_relu1_invalid_pattern(%arg0: tensor<1x3x4x3xf32>) -> (tensor<1x3x4x2xf32>) {
  %cst = "tf.Const"() {value = dense<[[[[-8.69931221, 6.44628429], [-9.18393421, 1.53671741], [8.68561744, -3.581774]]]]> : tensor<1x1x3x2xf32>} : () -> tensor<1x1x3x2xf32>
  %cst_0 = "tf.Const"() {value = dense<[-1.000000e+00, -3.000000e+00]> : tensor<2xf32>} : () -> tensor<2xf32>
  %cst_1 = "tf.Const"() {value = dense<[1.000000e+00, 3.000000e+00]> : tensor<2xf32>} : () -> tensor<2xf32>
  %0 = "tf.Conv2D"(%arg0, %cst) {data_format = "NHWC", dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "SAME", strides = [1, 1, 1, 1]} : (tensor<1x3x4x3xf32>, tensor<1x1x3x2xf32>) -> tensor<1x3x4x2xf32>
  %1 = "tf.Minimum"(%0, %cst_1) : (tensor<1x3x4x2xf32>, tensor<2xf32>) -> tensor<1x3x4x2xf32>
  %2 = "tf.Maximum"(%1, %cst_0) : (tensor<1x3x4x2xf32>, tensor<2xf32>) -> tensor<1x3x4x2xf32>
  func.return %2 : tensor<1x3x4x2xf32>
// CHECK-DAG: %[[CONST_0:.*]] = "tf.Const"() {value = dense<{{.*}}> : tensor<1x1x3x2xf32>} : () -> tensor<1x1x3x2xf32>
// CHECK-DAG: %[[CONST_1:.*]] = "tf.Const"() {value = dense<[-1.000000e+00, -3.000000e+00]> : tensor<2xf32>} : () -> tensor<2xf32>
// CHECK-DAG: %[[CONST_2:.*]] = "tf.Const"() {value = dense<[1.000000e+00, 3.000000e+00]> : tensor<2xf32>} : () -> tensor<2xf32>
// CHECK: %[[CONV2D_0:.*]] = "tf.Conv2D"(%arg0, %[[CONST_0]]) {data_format = "NHWC", dilations = [1, 1, 1, 1], explicit_paddings = [], padding = "SAME", strides = [1, 1, 1, 1]} : (tensor<1x3x4x3xf32>, tensor<1x1x3x2xf32>) -> tensor<1x3x4x2xf32>
// CHECK: %[[CUSTOM_0:.*]] = "tfl.custom"(%[[CONV2D_0]], %[[CONST_2]]) {custom_code = "FlexMinimum", custom_option = #tfl<const_bytes : "0x074D696E696D756D001812074D696E696D756D1A001A002A070A01541202300132000002231B1414042801">} : (tensor<1x3x4x2xf32>, tensor<2xf32>) -> tensor<1x3x4x2xf32>
// CHECK: %[[CUSTOM_1:.*]] = "tfl.custom"(%[[CUSTOM_0]], %[[CONST_1]]) {custom_code = "FlexMaximum", custom_option = #tfl<const_bytes : "0x074D6178696D756D001812074D6178696D756D1A001A002A070A01541202300132000002231B1414042801">} : (tensor<1x3x4x2xf32>, tensor<2xf32>) -> tensor<1x3x4x2xf32>
// CHECK: return %[[CUSTOM_1]] : tensor<1x3x4x2xf32>
}
