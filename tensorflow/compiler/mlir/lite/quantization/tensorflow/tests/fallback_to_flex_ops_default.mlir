// RUN: tf-opt -quant-raise-flex-fallback %s | FileCheck %s

// CHECK-LABEL: bias_add
func @bias_add(%arg0: tensor<1x10x10x32xf32>, %arg1: tensor<32xf32>) -> tensor<1x10x10x32xf32> {
  %0 = "tf.BiasAdd"(%arg0, %arg1) {T = "tfdtype$DT_FLOAT", data_format = "NHWC"} : (tensor<1x10x10x32xf32>, tensor<32xf32>) -> tensor<1x10x10x32xf32>
  return %0 : tensor<1x10x10x32xf32>
// CHECK: %[[BIASADD_0:.*]] = "tf.BiasAdd"(%arg0, %arg1) {T = "tfdtype$DT_FLOAT", data_format = "NHWC"} : (tensor<1x10x10x32xf32>, tensor<32xf32>) -> tensor<1x10x10x32xf32>
// CHECK: return %[[BIASADD_0]] : tensor<1x10x10x32xf32>
}

// CHECK-LABEL: add
func @add(%arg0: tensor<1xf32>, %arg1: tensor<1xf32>) -> tensor<1xf32> {
  %0 = "tf.Add"(%arg0, %arg1) : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
  return %0: tensor<1xf32>
// CHECK: %[[CUSTOM_0:.*]] = "tfl.custom"(%arg0, %arg1) {custom_code = "FlexAdd", custom_option = opaque<"tfl", "0x03416464001412034164641A001A002A070A015412023001320000021B171414042801"> : tensor<35xi8>} : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
// CHECK: return %[[CUSTOM_0]] : tensor<1xf32>
}

// CHECK-LABEL: softmax
func @softmax(%arg0: tensor<8x16xf32>) -> tensor<8x16xf32> {
  %0 = "tf.Softmax"(%arg0) : (tensor<8x16xf32>) -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
// CHECK: %[[CUSTOM_0:.*]] = "tfl.custom"(%arg0) {custom_code = "FlexSoftmax", custom_option = opaque<"tfl", "0x07536F66746D617800161207536F66746D61781A002A070A0154120230013200000221191414042801"> : tensor<41xi8>} : (tensor<8x16xf32>) -> tensor<8x16xf32>
// CHECK: return %[[CUSTOM_0]] : tensor<8x16xf32>
}

// CHECK-LABEL: conv2d_backprop_input_with_add
func @conv2d_backprop_input_with_add(%arg0: tensor<4xi32>, %arg1: tensor<3x3x1x32xf32>, %arg2: tensor<15x14x14x32xf32>) -> tensor<15x28x28x1xf32> {
  %0 = "tf.Conv2DBackpropInput"(%arg0, %arg1, %arg2) {strides = [1, 2, 2, 1], padding="SAME", dilations=[1, 1, 1, 1]}: (tensor<4xi32>, tensor<3x3x1x32xf32>, tensor<15x14x14x32xf32>) -> tensor<15x28x28x1xf32>
  %1 = "tf.Const"() {value = dense<0.000000e+00> : tensor<1xf32>} : () -> tensor<1xf32>
  %2 = "tf.AddV2"(%0, %1): (tensor<15x28x28x1xf32>, tensor<1xf32>) -> tensor<15x28x28x1xf32>
  return %2 : tensor<15x28x28x1xf32>
// CHECK: %[[CONST_0:.*]] = "tf.Const"() {value = dense<0.000000e+00> : tensor<1xf32>} : () -> tensor<1xf32>
// CHECK: %[[CONV2DBACKPROPINPUT_0:.*]] = "tf.Conv2DBackpropInput"(%arg0, %arg1, %arg2) {dilations = [1, 1, 1, 1], padding = "SAME", strides = [1, 2, 2, 1]} : (tensor<4xi32>, tensor<3x3x1x32xf32>, tensor<15x14x14x32xf32>) -> tensor<15x28x28x1xf32>
// CHECK: %[[ADD_0:.*]] = "tf.BiasAdd"(%[[CONV2DBACKPROPINPUT_0]], %[[CONST_0]]) {data_format = "NHWC"} : (tensor<15x28x28x1xf32>, tensor<1xf32>) -> tensor<15x28x28x1xf32>
// CHECK: return %[[ADD_0]] : tensor<15x28x28x1xf32>
}

// CHECK-LABEL: conv2d_backprop_input_with_sub
func @conv2d_backprop_input_with_sub(%arg0: tensor<4xi32>, %arg1: tensor<3x3x1x32xf32>, %arg2: tensor<15x14x14x32xf32>) -> tensor<15x28x28x1xf32> {
  %0 = "tf.Conv2DBackpropInput"(%arg0, %arg1, %arg2) {strides = [1, 2, 2, 1], padding="SAME", dilations=[1, 1, 1, 1]}: (tensor<4xi32>, tensor<3x3x1x32xf32>, tensor<15x14x14x32xf32>) -> tensor<15x28x28x1xf32>
  %1 = "tf.Const"() {value = dense<0.000000e+00> : tensor<1xf32>} : () -> tensor<1xf32>
  %2 = "tf.Sub"(%0, %1): (tensor<15x28x28x1xf32>, tensor<1xf32>) -> tensor<15x28x28x1xf32>
  return %2 : tensor<15x28x28x1xf32>
// CHECK: %[[CONST_0:.*]] = "tf.Const"() {value = dense<-0.000000e+00> : tensor<1xf32>} : () -> tensor<1xf32>
// CHECK: %[[CONV2DBACKPROPINPUT_0:.*]] = "tf.Conv2DBackpropInput"(%arg0, %arg1, %arg2) {dilations = [1, 1, 1, 1], padding = "SAME", strides = [1, 2, 2, 1]} : (tensor<4xi32>, tensor<3x3x1x32xf32>, tensor<15x14x14x32xf32>) -> tensor<15x28x28x1xf32>
// CHECK: %[[BIASADD_0:.*]] = "tf.BiasAdd"(%[[CONV2DBACKPROPINPUT_0]], %[[CONST_0]]) {data_format = "NHWC"} : (tensor<15x28x28x1xf32>, tensor<1xf32>) -> tensor<15x28x28x1xf32>
// CHECK: return %[[BIASADD_0]] : tensor<15x28x28x1xf32>
}

// CHECK-LABEL: depth_to_space
func @depth_to_space(%arg0: tensor<1x1x1x4xf32>) -> tensor<1x2x2x1xf32> {
  %0 = "tf.DepthToSpace"(%arg0) {block_size = 2: i64,  data_format = "NHWC"}: (tensor<1x1x1x4xf32>) -> tensor<1x2x2x1xf32>
  return %0 : tensor<1x2x2x1xf32>
// CHECK: %[[CUSTOM_0:.*]] = "tfl.custom"(%arg0) {custom_code = "FlexDepthToSpace", custom_option = opaque<"tfl", "{{.*}}"> : tensor<92xi8>} : (tensor<1x1x1x4xf32>) -> tensor<1x2x2x1xf32>
// CHECK: return %[[CUSTOM_0]] : tensor<1x2x2x1xf32>
}

// CHECK-LABEL: floor_mod
func @floor_mod(%arg0: tensor<5xf32>, %arg1: tensor<5xf32>) -> tensor<5xf32> {
  %0 = "tf.FloorMod"(%arg0, %arg1) : (tensor<5xf32>, tensor<5xf32>) -> tensor<5xf32>
  return %0 : tensor<5xf32>
// CHECK: %[[CUSTOM_0:.*]] = "tfl.custom"(%arg0, %arg1) {custom_code = "FlexFloorMod", custom_option = opaque<"tfl", "{{.*}}"> : tensor<45xi8>} : (tensor<5xf32>, tensor<5xf32>) -> tensor<5xf32>
// CHECK: return %[[CUSTOM_0]] : tensor<5xf32>
}
