// RUN: tf-opt %s -tfl-legalize-tf | FileCheck %s --dump-input-on-failure

func @addRelu(%arg0: tensor<1xi32>, %arg1: tensor<1xi32>) -> tensor<1xi32> {
  %0 = "tf.Add"(%arg0, %arg1) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
  %1 = "tf.Add"(%arg0, %0) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
  %2 = "tf.Relu"(%1) : (tensor<1xi32>) -> tensor<1xi32>
  %3 = "tf.Relu"(%arg0) : (tensor<1xi32>) -> tensor<1xi32>
  %4 = "tf.Add"(%3, %2) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
  %5 = "tf.Relu6"(%4) : (tensor<1xi32>) -> tensor<1xi32>
  %6 = "tfl.add"(%5, %3) {fused_activation_function = "NONE"} : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
  %7 = "tf.Relu6"(%6) : (tensor<1xi32>) -> tensor<1xi32>
  return %7: tensor<1xi32>

// CHECK-LABEL: addRelu
// CHECK:  %0 = tfl.add %arg0, %arg1 {fused_activation_function = "NONE"} : tensor<1xi32>
// CHECK:  %1 = tfl.add %arg0, %0 {fused_activation_function = "RELU"} : tensor<1xi32>
// CHECK:  %2 = "tfl.relu"(%arg0) : (tensor<1xi32>) -> tensor<1xi32>
// CHECK:  %3 = tfl.add %2, %1 {fused_activation_function = "RELU6"} : tensor<1xi32>
// CHECK:  %4 = tfl.add %3, %2 {fused_activation_function = "RELU6"} : tensor<1xi32>
// CHECK:  return %4 : tensor<1xi32>
}

func @LeakyRelu(%arg0: tensor<1xf32>) -> tensor<1xf32> {
  %2 = "tf.LeakyRelu"(%arg0) {alpha = 0.1 : f32} : (tensor<1xf32>) -> tensor<1xf32>
  return %2: tensor<1xf32>

// CHECK-LABEL: LeakyRelu
// CHECK:  %0 = "tfl.leaky_relu"(%arg0) {alpha = 1.000000e-01 : f32} : (tensor<1xf32>) -> tensor<1xf32>
}

func @biasAdd(%arg0: tensor<1x10x10x32xf32>, %arg1: tensor<32xf32>) -> tensor<1x10x10x32xf32> {
  %0 = "tf.BiasAdd"(%arg0, %arg1) {T = "tfdtype$DT_FLOAT", data_format = "NHWC"} : (tensor<1x10x10x32xf32>, tensor<32xf32>) -> tensor<1x10x10x32xf32>
  %1 = "tf.BiasAdd"(%0, %arg0) {T = "tfdtype$DT_FLOAT", data_format = "NHWC"} : (tensor<1x10x10x32xf32>, tensor<1x10x10x32xf32>) -> tensor<1x10x10x32xf32>
  %2 = "tf.Relu6"(%1) : (tensor<1x10x10x32xf32>) -> tensor<1x10x10x32xf32>
  return %2 : tensor<1x10x10x32xf32>

// CHECK-LABEL: biasAdd
// CHECK:  %0 = "tfl.add"(%arg0, %arg1) {fused_activation_function = "NONE"} : (tensor<1x10x10x32xf32>, tensor<32xf32>) -> tensor<1x10x10x32xf32>
// CHECK:  %1 = tfl.add %0, %arg0 {fused_activation_function = "RELU6"} : tensor<1x10x10x32xf32>
}

func @biasAddInt(%arg0: tensor<1x10x10x32xi32>, %arg1: tensor<32xi32>) -> tensor<1x10x10x32xi32> {
  %0 = "tf.BiasAdd"(%arg0, %arg1) {T = "tfdtype$DT_FLOAT", data_format = "NHWC"} : (tensor<1x10x10x32xi32>, tensor<32xi32>) -> tensor<1x10x10x32xi32>
  return %0 : tensor<1x10x10x32xi32>

// CHECK-LABEL: biasAddInt
// CHECK:  %0 = "tf.BiasAdd"(%arg0, %arg1)
}

func @squeezeAndReshape(%arg0: tensor<1x1x10xf32>, %arg1: tensor<?x10xf32>) -> i32 {
  %0 = "tf.Squeeze"(%arg0) {squeeze_dims = [0]} : (tensor<1x1x10xf32>) -> tensor<1x10xf32>
  %1 = "tf.Squeeze"(%arg1) : (tensor<?x10xf32>) -> tensor<*xf32>
  %2 = constant dense<[2, 5]> : tensor<2xi32>
  %3 = "tf.Reshape" (%0, %2) : (tensor<1x10xf32>, tensor<2xi32>) -> tensor<2x5xf32>
  %4 = "some_op"(%1, %3) : (tensor<*xf32>, tensor<2x5xf32>) -> i32
  return %4 : i32
// CHECK-LABEL: squeezeAndReshape
// CHECK:  %0 = "tfl.squeeze"(%arg0) {squeeze_dims = [0]} : (tensor<1x1x10xf32>) -> tensor<1x10xf32>
// CHECK:  %1 = "tfl.squeeze"(%arg1) {squeeze_dims = []} : (tensor<?x10xf32>) -> tensor<*xf32>
// CHECK:  %2 = "tfl.reshape"(%0) : (tensor<1x10xf32>) -> tensor<2x5xf32>
// CHECK:  %3 = "some_op"(%1, %2) : (tensor<*xf32>, tensor<2x5xf32>) -> i32
// CHECK:  return %3 : i32
}

func @dynamicReshape(%arg0: tensor<*xf32>, %arg1: tensor<2xi32>) -> tensor<?x?xf32> {
  %0 = "tf.Reshape"(%arg0, %arg1) : (tensor<*xf32>, tensor<2xi32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
// CHECK-LABEL: dynamicReshape
// CHECK:  %0 = "tf.Reshape"(%arg0, %arg1) : (tensor<*xf32>, tensor<2xi32>) -> tensor<?x?xf32>
// CHECK:  return %0 : tensor<?x?xf32>
}

func @avgPool2D(%arg0: tensor<1x6x6x16xf32>) -> tensor<1x1x1x16xf32> {
  // OK
  %0 = "tf.AvgPool"(%arg0) {T = "tfdtype$DT_FLOAT", data_format = "NHWC", ksize = [1, 3, 6, 1], padding = "VALID", strides = [1, 3, 1, 1]} : (tensor<1x6x6x16xf32>) -> tensor<1x1x1x16xf32>
  // Unsupported data format
  %1 = "tf.AvgPool"(%arg0) {T = "tfdtype$DT_FLOAT", data_format = "NCHW", ksize = [1, 3, 6, 1], padding = "VALID", strides = [1, 3, 1, 1]} : (tensor<1x6x6x16xf32>) -> tensor<1x1x1x16xf32>
  // Unsupported ksize
  %2 = "tf.AvgPool"(%arg0) {T = "tfdtype$DT_FLOAT", data_format = "NHWC", ksize = [3, 3, 6, 1], padding = "VALID", strides = [1, 3, 1, 1]} : (tensor<1x6x6x16xf32>) -> tensor<1x1x1x16xf32>
  // Unsupported strides
  %3 = "tf.AvgPool"(%arg0) {T = "tfdtype$DT_FLOAT", data_format = "NHWC", ksize = [1, 3, 6, 1], padding = "VALID", strides = [1, 3, 1, 3]} : (tensor<1x6x6x16xf32>) -> tensor<1x1x1x16xf32>

  %5 = addf %0, %1 : tensor<1x1x1x16xf32>
  %6 = addf %2, %3 : tensor<1x1x1x16xf32>
  %7 = addf %5, %6 : tensor<1x1x1x16xf32>
  return %7 : tensor<1x1x1x16xf32>

// CHECK-LABEL: func @avgPool2D
// CHECK:  %0 = "tfl.average_pool_2d"(%arg0) {filter_height = 3 : i32, filter_width = 6 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 3 : i32, stride_w = 1 : i32} : (tensor<1x6x6x16xf32>) -> tensor<1x1x1x16xf32>
// CHECK:  %1 = "tf.AvgPool"(%arg0)
// CHECK:  %2 = "tf.AvgPool"(%arg0)
// CHECK:  %3 = "tf.AvgPool"(%arg0)
}

func @softmax(%arg0: tensor<8x16xf32>) -> tensor<8x16xf32> {
  %0 = "tf.Softmax"(%arg0) : (tensor<8x16xf32>) -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>

// CHECK-LABEL: softmax
// CHECK:  %0 = "tfl.softmax"(%arg0) {beta = 1.000000e+00 : f32} : (tensor<8x16xf32>) -> tensor<8x16xf32>
}

func @fakeQuantArgsFalse(%arg0: tensor<8x8x8x8xf32>) -> tensor<8x8x8x8xf32> {
  %0 = "tf.FakeQuantWithMinMaxArgs"(%arg0) {min = -0.1 : f32, max = 0.2 : f32, num_bits = 3, narrow_range = false} : (tensor<8x8x8x8xf32>) -> tensor<8x8x8x8xf32>
  return %0 : tensor<8x8x8x8xf32>

  // CHECK-LABEL: fakeQuantArgsFalse
  // CHECK: %0 = "tfl.quantize"(%arg0) {qtype = tensor<8x8x8x8x!quant.uniform<u8:f32, 0.0011764706057660721:85>>}
  // CHECK: %1 = "tfl.dequantize"(%0) : (tensor<8x8x8x8x!quant.uniform<u8:f32, 0.0011764706057660721:85>>) -> tensor<8x8x8x8xf32>
}

func @fakeQuantArgsTrue(%arg0: tensor<8x8x8x8xf32>) -> tensor<8x8x8x8xf32> {
  %0 = "tf.FakeQuantWithMinMaxArgs"(%arg0) {min = -0.1 : f32, max = 0.2 : f32, num_bits = 3, narrow_range = true} : (tensor<8x8x8x8xf32>) -> tensor<8x8x8x8xf32>
  return %0 : tensor<8x8x8x8xf32>

  // CHECK-LABEL: fakeQuantArgsTrue
  // CHECK: %0 = "tfl.quantize"(%arg0) {qtype = tensor<8x8x8x8x!quant.uniform<u8<1:255>:f32, 0.001181102379804521:86>>} : (tensor<8x8x8x8xf32>) -> tensor<8x8x8x8x!quant.uniform<u8<1:255>:f32, 0.001181102379804521:86>>
  // CHECK: %1 = "tfl.dequantize"(%0) : (tensor<8x8x8x8x!quant.uniform<u8<1:255>:f32, 0.001181102379804521:86>>) -> tensor<8x8x8x8xf32>
}

func @fakeQuantVarsFalse(%arg0: tensor<8x8x8x8xf32>) -> tensor<8x8x8x8xf32> {
  %arg1 = constant dense<-0.1> : tensor<f32>
  %arg2 = constant dense<0.2> : tensor<f32>
  %0 = "tf.FakeQuantWithMinMaxVars"(%arg0, %arg1, %arg2) {num_bits = 3, narrow_range = false} : (tensor<8x8x8x8xf32>, tensor<f32>, tensor<f32>) -> tensor<8x8x8x8xf32>
  return %0 : tensor<8x8x8x8xf32>

  // CHECK-LABEL: fakeQuantVarsFalse
  // CHECK: %0 = "tfl.quantize"(%arg0) {qtype = tensor<8x8x8x8x!quant.uniform<u8:f32, 0.0011764706057660721:85>>}
  // CHECK: %1 = "tfl.dequantize"(%0) : (tensor<8x8x8x8x!quant.uniform<u8:f32, 0.0011764706057660721:85>>) -> tensor<8x8x8x8xf32>
}

func @fakeQuantVarsTrue(%arg0: tensor<8x8x8x8xf32>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> tensor<8x8x8x8xf32> {
  %0 = "tf.FakeQuantWithMinMaxVars"(%arg0, %arg1, %arg2) {min = 0.0 : f32, max = 1.0 : f32, num_bits = 3, narrow_range = true} : (tensor<8x8x8x8xf32>, tensor<f32>, tensor<f32>) -> tensor<8x8x8x8xf32>
  return %0 : tensor<8x8x8x8xf32>

  // CHECK-LABEL: fakeQuantVarsTrue
  // CHECK: %0 = "tf.FakeQuantWithMinMaxVars"(%arg0, %arg1, %arg2) {max = 1.000000e+00 : f32, min = 0.000000e+00 : f32, narrow_range = true, num_bits = 3 : i64}
}

func @const() -> tensor<2xi32> {
  %0 = "tf.Const"() {device = "", name = "weights_quant/min", dtype = "tfdtype$DT_INT32", value = opaque<"tf", "0x746674656E736F722464747970653A2044545F494E5433320A74656E736F725F7368617065207B0A202064696D207B0A2020202073697A653A20320A20207D0A7D0A74656E736F725F636F6E74656E743A20225C3230305C3030305C3030305C3030305C3230305C3030305C3030305C303030220A"> : tensor<2xi32>} : () -> (tensor<2xi32>)
  return %0: tensor<2xi32>

// CHECK-LABEL: @const
// CHECK: %0 = "tfl.pseudo_const"() {value = opaque<"tf", "0x746674656E736F722464747970653A2044545F494E5433320A74656E736F725F7368617065207B0A202064696D207B0A2020202073697A653A20320A20207D0A7D0A74656E736F725F636F6E74656E743A20225C3230305C3030305C3030305C3030305C3230305C3030305C3030305C303030220A"> : tensor<2xi32>} : () -> tensor<2xi32>
}

func @placeholder(%arg0: tensor<f32>) -> tensor<f32> {
  %0 = "tf.Placeholder.input"(%arg0) {name = "Input"} : (tensor<f32>) -> tensor<f32>
  return %0: tensor<f32>

// CHECK-LABEL: @placeholder
// CHECK:  %0 = "tfl.pseudo_input"(%arg0) : (tensor<f32>) -> tensor<f32>
}

func @placeholder_min(%arg0: tensor<f32>) -> tensor<f32> {
  %0 = "tf.Placeholder.input"(%arg0) {name = "Input", min = -0.1 : f32} : (tensor<f32>) -> tensor<f32>
  return %0: tensor<f32>

// CHECK-LABEL: @placeholder_min
// CHECK:  %0 = "tfl.pseudo_input"(%arg0) : (tensor<f32>) -> tensor<f32>
}

func @placeholder_type(%arg0: tensor<f32>) -> tensor<f32> {
  %0 = "tf.Placeholder.input"(%arg0) {name = "Input", type = i8} : (tensor<f32>) -> tensor<f32>
  return %0: tensor<f32>

// CHECK-LABEL: @placeholder_type
// CHECK:  %0 = "tfl.pseudo_input"(%arg0) : (tensor<f32>) -> tensor<f32>
}

func @placeholder_min_max(%arg0: tensor<2x3xf32>) -> tensor<2x3xf32> {
  %0 = "tf.Placeholder.input"(%arg0) {name = "Input", min = -0.1 : f32, max = 0.1 : f32, type = i8} : (tensor<2x3xf32>) -> tensor<2x3xf32>
  return %0: tensor<2x3xf32>

// CHECK-LABEL: @placeholder_min_max
// CHECK:  %0 = "tfl.pseudo_input"(%arg0) : (tensor<2x3xf32>) -> tensor<2x3xf32>
// CHECK:  %1 = "tfl.quantize"(%0) {qtype = tensor<2x3x!quant.uniform<u8:f32, 7.8431373717738134E-4:128>>}
// CHECK:  %2 = "tfl.dequantize"(%1) : (tensor<2x3x!quant.uniform<u8:f32, 7.8431373717738134E-4:128>>)
}

func @shape(%arg0: tensor<?x1001xf32>) -> tensor<2xi32> {
  %0 = "tf.Shape"(%arg0) {T = "tfdtype$DT_FLOAT", out_type = "tfdtype$DT_INT32"} : (tensor<?x1001xf32>) -> tensor<2xi32>
  %1 = "tf.Shape"(%arg0) {T = "tfdtype$DT_FLOAT"} : (tensor<?x1001xf32>) -> tensor<2xi32>
  %2 = "tf.Add"(%0, %1) : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
  return %2: tensor<2xi32>

// CHECK-LABEL: shape
// CHECK:  %0 = "tfl.shape"(%arg0) : (tensor<?x1001xf32>) -> tensor<2xi32>
// CHECK:  %1 = "tfl.shape"(%arg0) : (tensor<?x1001xf32>) -> tensor<2xi32>
}

func @fill(%arg0: tensor<3xi32>, %arg1: tensor<f32>) -> tensor<?x?x?xf32> {
  %0 = "tf.Fill"(%arg0, %arg1) : (tensor<3xi32>, tensor<f32>) -> tensor<?x?x?xf32>
  return %0 : tensor<?x?x?xf32>

// CHECK-LABEL:fill
// CHECK:  %0 = "tfl.fill"(%arg0, %arg1) : (tensor<3xi32>, tensor<f32>) -> tensor<?x?x?xf32>
}

func @sigmoid(%arg0: tensor<?x88xf16>) -> tensor<?x88xf16> {
  %0 = "tf.Sigmoid"(%arg0) : (tensor<?x88xf16>) -> tensor<?x88xf16>
  return %0 : tensor<?x88xf16>
// CHECK-LABEL: sigmoid
// CHECK:  %0 = "tfl.logistic"(%arg0) : (tensor<?x88xf16>) -> tensor<?x88xf16>
}

func @log_softmax(%arg0: tensor<8x16xf32>) -> tensor<8x16xf32> {
  %0 = "tf.LogSoftmax"(%arg0) : (tensor<8x16xf32>) -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
// CHECK-LABEL: log_softmax
// CHECK:  %0 = "tfl.log_softmax"(%arg0) : (tensor<8x16xf32>) -> tensor<8x16xf32>
}

func @zeros_like(%arg0: tensor<8x16xf32>) -> tensor<8x16xf32> {
  %0 = "tf.ZerosLike"(%arg0) : (tensor<8x16xf32>) -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
// CHECK-LABEL: zeros_like
// CHECK:  %0 = "tfl.zeros_like"(%arg0) : (tensor<8x16xf32>) -> tensor<8x16xf32>
}

func @divRelu(%arg0: tensor<1xi32>, %arg1: tensor<1xi32>) -> tensor<1xi32> {
  %0 = "tf.Div"(%arg0, %arg1) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
  %1 = "tf.Div"(%arg0, %0) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
  %2 = "tf.Relu"(%1) : (tensor<1xi32>) -> tensor<1xi32>
  %3 = "tf.Relu"(%arg0) : (tensor<1xi32>) -> tensor<1xi32>
  %4 = "tf.Div"(%3, %2) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
  %5 = "tf.Relu6"(%4) : (tensor<1xi32>) -> tensor<1xi32>
  return %5: tensor<1xi32>

// CHECK-LABEL: divRelu
// CHECK:  %0 = tfl.div %arg0, %arg1 {fused_activation_function = "NONE"} : tensor<1xi32>
// CHECK:  %1 = tfl.div %arg0, %0 {fused_activation_function = "RELU"} : tensor<1xi32>
// CHECK:  %2 = "tfl.relu"(%arg0) : (tensor<1xi32>) -> tensor<1xi32>
// CHECK:  %3 = tfl.div %2, %1 {fused_activation_function = "RELU6"} : tensor<1xi32>
// CHECK:  return %3 : tensor<1xi32>
}

func @squaredDifferenceRelu(tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32> {
^bb0(%arg0: tensor<1xi32>, %arg1: tensor<1xi32>):
  %0 = "tf.SquaredDifference"(%arg0, %arg1) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
  %1 = "tf.Relu6"(%0) : (tensor<1xi32>) -> tensor<1xi32>
  return %1: tensor<1xi32>

// CHECK-LABEL: squaredDifferenceRelu
// CHECK:  %0 = tfl.squared_difference %arg0, %arg1 : tensor<1xi32>
// CHECK:  %1 = "tfl.relu6"(%0) : (tensor<1xi32>) -> tensor<1xi32>
// CHECK:  return %1 : tensor<1xi32>
}

func @maxPool2D(%arg0: tensor<1x1x1x16xf32>) -> tensor<1x1x1x16xf32> {
  // OK
  %0 = "tf.MaxPool"(%arg0) {T = "tfdtype$DT_FLOAT", data_format = "NHWC", ksize = [1, 3, 6, 1], padding = "VALID", strides = [1, 3, 1, 1]} : (tensor<1x1x1x16xf32>) -> tensor<1x1x1x16xf32>

  // Unsupported data_format
  %1 = "tf.MaxPool"(%arg0) {T = "tfdtype$DT_FLOAT", data_format = "NCHW", ksize = [1, 3, 6, 1], padding = "VALID", strides = [1, 3, 1, 1]} : (tensor<1x1x1x16xf32>) -> tensor<1x1x1x16xf32>

  // Unsupported ksize
  %2 = "tf.MaxPool"(%arg0) {T = "tfdtype$DT_FLOAT", data_format = "NHWC", ksize = [3, 3, 6, 1], padding = "VALID", strides = [1, 3, 1, 1]} : (tensor<1x1x1x16xf32>) -> tensor<1x1x1x16xf32>

  // Unsupported strides
  %3 = "tf.MaxPool"(%arg0) {T = "tfdtype$DT_FLOAT", data_format = "NHWC", ksize = [1, 3, 6, 1], padding = "VALID", strides = [1, 3, 1, 3]} : (tensor<1x1x1x16xf32>) -> tensor<1x1x1x16xf32>

  %5 = addf %0, %1 : tensor<1x1x1x16xf32>
  %6 = addf %2, %3 : tensor<1x1x1x16xf32>
  %7 = addf %5, %6 : tensor<1x1x1x16xf32>
  return %7 : tensor<1x1x1x16xf32>

// CHECK-LABEL: func @maxPool2D
// CHECK:  %0 = "tfl.max_pool_2d"(%arg0) {filter_height = 3 : i32, filter_width = 6 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 3 : i32, stride_w = 1 : i32} : (tensor<1x1x1x16xf32>) -> tensor<1x1x1x16xf32>
// CHECK:  %1 = "tf.MaxPool"(%arg0)
// CHECK:  %2 = "tf.MaxPool"(%arg0)
// CHECK:  %3 = "tf.MaxPool"(%arg0)
}

func @abs(%arg0: tensor<8x16xf32>) -> tensor<8x16xf32> {
  %0 = "tf.Abs"(%arg0) : (tensor<8x16xf32>) -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>

// CHECK-LABEL:abs
// CHECK:  %0 = "tfl.abs"(%arg0) : (tensor<8x16xf32>) -> tensor<8x16xf32>
}

func @ceil(%arg0: tensor<8x16xf32>) -> tensor<8x16xf32> {
  %0 = "tf.Ceil"(%arg0) : (tensor<8x16xf32>) -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>

// CHECK-LABEL: ceil
// CHECK:  %0 = "tfl.ceil"(%arg0) : (tensor<8x16xf32>) -> tensor<8x16xf32>
// CHECK:  return %0 : tensor<8x16xf32>
}

func @cos(%arg0: tensor<f32>) -> tensor<f32> {
  %0 = "tf.Cos"(%arg0) : (tensor<f32>) -> tensor<f32>
  return %0 : tensor<f32>

// CHECK-LABEL:cos
// CHECK:  %0 = "tfl.cos"(%arg0) : (tensor<f32>) -> tensor<f32>
}

func @elu(%arg0: tensor<11x16xf32>) -> tensor<11x16xf32> {
  %0 = "tf.Elu"(%arg0) : (tensor<11x16xf32>) -> tensor<11x16xf32>
  return %0 : tensor<11x16xf32>

// CHECK-LABEL:elu
// CHECK:  %0 = "tfl.elu"(%arg0) : (tensor<11x16xf32>) -> tensor<11x16xf32>
}

func @expandDims(%arg0: tensor<2x2xf32>, %arg1: tensor<i32>) -> tensor<1x2x2xf32> {
  %0 = "tf.ExpandDims"(%arg0, %arg1) : (tensor<2x2xf32>, tensor<i32>) -> tensor<1x2x2xf32>
  return %0 : tensor<1x2x2xf32>

// CHECK-LABEL:expandDims
// CHECK:  %0 = "tfl.expand_dims"(%arg0, %arg1) : (tensor<2x2xf32>, tensor<i32>) -> tensor<1x2x2xf32>
}

func @squeezeDefault(%arg0: tensor<1x2x2xf32>) -> tensor<2x2xf32> {
  %0 = "tf.Squeeze"(%arg0) : (tensor<1x2x2xf32>) -> tensor<2x2xf32>
  return %0 : tensor<2x2xf32>

// CHECK-LABEL:squeezeDefault
// CHECK:  %0 = "tfl.squeeze"(%arg0) {squeeze_dims = []} : (tensor<1x2x2xf32>) -> tensor<2x2xf32>
}

func @squeezeSingleAxis(%arg0: tensor<2x1x2xf32>) -> tensor<2x2xf32> {
  %0 = "tf.Squeeze"(%arg0) {squeeze_dims = [1]} : (tensor<2x1x2xf32>) -> tensor<2x2xf32>
  return %0 : tensor<2x2xf32>

// CHECK-LABEL:squeezeSingleAxis
// CHECK:  %0 = "tfl.squeeze"(%arg0) {squeeze_dims = [1]} : (tensor<2x1x2xf32>) -> tensor<2x2xf32>
}

func @squeezeTwoAxes(%arg0: tensor<1x2x1x2xf32>) -> tensor<2x2xf32> {
  %0 = "tf.Squeeze"(%arg0) {squeeze_dims = [0, 2]} : (tensor<1x2x1x2xf32>) -> tensor<2x2xf32>
  return %0 : tensor<2x2xf32>

// CHECK-LABEL:squeezeTwoAxes
// CHECK:  %0 = "tfl.squeeze"(%arg0) {squeeze_dims = [0, 2]} : (tensor<1x2x1x2xf32>) -> tensor<2x2xf32>
}

func @gatherScalarIndices(%arg0 : tensor<3x2xf32>, %arg1 : tensor<i32>) -> tensor<2xf32> {
  %0 = "tf.Gather"(%arg0, %arg1) : (tensor<3x2xf32>, tensor<i32>) -> tensor<2xf32>
  return %0 : tensor<2xf32>

// CHECK-LABEL:gatherScalarIndices
// CHECK:  %0 = "tfl.gather"(%arg0, %arg1) {axis = 0 : i32} : (tensor<3x2xf32>, tensor<i32>) -> tensor<2xf32>
}

func @gatherVectorIndices(%arg0 : tensor<2xf32>, %arg1 : tensor<3xi32>) -> tensor<3xf32> {
  %0 = "tf.Gather"(%arg0, %arg1) : (tensor<2xf32>, tensor<3xi32>) -> tensor<3xf32>
  return %0 : tensor<3xf32>

// CHECK-LABEL:gatherVectorIndices
// CHECK:  %0 = "tfl.gather"(%arg0, %arg1) {axis = 0 : i32} : (tensor<2xf32>, tensor<3xi32>) -> tensor<3xf32>
}

func @gatherHigherRankIndices(%arg0 : tensor<2x3x6xf32>, %arg1 : tensor<4x5xi32>) -> tensor<4x5x3x6xf32> {
  %0 = "tf.Gather"(%arg0, %arg1) : (tensor<2x3x6xf32>, tensor<4x5xi32>) -> tensor<4x5x3x6xf32>
  return %0 : tensor<4x5x3x6xf32>

// CHECK-LABEL:gatherHigherRankIndices
// CHECK:  %0 = "tfl.gather"(%arg0, %arg1) {axis = 0 : i32} : (tensor<2x3x6xf32>, tensor<4x5xi32>) -> tensor<4x5x3x6xf32>
}

func @gatherV2VectorIndices(%arg0 : tensor<1x2x20xf32>, %arg1 : tensor<3x5xi32>) -> tensor<1x3x5x20xf32> {
  %0 = constant dense<[1]> : tensor<1xi32>
  %1 = "tf.GatherV2"(%arg0, %arg1, %0) : (tensor<1x2x20xf32>, tensor<3x5xi32>, tensor<1xi32>) -> tensor<1x3x5x20xf32>
  return %1 : tensor<1x3x5x20xf32>

// CHECK-LABEL:gatherV2VectorIndices
// CHECK:  %0 = "tfl.gather"(%arg0, %arg1) {axis = 1 : i32} : (tensor<1x2x20xf32>, tensor<3x5xi32>) -> tensor<1x3x5x20xf32>
}

func @gatherV2VectorIndicesNegAxis(%arg0 : tensor<1x2x20xf32>, %arg1 : tensor<3x5xi32>) -> tensor<1x2x3x5xf32> {
  %0 = constant dense<[-1]> : tensor<1xi32>
  %1 = "tf.GatherV2"(%arg0, %arg1, %0) : (tensor<1x2x20xf32>, tensor<3x5xi32>, tensor<1xi32>) -> tensor<1x2x3x5xf32>
  return %1 : tensor<1x2x3x5xf32>

// CHECK-LABEL:gatherV2VectorIndices
// CHECK:  %0 = "tfl.gather"(%arg0, %arg1) {axis = -1 : i32} : (tensor<1x2x20xf32>, tensor<3x5xi32>) -> tensor<1x2x3x5xf32>
}

func @greater(%arg0: tensor<8x16xf32>, %arg1: tensor<8x16xf32>) -> tensor<8x16xi1> {
  %0 = "tf.Greater"(%arg0, %arg1) : (tensor<8x16xf32>, tensor<8x16xf32>) -> tensor<8x16xi1>
  return %0 : tensor<8x16xi1>

// CHECK-LABEL: greater
// CHECK:  %0 = "tfl.greater"(%arg0, %arg1) : (tensor<8x16xf32>, tensor<8x16xf32>) -> tensor<8x16xi1>
// CHECK:  return %0 : tensor<8x16xi1>
}

func @greater_equal(%arg0: tensor<8x16xf32>, %arg1: tensor<8x16xf32>) -> tensor<8x16xi1> {
  %0 = "tf.GreaterEqual"(%arg0, %arg1) : (tensor<8x16xf32>, tensor<8x16xf32>) -> tensor<8x16xi1>
  return %0 : tensor<8x16xi1>

// CHECK-LABEL: greater_equal
// CHECK:  %0 = "tfl.greater_equal"(%arg0, %arg1) : (tensor<8x16xf32>, tensor<8x16xf32>) -> tensor<8x16xi1>
// CHECK:  return %0 : tensor<8x16xi1>
}

//TODO(b/136498739): Add failure test for non-broadcastable types, since currently
// we can't catch this error.
func @less_equal(%arg0: tensor<8x16xf32>, %arg1: tensor<8x16xf32>) -> tensor<8x16xi1> {
  %0 = "tf.LessEqual"(%arg0, %arg1) : (tensor<8x16xf32>, tensor<8x16xf32>) -> tensor<8x16xi1>
  return %0 : tensor<8x16xi1>

// CHECK-LABEL: less_equal
// CHECK:  %0 = "tfl.less_equal"(%arg0, %arg1) : (tensor<8x16xf32>, tensor<8x16xf32>) -> tensor<8x16xi1>
// CHECK:  return %0 : tensor<8x16xi1>
}

func @rank(%arg0: tensor<11x16xf32>) -> tensor<1xi32> {
  %0 = "tf.Rank"(%arg0) : (tensor<11x16xf32>) -> tensor<1xi32>
  return %0 : tensor<1xi32>

// CHECK-LABEL:rank
// CHECK:  %0 = "tfl.rank"(%arg0) : (tensor<11x16xf32>) -> tensor<1xi32>
}

func @floor(%arg0: tensor<8x16xf32>) -> tensor<8x16xf32> {
  %0 = "tf.Floor"(%arg0) : (tensor<8x16xf32>) -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>

// CHECK-LABEL: floor
// CHECK:  %0 = "tfl.floor"(%arg0) : (tensor<8x16xf32>) -> tensor<8x16xf32>
// CHECK:  return %0 : tensor<8x16xf32>
}

func @floor_div(tensor<8x16xf32>, tensor<8x16xf32>) -> tensor<8x16xf32> {
^bb0(%arg0: tensor<8x16xf32>, %arg1: tensor<8x16xf32>):
  %0 = "tf.FloorDiv"(%arg0, %arg1) : (tensor<8x16xf32>, tensor<8x16xf32>) -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>

// CHECK-LABEL: floor_div
// CHECK:  %0 = tfl.floor_div %arg0, %arg1 : tensor<8x16xf32>
// CHECK:  return %0 : tensor<8x16xf32>
}

func @not_equal(%arg0: tensor<8x16xf32>, %arg1: tensor<8x16xf32>) -> tensor<8x16xi1> {
  %0 = "tf.NotEqual"(%arg0, %arg1) : (tensor<8x16xf32>, tensor<8x16xf32>) -> tensor<8x16xi1>
  return %0 : tensor<8x16xi1>

// CHECK-LABEL: not_equal
// CHECK:  %0 = "tfl.not_equal"(%arg0, %arg1) : (tensor<8x16xf32>, tensor<8x16xf32>) -> tensor<8x16xi1>
// CHECK:  return %0 : tensor<8x16xi1>
}

func @select(%arg0: tensor<8xi1>, %arg1: tensor<8xf32>, %arg2: tensor<8xf32>) -> tensor<8xf32> {
  %0 = "tf.Select"(%arg0, %arg1, %arg2) : (tensor<8xi1>, tensor<8xf32>, tensor<8xf32>) -> tensor<8xf32>
  return %0: tensor<8xf32>

// CHECK-LABEL: select
// CHECK:  %0 = "tfl.select"(%arg0, %arg1, %arg2)
// CHECK:  return %0 : tensor<8xf32>
}

func @sin(%arg0: tensor<f32>) -> tensor<f32> {
  %0 = "tf.Sin"(%arg0) : (tensor<f32>) -> tensor<f32>
  return %0 : tensor<f32>

// CHECK-LABEL:sin
// CHECK:  %0 = "tfl.sin"(%arg0) : (tensor<f32>) -> tensor<f32>
}

func @topk(%arg0: tensor<8xf32>, %arg1: tensor<i32>) -> (tensor<?xf32>, tensor<?xi32>) {
  %0, %1 = "tf.TopKV2"(%arg0, %arg1) : (tensor<8xf32>, tensor<i32>) -> (tensor<?xf32>, tensor<?xi32>)
  return %0, %1: tensor<?xf32>, tensor<?xi32>

// CHECK-LABEL: topk
// CHECK:  %0:2 = "tfl.topk_v2"(%arg0, %arg1)
// CHECK:  return %0
}

func @topk_2(%arg0: tensor<8xf32>) -> (tensor<2xf32>, tensor<2xi32>) {
  %0 = constant dense<2> : tensor<i32>
  %1:2 = "tf.TopKV2"(%arg0, %0) : (tensor<8xf32>, tensor<i32>) -> (tensor<2xf32>, tensor<2xi32>)
  return %1#0, %1#1: tensor<2xf32>, tensor<2xi32>

// CHECK-LABEL: topk_2
// CHECK:  %0:2 = "tfl.topk_v2"(%arg0, %cst)
// CHECK:  return %0
}

func @topk_3(%arg0: tensor<?x8xf32>) -> (tensor<?x2xf32>, tensor<?x2xi32>) {
  %0 = constant dense<2> : tensor<i32>
  %1:2 = "tf.TopKV2"(%arg0, %0) : (tensor<?x8xf32>, tensor<i32>) -> (tensor<?x2xf32>, tensor<?x2xi32>)
  return %1#0, %1#1: tensor<?x2xf32>, tensor<?x2xi32>

// CHECK-LABEL: topk_3
// CHECK:  %0:2 = "tfl.topk_v2"(%arg0, %cst) : (tensor<?x8xf32>, tensor<i32>) -> (tensor<?x2xf32>, tensor<?x2xi32>)
// CHECK:  return %0
}

func @topk_4(%arg0: tensor<1x2x3x4xf32>) -> (tensor<1x2x3x2xf32>, tensor<1x2x3x2xi32>) {
  %0 = constant dense<2> : tensor<i32>
  %1:2 = "tf.TopKV2"(%arg0, %0) : (tensor<1x2x3x4xf32>, tensor<i32>) -> (tensor<1x2x3x2xf32>, tensor<1x2x3x2xi32>)
  return %1#0, %1#1: tensor<1x2x3x2xf32>, tensor<1x2x3x2xi32>

// CHECK-LABEL: topk_4
// CHECK:  %0:2 = "tfl.topk_v2"(%arg0, %cst)
// CHECK:  return %0
}

func @topk_5(%arg0: tensor<*xf32>) -> (tensor<*xf32>, tensor<*xi32>) {
  %0 = constant dense<2> : tensor<i32>
  %1:2 = "tf.TopKV2"(%arg0, %0) : (tensor<*xf32>, tensor<i32>) -> (tensor<*xf32>, tensor<*xi32>)
  return %1#0, %1#1: tensor<*xf32>, tensor<*xi32>

// CHECK-LABEL: topk_5
// CHECK:  %0:2 = "tfl.topk_v2"(%arg0, %cst)
// CHECK:  return %0
}

func @logicalAnd(%arg0: tensor<8xi1>, %arg1: tensor<8xi1>) -> tensor<8xi1> {
  %0 = "tf.LogicalAnd"(%arg0, %arg1) : (tensor<8xi1>, tensor<8xi1>) -> tensor<8xi1>
  return %0: tensor<8xi1>

// CHECK-LABEL: logicalAnd
// CHECK:  %0 = tfl.logical_and %arg0, %arg1 : tensor<8xi1>
// CHECK:  return %0 : tensor<8xi1>
}

func @logicalNot(%arg0: tensor<8xi1>) -> tensor<8xi1> {
  %0 = "tf.LogicalNot"(%arg0) : (tensor<8xi1>) -> tensor<8xi1>
  return %0 : tensor<8xi1>
// CHECK-LABEL: logicalNot
// CHECK:  %0 = "tfl.logical_not"(%arg0) : (tensor<8xi1>) -> tensor<8xi1>
}

func @logicalOr(%arg0: tensor<8xi1>, %arg1: tensor<8xi1>) -> tensor<8xi1> {
  %0 = "tf.LogicalOr"(%arg0, %arg1) : (tensor<8xi1>, tensor<8xi1>) -> tensor<8xi1>
  return %0: tensor<8xi1>

// CHECK-LABEL: logicalOr
// CHECK:  %0 = tfl.logical_or %arg0, %arg1 : tensor<8xi1>
// CHECK:  return %0 : tensor<8xi1>
}

func @addV2(%arg0: tensor<1xi32>, %arg1: tensor<1xi32>) -> tensor<1xi32> {
  %0 = "tf.AddV2"(%arg0, %arg1) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
  return %0 : tensor<1xi32>

// CHECK-LABEL: addV2
// CHECK:  %0 = tfl.add %arg0, %arg1 {fused_activation_function = "NONE"} : tensor<1xi32>
}

func @reverse_v2(%arg0: tensor<1x2x3x4xf32>, %arg1: tensor<1xi32>) -> tensor<1x2x3x4xf32> {
  %0 = "tf.ReverseV2"(%arg0, %arg1) : (tensor<1x2x3x4xf32>, tensor<1xi32>) -> tensor<1x2x3x4xf32>
  return %0 : tensor<1x2x3x4xf32>

// CHECK-LABEL:reverse_v2
// CHECK:  %0 = "tfl.reverse_v2"(%arg0, %arg1) : (tensor<1x2x3x4xf32>, tensor<1xi32>) -> tensor<1x2x3x4xf32>
// CHECK:  return %0 : tensor<1x2x3x4xf32>
}

func @maximum(%arg0: tensor<8x16xf32>, %arg1: tensor<8x16xf32>) -> tensor<8x16xf32> {
  %0 = "tf.Maximum"(%arg0, %arg1) : (tensor<8x16xf32>, tensor<8x16xf32>) -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>

// CHECK-LABEL:maximum
// CHECK:  %0 = "tfl.maximum"(%arg0, %arg1) : (tensor<8x16xf32>, tensor<8x16xf32>) -> tensor<8x16xf32>
}

func @minimum(%arg0: tensor<8x16xf32>, %arg1: tensor<8x16xf32>) -> tensor<8x16xf32> {
  %0 = "tf.Minimum"(%arg0, %arg1) : (tensor<8x16xf32>, tensor<8x16xf32>) -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>

// CHECK-LABEL:minimum
// CHECK:  %0 = "tfl.minimum"(%arg0, %arg1) : (tensor<8x16xf32>, tensor<8x16xf32>) -> tensor<8x16xf32>
}

func @realDiv(%arg0: tensor<8x16xf32>, %arg1: tensor<8x16xf32>) -> tensor<8x16xf32> {
  %0 = "tf.RealDiv"(%arg0, %arg1) : (tensor<8x16xf32>, tensor<8x16xf32>) -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>

// CHECK-LABEL: realDiv
// CHECK:  %0 = tfl.div %arg0, %arg1 {fused_activation_function = "NONE"} : tensor<8x16xf32>
}

func @equal(%arg0: tensor<8x16xf32>, %arg1: tensor<8x16xf32>) -> tensor<8x16xi1> {
  %0 = "tf.Equal"(%arg0, %arg1) : (tensor<8x16xf32>, tensor<8x16xf32>) -> tensor<8x16xi1>
  return %0 : tensor<8x16xi1>

// CHECK-LABEL: equal
// CHECK:  %0 = "tfl.equal"(%arg0, %arg1) : (tensor<8x16xf32>, tensor<8x16xf32>) -> tensor<8x16xi1>
// CHECK:  return %0 : tensor<8x16xi1>
}

func @pad(tensor<2x1x3xf32>, tensor<3x2xi32>) -> tensor<? x f32> {
^bb0(%arg0: tensor<2x1x3xf32>, %arg1: tensor<3x2xi32>):
  %0 = "tf.Pad"(%arg0, %arg1) : (tensor<2x1x3xf32>, tensor<3x2xi32>) -> tensor<? x f32>
  return %0#0 : tensor<? x f32>

  // CHECK-LABEL: pad
  // CHECK:  %0 = "tfl.pad"(%arg0, %arg1) : (tensor<2x1x3xf32>, tensor<3x2xi32>) -> tensor<?xf32>
  // CHECK:  return %0 : tensor<?xf32>
}

func @padv2(tensor<2x1x3xf32>, tensor<3x2xi32>) -> tensor<? x f32> {
^bb0(%arg0: tensor<2x1x3xf32>, %arg1: tensor<3x2xi32>):
  %cst = constant dense<2.0> : tensor<f32>
  %0 = "tf.PadV2"(%arg0, %arg1, %cst) : (tensor<2x1x3xf32>, tensor<3x2xi32>, tensor<f32>) -> tensor<? x f32>
  return %0#0 : tensor<? x f32>

  // CHECK-LABEL: padv2
  // CHECK:  %0 = "tfl.padv2"(%arg0, %arg1, %cst) : (tensor<2x1x3xf32>, tensor<3x2xi32>, tensor<f32>) -> tensor<?xf32>
  // CHECK:  return %0 : tensor<?xf32>
}

func @pack2Tensors(%arg0: tensor<2xi32>, %arg1: tensor<2xi32>) -> tensor<2x2xi32> {
  %0 = "tf.Pack"(%arg0, %arg1) {N = 2 : i64} : (tensor<2xi32>, tensor<2xi32>) -> tensor<2x2xi32>
  return %0 : tensor<2x2xi32>

// CHECK-LABEL: pack2Tensors
// CHECK: %0 = "tfl.pack"(%arg0, %arg1) {axis = 0 : i32, values_count = 2 : i32} : (tensor<2xi32>, tensor<2xi32>) -> tensor<2x2xi32>
}

func @pack3Tensors(%arg0: tensor<2xi32>, %arg1: tensor<2xi32>, %arg2 : tensor<2xi32>) -> tensor<2x3xi32> {
  %0 = "tf.Pack"(%arg0, %arg1, %arg2) {N = 3 : i64, axis = 1 : i64} : (tensor<2xi32>, tensor<2xi32>, tensor<2xi32>) -> tensor<2x3xi32>
  return %0 : tensor<2x3xi32>

// CHECK-LABEL: pack3Tensors
// CHECK: %0 = "tfl.pack"(%arg0, %arg1, %arg2) {axis = 1 : i32, values_count = 3 : i32} : (tensor<2xi32>, tensor<2xi32>, tensor<2xi32>) -> tensor<2x3xi32>
}

func @packNegAxis(%arg0: tensor<2xi32>, %arg1: tensor<2xi32>, %arg2 : tensor<2xi32>) -> tensor<2x3xi32> {
  %0 = "tf.Pack"(%arg0, %arg1, %arg2) {N = 3 : i64, axis = -1 : i64} : (tensor<2xi32>, tensor<2xi32>, tensor<2xi32>) -> tensor<2x3xi32>
  return %0 : tensor<2x3xi32>

// CHECK-LABEL: packNegAxis
// CHECK: %0 = "tfl.pack"(%arg0, %arg1, %arg2) {axis = -1 : i32, values_count = 3 : i32} : (tensor<2xi32>, tensor<2xi32>, tensor<2xi32>) -> tensor<2x3xi32>
}

func @unpack2Tensors(%arg0: tensor<2x2xi32>) -> tensor<2xi32> {
  %0:2 = "tf.Unpack"(%arg0) {num = 2 : i64} : (tensor<2x2xi32>) -> (tensor<2xi32>, tensor<2xi32>)
  return %0#0 : tensor<2xi32>

// CHECK-LABEL: unpack2Tensors
// CHECK: %0:2 = "tfl.unpack"(%arg0) {axis = 0 : i32, num = 2 : i32} : (tensor<2x2xi32>) -> (tensor<2xi32>, tensor<2xi32>)
}

func @unpack3Tensors(%arg0: tensor<2x3xi32>) -> tensor<2xi32> {
  %0:3 = "tf.Unpack"(%arg0) {num = 3 : i64, axis = 1 : i64} : (tensor<2x3xi32>) -> (tensor<2xi32>, tensor<2xi32>, tensor<2xi32>)
  return %0#0 : tensor<2xi32>

// CHECK-LABEL: unpack3Tensors
// CHECK: %0:3 = "tfl.unpack"(%arg0) {axis = 1 : i32, num = 3 : i32} : (tensor<2x3xi32>) -> (tensor<2xi32>, tensor<2xi32>, tensor<2xi32>)
}

func @unpackNegAxis(%arg0: tensor<2x3xi32>) -> tensor<2xi32> {
  %0:3 = "tf.Unpack"(%arg0) {num = 3 : i64, axis = -1 : i64} : (tensor<2x3xi32>) -> (tensor<2xi32>, tensor<2xi32>, tensor<2xi32>)
  return %0#0 : tensor<2xi32>

// CHECK-LABEL: unpackNegAxis
// CHECK: %0:3 = "tfl.unpack"(%arg0) {axis = -1 : i32, num = 3 : i32} : (tensor<2x3xi32>) -> (tensor<2xi32>, tensor<2xi32>, tensor<2xi32>)
}

func @mean(%arg0: tensor<2x2xf32>, %arg1: tensor<1xi32>) -> tensor<1x2xf32> {
  %0 = "tf.Mean"(%arg0, %arg1) : (tensor<2x2xf32>, tensor<1xi32>) -> tensor<1x2xf32>
  return %0 : tensor<1x2xf32>

// CHECK-LABEL: mean
// CHECK:  %0 = "tfl.mean"(%arg0, %arg1) {keep_dims = false} : (tensor<2x2xf32>, tensor<1xi32>) -> tensor<1x2xf32>
}

func @mean_true(%arg0: tensor<2x2xf32>, %arg1: tensor<1xi32>) -> tensor<1x2xf32> {
  %0 = "tf.Mean"(%arg0, %arg1) {keep_dims = true} : (tensor<2x2xf32>, tensor<1xi32>) -> tensor<1x2xf32>
  return %0 : tensor<1x2xf32>

// CHECK-LABEL: mean_true
// CHECK:  %0 = "tfl.mean"(%arg0, %arg1) {keep_dims = true} : (tensor<2x2xf32>, tensor<1xi32>) -> tensor<1x2xf32>
}

func @sum(%arg0: tensor<8x16x16xf32>, %arg1: tensor<2xi32>) -> tensor<?xf32> {
  %0 = "tf.Sum"(%arg0, %arg1) {keep_dims = false} : (tensor<8x16x16xf32>, tensor<2xi32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>

  // CHECK-LABEL: sum
  // CHECK: %0 = "tfl.sum"(%arg0, %arg1) {keep_dims = false} : (tensor<8x16x16xf32>, tensor<2xi32>) -> tensor<?xf32>
}

func @sum_true(%arg0: tensor<8x16x16xf32>, %arg1: tensor<2xi32>) -> tensor<?xf32> {
  %0 = "tf.Sum"(%arg0, %arg1) {keep_dims = true} : (tensor<8x16x16xf32>, tensor<2xi32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>

  // CHECK-LABEL: sum_true
  // CHECK: %0 = "tfl.sum"(%arg0, %arg1) {keep_dims = true} : (tensor<8x16x16xf32>, tensor<2xi32>) -> tensor<?xf32>
}

func @reduce_min(%arg0: tensor<8x16x16xf32>, %arg1: tensor<2xi32>) -> tensor<?xf32> {
  %0 = "tf.Min"(%arg0, %arg1) {keep_dims = false} : (tensor<8x16x16xf32>, tensor<2xi32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>

  // CHECK-LABEL: reduce_min
  // CHECK: %0 = "tfl.reduce_min"(%arg0, %arg1) {keep_dims = false} : (tensor<8x16x16xf32>, tensor<2xi32>) -> tensor<?xf32>
}

func @reduce_min_true(%arg0: tensor<8x16x16xf32>, %arg1: tensor<2xi32>) -> tensor<?xf32> {
  %0 = "tf.Min"(%arg0, %arg1) {keep_dims = true} : (tensor<8x16x16xf32>, tensor<2xi32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>

  // CHECK-LABEL: reduce_min_true
  // CHECK: %0 = "tfl.reduce_min"(%arg0, %arg1) {keep_dims = true} : (tensor<8x16x16xf32>, tensor<2xi32>) -> tensor<?xf32>
}

func @reduce_max(%arg0: tensor<8x16x16xf32>, %arg1: tensor<2xi32>) -> tensor<?xf32> {
  %0 = "tf.Max"(%arg0, %arg1) {keep_dims = false} : (tensor<8x16x16xf32>, tensor<2xi32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>

  // CHECK-LABEL: reduce_max
  // CHECK: %0 = "tfl.reduce_max"(%arg0, %arg1) {keep_dims = false} : (tensor<8x16x16xf32>, tensor<2xi32>) -> tensor<?xf32>
}

func @reduce_max_true(%arg0: tensor<8x16x16xf32>, %arg1: tensor<2xi32>) -> tensor<?xf32> {
  %0 = "tf.Max"(%arg0, %arg1) {keep_dims = true} : (tensor<8x16x16xf32>, tensor<2xi32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>

  // CHECK-LABEL: reduce_max_true
  // CHECK: %0 = "tfl.reduce_max"(%arg0, %arg1) {keep_dims = true} : (tensor<8x16x16xf32>, tensor<2xi32>) -> tensor<?xf32>
}

func @reduce_prod(%arg0: tensor<8x16x16xf32>, %arg1: tensor<2xi32>) -> tensor<?xf32> {
  %0 = "tf.Prod"(%arg0, %arg1) {keep_dims = false} : (tensor<8x16x16xf32>, tensor<2xi32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>

  // CHECK-LABEL: reduce_prod
  // CHECK: %0 = "tfl.reduce_prod"(%arg0, %arg1) {keep_dims = false} : (tensor<8x16x16xf32>, tensor<2xi32>) -> tensor<?xf32>
}

func @reduce_prod_true(%arg0: tensor<8x16x16xf32>, %arg1: tensor<2xi32>) -> tensor<?xf32> {
  %0 = "tf.Prod"(%arg0, %arg1) {keep_dims = true} : (tensor<8x16x16xf32>, tensor<2xi32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>

  // CHECK-LABEL: reduce_prod_true
  // CHECK: %0 = "tfl.reduce_prod"(%arg0, %arg1) {keep_dims = true} : (tensor<8x16x16xf32>, tensor<2xi32>) -> tensor<?xf32>
}

func @batch_to_space_nd(%arg0: tensor<4x2x2x3xf32>, %arg1: tensor<2xi32>, %arg2: tensor<2x2xi32>) -> tensor<?xf32> {
  %0 = "tf.BatchToSpaceND"(%arg0, %arg1, %arg2) : (tensor<4x2x2x3xf32>, tensor<2xi32>, tensor<2x2xi32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
  // CHECK-LABEL: batch_to_space_nd
  // CHECK: %0 = "tfl.batch_to_space_nd"(%arg0, %arg1, %arg2) : (tensor<4x2x2x3xf32>, tensor<2xi32>, tensor<2x2xi32>) -> tensor<?xf32>
}

func @space_to_batch_nd(%arg0: tensor<1x4x4x3xf32>, %arg1: tensor<2xi32>, %arg2: tensor<2x2xi32>) -> tensor<?xf32> {
  %0 = "tf.SpaceToBatchND"(%arg0, %arg1, %arg2) : (tensor<1x4x4x3xf32>, tensor<2xi32>, tensor<2x2xi32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
  // CHECK-LABEL: space_to_batch_nd
  // CHECK: %0 = "tfl.space_to_batch_nd"(%arg0, %arg1, %arg2) : (tensor<1x4x4x3xf32>, tensor<2xi32>, tensor<2x2xi32>) -> tensor<?xf32>
}

func @split(%arg0: tensor<1xi32>, %arg1: tensor<1x4x3x3xf32>) -> tensor<1x4x3xf32> {
  %0:3 = "tf.Split"(%arg0, %arg1) {num_split = 3 : i64} : (tensor<1xi32>, tensor<1x4x3x3xf32>) -> (tensor<1x4x3xf32>, tensor<1x4x3xf32>, tensor<1x4x3xf32>)
  return %0#0 : tensor<1x4x3xf32>

  // CHECK-LABEL: split
  // CHECK: %0:3 = "tfl.split"(%arg0, %arg1) {num_splits = 3 : i32} : (tensor<1xi32>, tensor<1x4x3x3xf32>) -> (tensor<1x4x3xf32>, tensor<1x4x3xf32>, tensor<1x4x3xf32>)
}

func @splitv(%arg0: tensor<1x4x3x3xf32>, %arg1: tensor<2xi32>, %arg2: tensor<1xi32>) -> tensor<1x4x2x3xf32> {
  %0:2 = "tf.SplitV"(%arg0, %arg1, %arg2) {num_split = 2 : i64} : (tensor<1x4x3x3xf32>, tensor<2xi32>, tensor<1xi32>) -> (tensor<1x4x2x3xf32>, tensor<1x4x1x3xf32>)
  return %0#0 : tensor<1x4x2x3xf32>

  // CHECK-LABEL: splitv
  // CHECK: %0:2 = "tfl.split_v"(%arg0, %arg1, %arg2) {num_splits = 2 : i32} : (tensor<1x4x3x3xf32>, tensor<2xi32>, tensor<1xi32>) -> (tensor<1x4x2x3xf32>, tensor<1x4x1x3xf32>)
}

func @matmul_transposed(%arg0: tensor<40x37xf32>, %arg1: tensor<40x37xf32>) -> tensor<40x40xf32> {
  %0 = "tf.MatMul"(%arg0, %arg1) {T = "tfdtype$DT_FLOAT", device = "/device:CPU:0", name = "MatMul", transpose_a = false, transpose_b = true} :
(tensor<40x37xf32>, tensor<40x37xf32>) -> tensor<40x40xf32>
  return %0 : tensor<40x40xf32>
// CHECK-LABEL: matmul_transposed
// CHECK: %0 = "tfl.fully_connected"(%arg0, %arg1, %cst) {fused_activation_function = "NONE", keep_num_dims = false, weights_format = "DEFAULT"} : (tensor<40x37xf32>, tensor<40x37xf32>, none) -> tensor<40x40xf32>
}

func @concat2Tensors(%arg0: tensor<2xi32>, %arg1: tensor<2xi32>) -> tensor<2x2xi32> {
  %0 = constant dense<[1]> : tensor<1xi32>
  %1 = "tf.Concat"(%0, %arg0, %arg1) {N = 2 : i64} : (tensor<1xi32>, tensor<2xi32>, tensor<2xi32>) -> tensor<2x2xi32>
  return %1 : tensor<2x2xi32>

// CHECK-LABEL: concat2Tensors
// CHECK: %0 = "tfl.concatenation"(%arg0, %arg1) {axis = 1 : i32, fused_activation_function = "NONE"} : (tensor<2xi32>, tensor<2xi32>) -> tensor<2x2xi32>
}

func @concat3Tensors(%arg0: tensor<2xi32>, %arg1: tensor<2xi32>, %arg2: tensor<2xi32>) -> tensor<2x3xi32> {
  %0 = constant dense<[-1]> : tensor<1xi32>
  %1 = "tf.Concat"(%0, %arg0, %arg1, %arg2) {N = 3 : i64} : (tensor<1xi32>, tensor<2xi32>, tensor<2xi32>, tensor<2xi32>) -> tensor<2x3xi32>
  return %1 : tensor<2x3xi32>

// CHECK-LABEL: concat3Tensors
// CHECK: %0 = "tfl.concatenation"(%arg0, %arg1, %arg2) {axis = -1 : i32, fused_activation_function = "NONE"} : (tensor<2xi32>, tensor<2xi32>, tensor<2xi32>) -> tensor<2x3xi32>
}

func @concatv2With3Tensors(%arg0: tensor<2xi32>, %arg1: tensor<2xi32>, %arg2: tensor<2xi32>) -> tensor<2x3xi32> {
  %0 = constant dense<[-1]> : tensor<1xi32>
  %1 = "tf.ConcatV2"(%arg0, %arg1, %arg2, %0) {N = 3 : i64} : (tensor<2xi32>, tensor<2xi32>, tensor<2xi32>, tensor<1xi32>) -> tensor<2x3xi32>
  return %1 : tensor<2x3xi32>

// CHECK-LABEL: concatv2With3Tensors
// CHECK: %0 = "tfl.concatenation"(%arg0, %arg1, %arg2) {axis = -1 : i32, fused_activation_function = "NONE"} : (tensor<2xi32>, tensor<2xi32>, tensor<2xi32>) -> tensor<2x3xi32>
}

func @resize_with_bilinear(%arg0: tensor<1x100x100x3xf32>, %arg1: tensor<4xi32>) -> tensor<?xf32> {
  %0 = "tf.ResizeBilinear"(%arg0, %arg1) {align_corners = true} : (tensor<1x100x100x3xf32>, tensor<4xi32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
  // CHECK-LABEL: resize_with_bilinear
  // CHECK: "tfl.resize_bilinear"(%arg0, %arg1) {align_corners = true} : (tensor<1x100x100x3xf32>, tensor<4xi32>) -> tensor<?xf32>
}

// Note: half_pixel_centers isn't supported by TFLite, so it's not
// legalized.
func @resize_with_bilinear_with_half_pixel_centers(%arg0: tensor<1x100x100x3xf32>, %arg1: tensor<4xi32>) -> tensor<?xf32> {
  %0 = "tf.ResizeBilinear"(%arg0, %arg1) {align_corners = true, half_pixel_centers = true} : (tensor<1x100x100x3xf32>, tensor<4xi32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
  // CHECK-LABEL: resize_with_bilinear_with_half_pixel_centers
  // CHECK: "tf.ResizeBilinear"(%arg0, %arg1) {align_corners = true, half_pixel_centers = true}
}

func @strided_slice(%arg0: tensor<12x2x2x5xf32>, %arg1: tensor<1xi32>, %arg2: tensor<1xi32>, %arg3: tensor<1xi32>) -> tensor<1x2x2x5xf32> {
  %0 = "tf.StridedSlice"(%arg0, %arg1, %arg2, %arg3) {begin_mask = 0 : i64, ellipsis_mask = 0 : i64, end_mask = 0 : i64, new_axis_mask = 0 : i64, shrink_axis_mask = 0 : i64} : (tensor<12x2x2x5xf32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1x2x2x5xf32>
  return %0 : tensor<1x2x2x5xf32>
  // CHECK-LABEL: strided_slice
  // CHECK: "tfl.strided_slice"(%arg0, %arg1, %arg2, %arg3) {begin_mask = 0 : i32, ellipsis_mask = 0 : i32, end_mask = 0 : i32, new_axis_mask = 0 : i32, shrink_axis_mask = 0 : i32} : (tensor<12x2x2x5xf32>, tensor<1xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1x2x2x5xf32>
}

func @slice1Tensor(%arg0: tensor<2x3x5xf32>, %arg1: tensor<3xi32>, %arg2: tensor<3xi32>) -> tensor<?x3x5xf32> {
  %0 = "tf.Slice"(%arg0, %arg1, %arg2) : (tensor<2x3x5xf32>, tensor<3xi32>, tensor<3xi32>) -> tensor<?x3x5xf32>
  return %0 : tensor<?x3x5xf32>
  // CHECK-LABEL: slice1Tensor
  // CHECK: "tfl.slice"(%arg0, %arg1, %arg2) : (tensor<2x3x5xf32>, tensor<3xi32>, tensor<3xi32>) -> tensor<?x3x5xf32>
}

func @mirror_pad(tensor<2x1x3xf32>, tensor<3x2xi32>) -> tensor<? x f32> {
^bb0(%arg0: tensor<2x1x3xf32>, %arg1: tensor<3x2xi32>):
  %0 = "tf.MirrorPad"(%arg0, %arg1) { mode = "SYMMETRIC" }: (tensor<2x1x3xf32>, tensor<3x2xi32>) -> tensor<? x f32>
  return %0#0 : tensor<? x f32>

  // CHECK-LABEL: mirror_pad
  // CHECK:  %0 = "tfl.mirror_pad"(%arg0, %arg1) {mode = "SYMMETRIC"} : (tensor<2x1x3xf32>, tensor<3x2xi32>) -> tensor<?xf32>
  // CHECK:  return %0 : tensor<?xf32>
}

func @mirror_pad_reflect(tensor<2x1x3xf32>, tensor<3x2xi32>) -> tensor<? x f32> {
^bb0(%arg0: tensor<2x1x3xf32>, %arg1: tensor<3x2xi32>):
  %0 = "tf.MirrorPad"(%arg0, %arg1) { mode = "REFLECT" }: (tensor<2x1x3xf32>, tensor<3x2xi32>) -> tensor<? x f32>
  return %0#0 : tensor<? x f32>

  // CHECK-LABEL: mirror_pad_reflect
  // CHECK:  %0 = "tfl.mirror_pad"(%arg0, %arg1) {mode = "REFLECT"} : (tensor<2x1x3xf32>, tensor<3x2xi32>) -> tensor<?xf32>
  // CHECK:  return %0 : tensor<?xf32>
}

func @Tanh(%arg0: tensor<1xf32>) -> tensor<1xf32> {
  %2 = "tf.Tanh"(%arg0) : (tensor<1xf32>) -> tensor<1xf32>
  return %2: tensor<1xf32>

// CHECK-LABEL: Tanh
// CHECK:  %0 = "tfl.tanh"(%arg0) : (tensor<1xf32>) -> tensor<1xf32>
}

func @cast(%arg0: tensor<1x2x2x5xi32>) -> tensor<1x2x2x5xf32> {
  %0 = "tf.Cast"(%arg0) : (tensor<1x2x2x5xi32>) -> tensor<1x2x2x5xf32>
  return %0 : tensor<1x2x2x5xf32>

  // CHECK-LABEL: cast
  // CHECK: "tfl.cast"(%arg0) : (tensor<1x2x2x5xi32>) -> tensor<1x2x2x5xf32>
}
