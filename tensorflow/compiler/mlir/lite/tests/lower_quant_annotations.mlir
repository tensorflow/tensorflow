// RUN: litert-opt %s --tfl-lower-quant-annotations | FileCheck %s

func.func private @XlaCallModule_quant.fake_quant.impl_0(tensor<1x28x28x3xf32>) -> tensor<1x28x28x3xf32>
func.func private @XlaCallModule_quant.fake_quant.impl_5_0(tensor<2x1x1x1xf32>) -> tensor<2x1x1x1xf32>
func.func private @XlaCallModule_quant.fake_quant.impl_17_0(tensor<1x30x30x2xf32>) -> tensor<1x30x30x2xf32>
// CHECK-LABEL: func.func @serving_default
func.func @serving_default(%arg0: tensor<1x28x28x3xf32>) -> (tensor<1x30x30x2xf32>) {
  %cst = arith.constant dense<[[0, 0], [1, 1], [1, 1], [0, 0]]> : tensor<4x2xi32>
  %cst_0 = arith.constant dense<[0, 2, 3, 1]> : tensor<4xi32>
  %cst_1 = arith.constant dense<0.000000e+00> : tensor<2xf32>
  %cst_2 = arith.constant dense<[[[[1.0]]], [[[2.0]]]]> : tensor<2x1x1x1xf32>
  // CHECK: %[[QUANT0:.+]] = "tfl.quantize"(%arg0) <{qtype = tensor<1x28x28x3x!quant.uniform<i8:f32, 0.0039197271689772606:-128>>}> : (tensor<1x28x28x3xf32>) -> tensor<1x28x28x3x!quant.uniform<i8:f32, 0.0039197271689772606:-128>>
  // CHECK: %[[DEQUANT0:.+]] = "tfl.dequantize"(%[[QUANT0]]) : (tensor<1x28x28x3x!quant.uniform<i8:f32, 0.0039197271689772606:-128>>) -> tensor<1x28x28x3xf32>
  %0 = stablehlo.composite "quant.fake_quant" %arg0 {composite_attributes = {dtype = "i8", narrow_range = false, scale = dense<0.00391972717> : tensor<1xf32>, zero_point = dense<-128> : tensor<1xi32>}, decomposition = @XlaCallModule_quant.fake_quant.impl_0} : (tensor<1x28x28x3xf32>) -> tensor<1x28x28x3xf32>
  // CHECK: %[[QUANT1:.+]] = "tfl.quantize"(%{{.+}}) <{qtype = tensor<2x1x1x1x!quant.uniform<i8<-127:127>:f32:0, {0.0058756377547979355,0.0049431771039962769}>>}> : (tensor<2x1x1x1xf32>) -> tensor<2x1x1x1x!quant.uniform<i8<-127:127>:f32:0, {0.0058756377547979355,0.0049431771039962769}>>
  // CHECK: %[[DEQUANT1:.+]] = "tfl.dequantize"(%[[QUANT1]]) : (tensor<2x1x1x1x!quant.uniform<i8<-127:127>:f32:0, {0.0058756377547979355,0.0049431771039962769}>>) -> tensor<2x1x1x1xf32>
  %1 = stablehlo.composite "quant.fake_quant" %cst_2 {composite_attributes = {dtype = "i8", narrow_range = true, quantization_dimension = 0 : i32, scale = dense<[0.00587563775, 0.0049431771]> : tensor<2xf32>}, decomposition = @XlaCallModule_quant.fake_quant.impl_5_0} : (tensor<2x1x1x1xf32>) -> tensor<2x1x1x1xf32>
  %2 = "tfl.transpose"(%1, %cst_0) : (tensor<2x1x1x1xf32>, tensor<4xi32>) -> tensor<2x1x1x1xf32>
  %3 = "tfl.pad"(%0, %cst) : (tensor<1x28x28x3xf32>, tensor<4x2xi32>) -> tensor<1x30x30x3xf32>
  %4 = "tfl.conv_2d"(%3, %2, %cst_1) <{dilation_h_factor = 1 : i32, dilation_w_factor = 1 : i32, fused_activation_function = "NONE", padding = "VALID", stride_h = 1 : i32, stride_w = 1 : i32}> : (tensor<1x30x30x3xf32>, tensor<2x1x1x1xf32>, tensor<2xf32>) -> tensor<1x30x30x2xf32>
  // CHECK-OFF: %[[QUANT2:.+]] = "tfl.quantize"(%{{.+}}) <{qtype = tensor<1x30x30x2x!quant.uniform<i8:f32, 0.018049469217658043:8>>}> : (tensor<1x30x30x2xf32>) -> tensor<1x30x30x2x!quant.uniform<i8:f32, 0.018049469217658043:8>>
  // CHECK-OFF: %[[DEQUANT2:.+]] = "tfl.dequantize"(%[[QUANT2]]) : (tensor<1x30x30x2x!quant.uniform<i8:f32, 0.018049469217658043:8>>) -> tensor<1x30x30x2xf32>
  %5 = stablehlo.composite "quant.fake_quant" %4 {composite_attributes = {dtype = "i8", narrow_range = false, scale = dense<0.0180494692> : tensor<1xf32>, zero_point = dense<8> : tensor<1xi32>}, decomposition = @XlaCallModule_quant.fake_quant.impl_17_0} : (tensor<1x30x30x2xf32>) -> tensor<1x30x30x2xf32>
  return %5 : tensor<1x30x30x2xf32>
}