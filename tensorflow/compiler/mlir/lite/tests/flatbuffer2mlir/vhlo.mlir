// RUN: flatbuffer_translate -mlir-to-tflite-flatbuffer --emit-stablehlo-ops=true %s -o - | flatbuffer_translate --tflite-flatbuffer-to-mlir --disable-vhlo-to-stablehlo=true - -o - | FileCheck %s
// test stablehlo roundtrip

// Identity function to make the exporter happy
func.func @main(%arg0: tensor<4xi8>) -> tensor<4xi8> {
  func.return %arg0 : tensor<4xi8>
}

//CHECK:func.func @main(%arg0: tensor<4xi8>) -> tensor<4xi8> attributes {tf.entry_function = {inputs = "arg0", outputs = "arg0"}} {
//CHECK: return %arg0 : tensor<4xi8>
//CHECK:}

func.func @logistic(%arg0: tensor<1x1x1x96xf32>) -> tensor<1x1x1x96xf32> {
  %0 = "vhlo.logistic_v1"(%arg0) : (tensor<1x1x1x96xf32>) -> tensor<1x1x1x96xf32>
  func.return %0 : tensor<1x1x1x96xf32>
}

// CHECK:func.func private @logistic(%arg0: tensor<1x1x1x96xf32>) -> tensor<1x1x1x96xf32> {
// CHECK: %0 = "vhlo.logistic_v1"(%arg0) : (tensor<1x1x1x96xf32>) -> tensor<1x1x1x96xf32>
// CHECK: return %0 : tensor<1x1x1x96xf32>
// CHECK:}

func.func @add(%arg0: tensor<1xf32>, %arg1: tensor<1xf32>) -> tensor<1xf32> {
  %0 = "vhlo.add_v1"(%arg0, %arg1) : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
  func.return %0 : tensor<1xf32>
}

// CHECK:func.func private @add(%arg0: tensor<1xf32>, %arg1: tensor<1xf32>) -> tensor<1xf32> {
// CHECK: %0 = "vhlo.add_v1"(%arg0, %arg1) : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
// CHECK: return %0 : tensor<1xf32>
// CHECK:}

func.func @multiply(%arg0: tensor<1xf32>, %arg1: tensor<1xf32>) -> tensor<1xf32> {
  %0 = "vhlo.multiply_v1"(%arg0, %arg1) : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
  func.return %0 : tensor<1xf32>
}

// CHECK:func.func private @multiply(%arg0: tensor<1xf32>, %arg1: tensor<1xf32>) -> tensor<1xf32> {
// CHECK: %0 = "vhlo.multiply_v1"(%arg0, %arg1) : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
// CHECK: return %0 : tensor<1xf32>
// CHECK:}

func.func @divide(%arg0: tensor<1xf32>, %arg1: tensor<1xf32>) -> tensor<1xf32> {
  %0 = "vhlo.divide_v1"(%arg0, %arg1) : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
  func.return %0 : tensor<1xf32>
}

// CHECK:func.func private @divide(%arg0: tensor<1xf32>, %arg1: tensor<1xf32>) -> tensor<1xf32> {
// CHECK: %0 = "vhlo.divide_v1"(%arg0, %arg1) : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
// CHECK: return %0 : tensor<1xf32>
// CHECK:}

func.func @maximum(%arg0: tensor<1xf32>, %arg1: tensor<1xf32>) -> tensor<1xf32> {
  %0 = "vhlo.maximum_v1"(%arg0, %arg1) : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
  func.return %0 : tensor<1xf32>
}

// CHECK:func.func private @maximum(%arg0: tensor<1xf32>, %arg1: tensor<1xf32>) -> tensor<1xf32> {
// CHECK: %0 = "vhlo.maximum_v1"(%arg0, %arg1) : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
// CHECK: return %0 : tensor<1xf32>
// CHECK:}

func.func @reshape(%arg0 : tensor<1x128xi32>) -> tensor<4x32x1xi32>{
  %0 = "vhlo.reshape_v1"(%arg0) : (tensor<1x128xi32>) -> tensor<4x32x1xi32>
  func.return %0 : tensor<4x32x1xi32>
}

//CHECK:func.func private @reshape(%arg0: tensor<1x128xi32>) -> tensor<4x32x1xi32> {
//CHECK-NEXT: %0 = "vhlo.reshape_v1"(%arg0) : (tensor<1x128xi32>) -> tensor<4x32x1xi32>
//CHECK-NEXT: return %0 : tensor<4x32x1xi32>
//CHECK-NEXT:}

func.func @clamp(%arg0: tensor<f32>, %arg1: tensor<1x256x256x24xf32>, %arg2: tensor<f32>) -> tensor<1x256x256x24xf32>{
  %0 = "vhlo.clamp_v1"(%arg0, %arg1, %arg2) : (tensor<f32>, tensor<1x256x256x24xf32>, tensor<f32>) -> tensor<1x256x256x24xf32>
  return %0 : tensor<1x256x256x24xf32>
}

//CHECK:func.func private @clamp(%arg0: tensor<f32>, %arg1: tensor<1x256x256x24xf32>, %arg2: tensor<f32>) -> tensor<1x256x256x24xf32> {
//CHECK-NEXT: %0 = "vhlo.clamp_v1"(%arg0, %arg1, %arg2) : (tensor<f32>, tensor<1x256x256x24xf32>, tensor<f32>) -> tensor<1x256x256x24xf32>
//CHECK-NEXT: return %0 : tensor<1x256x256x24xf32>
//CHECK-NEXT:}

func.func @concat(%arg0: tensor<1x30x1xi32>, %arg1: tensor<1x30x1xi32>) -> tensor<1x30x2xi32> {
  %0 = "vhlo.concatenate_v1"(%arg0, %arg1) <{dimension = #vhlo.integer_v1<2 : i64>}> : (tensor<1x30x1xi32>, tensor<1x30x1xi32>) -> tensor<1x30x2xi32>
  func.return %0 : tensor<1x30x2xi32>
}

//CHECK:func.func private @concat(%arg0: tensor<1x30x1xi32>, %arg1: tensor<1x30x1xi32>) -> tensor<1x30x2xi32> {
//CHECK-NEXT: %0 = "vhlo.concatenate_v1"(%arg0, %arg1) <{dimension = #vhlo.integer_v1<2 : i64>}> : (tensor<1x30x1xi32>, tensor<1x30x1xi32>) -> tensor<1x30x2xi32>
//CHECK-NEXT: return %0 : tensor<1x30x2xi32>
//CHECK-NEXT:}

func.func @broadcast_in_dim(%arg0: tensor<1x32x256xf32>) -> tensor<4x32x256xf32>{
  %0 = "vhlo.broadcast_in_dim_v1"(%arg0) <{broadcast_dimensions = #vhlo.tensor_v1<dense<[0, 1, 2]> : tensor<3xi64>>}> : (tensor<1x32x256xf32>) -> tensor<4x32x256xf32>
  return %0 : tensor<4x32x256xf32>
}

//CHECK:func.func private @broadcast_in_dim(%arg0: tensor<1x32x256xf32>) -> tensor<4x32x256xf32> {
//CHECK-NEXT: %0 = "vhlo.broadcast_in_dim_v1"(%arg0) <{broadcast_dimensions = #vhlo.tensor_v1<dense<[0, 1, 2]> : tensor<3xi64>>}> : (tensor<1x32x256xf32>) -> tensor<4x32x256xf32>
//CHECK-NEXT: return %0 : tensor<4x32x256xf32>
//CHECK-NEXT:}

func.func @slice(%arg0: tensor<160x20x1xf32>) -> tensor<1x1x1xf32> {
  %0 = "vhlo.slice_v1"(%arg0) <{limit_indices = #vhlo.tensor_v1<dense<[0, 0, 0]> : tensor<3xi64>>, 
                                start_indices = #vhlo.tensor_v1<dense<[1, 1, 1]> : tensor<3xi64>>,
                                strides = #vhlo.tensor_v1<dense<[1, 1, 1]> : tensor<3xi64>>}> : (tensor<160x20x1xf32>) -> tensor<1x1x1xf32>
  return %0 : tensor<1x1x1xf32>
}

//CHECK:func.func private @slice(%arg0: tensor<160x20x1xf32>) -> tensor<1x1x1xf32> {
//CHECK-NEXT: %0 = "vhlo.slice_v1"(%arg0) <{limit_indices = #vhlo.tensor_v1<dense<0> : tensor<3xi64>>, start_indices = #vhlo.tensor_v1<dense<1> : tensor<3xi64>>, strides = #vhlo.tensor_v1<dense<1> : tensor<3xi64>>}> : (tensor<160x20x1xf32>) -> tensor<1x1x1xf32> 
//CHECK-NEXT: return %0 : tensor<1x1x1xf32>
//CHECK-NEXT:}

func.func @convolution(%arg0: tensor<1x1x1600x32xf32>, %arg1: tensor<1x13x1x32xf32>) -> tensor<1x1x1600x32xf32> {
  %0 = "vhlo.convolution_v1"(%arg0, %arg1) <{batch_group_count = #vhlo.integer_v1<1 : i64>,
                                             feature_group_count = #vhlo.integer_v1<32 : i64>,
                                             input_batch_dimension = #vhlo.integer_v1<0 : i64>, 
                                             input_feature_dimension = #vhlo.integer_v1<3 : i64>,
                                             input_spatial_dimensions = #vhlo.tensor_v1<dense<[1, 2]> : tensor<2xi64>>,
                                             kernel_input_feature_dimension = #vhlo.integer_v1<2 : i64>,
                                             kernel_output_feature_dimension = #vhlo.integer_v1<3 : i64>,
                                             kernel_spatial_dimensions = #vhlo.tensor_v1<dense<[0, 1]> : tensor<2xi64>>,
                                             lhs_dilation = #vhlo.tensor_v1<dense<1> : tensor<2xi64>>,
                                             output_batch_dimension = #vhlo.integer_v1<0 : i64>,
                                             output_feature_dimension = #vhlo.integer_v1<3 : i64>,
                                             output_spatial_dimensions = #vhlo.tensor_v1<dense<[1, 2]> : tensor<2xi64>>,
                                             padding = #vhlo.tensor_v1<dense<0> : tensor<2x2xi64>>,
                                             precision_config = #vhlo.array_v1<[#vhlo<precision_v1 DEFAULT>, #vhlo<precision_v1 DEFAULT>]>,
                                             rhs_dilation = #vhlo.tensor_v1<dense<1> : tensor<2xi64>>,
                                             window_reversal = #vhlo.tensor_v1<dense<false> : tensor<2xi1>>,
                                             window_strides = #vhlo.tensor_v1<dense<1> : tensor<2xi64>>}>
                                              : (tensor<1x1x1600x32xf32>, tensor<1x13x1x32xf32>) -> tensor<1x1x1600x32xf32>
  return %0 : tensor<1x1x1600x32xf32>
}

//CHECK:func.func private @convolution(%arg0: tensor<1x1x1600x32xf32>, %arg1: tensor<1x13x1x32xf32>) -> tensor<1x1x1600x32xf32> {
//CHECK: %0 = "vhlo.convolution_v1"(%arg0, %arg1) <{
//CHECK-SAME: batch_group_count = #vhlo.integer_v1<1 : i64>,
//CHECK-SAME: feature_group_count = #vhlo.integer_v1<32 : i64>,
//CHECK-SAME: input_batch_dimension = #vhlo.integer_v1<0 : i64>,
//CHECK-SAME: input_feature_dimension = #vhlo.integer_v1<3 : i64>,
//CHECK-SAME: input_spatial_dimensions = #vhlo.tensor_v1<dense<[1, 2]> : tensor<2xi64>>,
//CHECK-SAME: kernel_input_feature_dimension = #vhlo.integer_v1<2 : i64>,
//CHECK-SAME: kernel_output_feature_dimension = #vhlo.integer_v1<3 : i64>,
//CHECK-SAME: kernel_spatial_dimensions = #vhlo.tensor_v1<dense<[0, 1]> : tensor<2xi64>>,
//CHECK-SAME: lhs_dilation = #vhlo.tensor_v1<dense<1> : tensor<2xi64>>,
//CHECK-SAME: output_batch_dimension = #vhlo.integer_v1<0 : i64>,
//CHECK-SAME: output_feature_dimension = #vhlo.integer_v1<3 : i64>,
//CHECK-SAME: output_spatial_dimensions = #vhlo.tensor_v1<dense<[1, 2]> : tensor<2xi64>>,
//CHECK-SAME: padding = #vhlo.tensor_v1<dense<0> : tensor<2x2xi64>>,
//CHECK-SAME: precision_config = #vhlo.array_v1<[#vhlo<precision_v1 DEFAULT>, #vhlo<precision_v1 DEFAULT>]>,
//CHECK-SAME: rhs_dilation = #vhlo.tensor_v1<dense<1> : tensor<2xi64>>,
//CHECK-SAME: window_reversal = #vhlo.tensor_v1<dense<false> : tensor<2xi1>>,
//CHECK-SAME: window_strides = #vhlo.tensor_v1<dense<1> : tensor<2xi64>>}> : (tensor<1x1x1600x32xf32>, tensor<1x13x1x32xf32>) -> tensor<1x1x1600x32xf32> 
//CHECK-NEXT: return %0 : tensor<1x1x1600x32xf32>
//CHECK-NEXT:}

func.func @reduce(%arg0: tensor<1x16x16x320xf32>, %arg3 : tensor<f32>) -> tensor<1x320xf32> {
  %0 = "vhlo.reduce_v1" (%arg0, %arg3) <{
    dimensions = #vhlo.tensor_v1<dense<[1, 2]> : tensor<2xi64>>
  }> ({
    ^bb0(%arg1: tensor<1xf32>, %arg2: tensor<1xf32>):
    %421 = "vhlo.add_v1"(%arg1, %arg2) : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
    "vhlo.return_v1" (%421) : (tensor<1xf32>) -> ()
   }) : (tensor<1x16x16x320xf32>, tensor<f32>) -> tensor<1x320xf32>
  return %0 : tensor<1x320xf32>
}

//CHECK:func.func private @reduce(%arg0: tensor<1x16x16x320xf32>, %arg1: tensor<f32>) -> tensor<1x320xf32> {
//CHECK-NEXT:  %0 = "vhlo.reduce_v1"(%arg0, %arg1) <{dimensions = #vhlo.tensor_v1<dense<[1, 2]> : tensor<2xi64>>}> ({ 
//CHECK-NEXT:   ^bb0(%arg2: tensor<1xf32>, %arg3: tensor<1xf32>):
//CHECK-NEXT:     %1 = "vhlo.add_v1"(%arg2, %arg3) : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
//CHECK-NEXT:     "vhlo.return_v1"(%1) : (tensor<1xf32>) -> ()
//CHECK-NEXT:   }) : (tensor<1x16x16x320xf32>, tensor<f32>) -> tensor<1x320xf32>
//CHECK-NEXT:  return %0 : tensor<1x320xf32>
//CHECK-NEXT:}

func.func @abs(%arg0: tensor<1x1x1x96xf32>) -> tensor<1x1x1x96xf32> {
  %0 = "vhlo.abs_v1" (%arg0) : (tensor<1x1x1x96xf32>) -> tensor<1x1x1x96xf32>
  func.return %0 : tensor<1x1x1x96xf32>
}

//CHECK:func.func private @abs(%arg0: tensor<1x1x1x96xf32>) -> tensor<1x1x1x96xf32> {
//CHECK-NEXT: %0 = "vhlo.abs_v1"(%arg0) : (tensor<1x1x1x96xf32>) -> tensor<1x1x1x96xf32>
//CHECK-NEXT: return %0 : tensor<1x1x1x96xf32>
//CHECK-NEXT:}

func.func @and(%arg0: tensor<1x30x1xi32>, %arg1: tensor<1x30x1xi32>) -> tensor<1x30x1xi32> {
  %0 = "vhlo.and_v1" (%arg0, %arg1) : (tensor<1x30x1xi32>, tensor<1x30x1xi32>) -> tensor<1x30x1xi32>
  func.return %0 : tensor<1x30x1xi32>
}

//CHECK:func.func private @and(%arg0: tensor<1x30x1xi32>, %arg1: tensor<1x30x1xi32>) -> tensor<1x30x1xi32> {
//CHECK-NEXT: %0 = "vhlo.and_v1"(%arg0, %arg1) : (tensor<1x30x1xi32>, tensor<1x30x1xi32>) -> tensor<1x30x1xi32>
//CHECK-NEXT: return %0 : tensor<1x30x1xi32>
//CHECK-NEXT:}

func.func @cos(%arg0: tensor<1x1x1x96xf32>) -> tensor<1x1x1x96xf32> {
  %0 = "vhlo.cosine_v1" (%arg0) : (tensor<1x1x1x96xf32>) -> tensor<1x1x1x96xf32>
  func.return %0 : tensor<1x1x1x96xf32>
}

//CHECK:func.func private @cos(%arg0: tensor<1x1x1x96xf32>) -> tensor<1x1x1x96xf32> {
//CHECK-NEXT: %0 = "vhlo.cosine_v1"(%arg0) : (tensor<1x1x1x96xf32>) -> tensor<1x1x1x96xf32>
//CHECK-NEXT: return %0 : tensor<1x1x1x96xf32>
//CHECK-NEXT:}

func.func @exp(%arg0: tensor<1x1x1x96xf32>) -> tensor<1x1x1x96xf32> {
  %0 = "vhlo.exponential_v1" (%arg0) : (tensor<1x1x1x96xf32>) -> tensor<1x1x1x96xf32>
  func.return %0 : tensor<1x1x1x96xf32>
}

//CHECK:func.func private @exp(%arg0: tensor<1x1x1x96xf32>) -> tensor<1x1x1x96xf32> {
//CHECK-NEXT: %0 = "vhlo.exponential_v1"(%arg0) : (tensor<1x1x1x96xf32>) -> tensor<1x1x1x96xf32>
//CHECK-NEXT: return %0 : tensor<1x1x1x96xf32>
//CHECK-NEXT:}

func.func @floor(%arg0: tensor<1x1x1x96xf32>) -> tensor<1x1x1x96xf32> {
 %0 = "vhlo.floor_v1" (%arg0) : (tensor<1x1x1x96xf32>) -> tensor<1x1x1x96xf32>
 func.return %0 : tensor<1x1x1x96xf32>
}

//CHECK:func.func private @floor(%arg0: tensor<1x1x1x96xf32>) -> tensor<1x1x1x96xf32> {
//CHECK-NEXT: %0 = "vhlo.floor_v1"(%arg0) : (tensor<1x1x1x96xf32>) -> tensor<1x1x1x96xf32>
//CHECK-NEXT: return %0 : tensor<1x1x1x96xf32>
//CHECK-NEXT:}

func.func @log(%arg0: tensor<1x1x1x96xf32>) -> tensor<1x1x1x96xf32> {
  %0 = "vhlo.log_v1" (%arg0) : (tensor<1x1x1x96xf32>) -> tensor<1x1x1x96xf32>
  func.return %0 : tensor<1x1x1x96xf32>
}

//CHECK:func.func private @log(%arg0: tensor<1x1x1x96xf32>) -> tensor<1x1x1x96xf32> {
//CHECK-NEXT: %0 = "vhlo.log_v1"(%arg0) : (tensor<1x1x1x96xf32>) -> tensor<1x1x1x96xf32>
//CHECK-NEXT: return %0 : tensor<1x1x1x96xf32>
//CHECK-NEXT:}

func.func @min(%arg0: tensor<1x30x1xi32>, %arg1: tensor<1x30x1xi32>) -> tensor<1x30x1xi32> {
  %0 = "vhlo.minimum_v1" (%arg0, %arg1) : (tensor<1x30x1xi32>, tensor<1x30x1xi32>) -> tensor<1x30x1xi32>
  func.return %0 : tensor<1x30x1xi32>
}

//CHECK:func.func private @min(%arg0: tensor<1x30x1xi32>, %arg1: tensor<1x30x1xi32>) -> tensor<1x30x1xi32> {
//CHECK-NEXT: %0 = "vhlo.minimum_v1"(%arg0, %arg1) : (tensor<1x30x1xi32>, tensor<1x30x1xi32>) -> tensor<1x30x1xi32>
//CHECK-NEXT: return %0 : tensor<1x30x1xi32>
//CHECK-NEXT:}

func.func @neg(%arg0: tensor<1x1x1x96xf32>) -> tensor<1x1x1x96xf32> {
  %0 = "vhlo.negate_v1" (%arg0) : (tensor<1x1x1x96xf32>) -> tensor<1x1x1x96xf32>
  func.return %0 : tensor<1x1x1x96xf32>
}

//CHECK:func.func private @neg(%arg0: tensor<1x1x1x96xf32>) -> tensor<1x1x1x96xf32> {
//CHECK-NEXT: %0 = "vhlo.negate_v1"(%arg0) : (tensor<1x1x1x96xf32>) -> tensor<1x1x1x96xf32>
//CHECK-NEXT: return %0 : tensor<1x1x1x96xf32>
//CHECK-NEXT:}

func.func @or(%arg0: tensor<1x30x1xi32>, %arg1: tensor<1x30x1xi32>) -> tensor<1x30x1xi32> {
  %0 = "vhlo.or_v1" (%arg0, %arg1) : (tensor<1x30x1xi32>, tensor<1x30x1xi32>) -> tensor<1x30x1xi32>
  func.return %0 : tensor<1x30x1xi32>
}

//CHECK:func.func private @or(%arg0: tensor<1x30x1xi32>, %arg1: tensor<1x30x1xi32>) -> tensor<1x30x1xi32> {
//CHECK-NEXT: %0 = "vhlo.or_v1"(%arg0, %arg1) : (tensor<1x30x1xi32>, tensor<1x30x1xi32>) -> tensor<1x30x1xi32>
//CHECK-NEXT: return %0 : tensor<1x30x1xi32>
//CHECK-NEXT:}

func.func @power(%arg0: tensor<1x30x1xi32>, %arg1: tensor<1x30x1xi32>) -> tensor<1x30x1xi32> {
  %0 = "vhlo.power_v1" (%arg0, %arg1) : (tensor<1x30x1xi32>, tensor<1x30x1xi32>) -> tensor<1x30x1xi32>
  func.return %0 : tensor<1x30x1xi32>
}

//CHECK:func.func private @power(%arg0: tensor<1x30x1xi32>, %arg1: tensor<1x30x1xi32>) -> tensor<1x30x1xi32> {
//CHECK-NEXT: %0 = "vhlo.power_v1"(%arg0, %arg1) : (tensor<1x30x1xi32>, tensor<1x30x1xi32>) -> tensor<1x30x1xi32>
//CHECK-NEXT: return %0 : tensor<1x30x1xi32>
//CHECK-NEXT:}

func.func @remainder(%arg0: tensor<1x30x1xi32>, %arg1: tensor<1x30x1xi32>) -> tensor<1x30x1xi32> {
 %0 = "vhlo.remainder_v1" (%arg0, %arg1) : (tensor<1x30x1xi32>, tensor<1x30x1xi32>) -> tensor<1x30x1xi32>
 func.return %0 : tensor<1x30x1xi32>
}

//CHECK:func.func private @remainder(%arg0: tensor<1x30x1xi32>, %arg1: tensor<1x30x1xi32>) -> tensor<1x30x1xi32> {
//CHECK-NEXT: %0 = "vhlo.remainder_v1"(%arg0, %arg1) : (tensor<1x30x1xi32>, tensor<1x30x1xi32>) -> tensor<1x30x1xi32>
//CHECK-NEXT: return %0 : tensor<1x30x1xi32>
//CHECK-NEXT:}

func.func @rsqrt(%arg0: tensor<1x1x1x96xf32>) -> tensor<1x1x1x96xf32> {
 %0 = "vhlo.rsqrt_v1" (%arg0) : (tensor<1x1x1x96xf32>) -> tensor<1x1x1x96xf32>
 func.return %0 : tensor<1x1x1x96xf32>
}

//CHECK:func.func private @rsqrt(%arg0: tensor<1x1x1x96xf32>) -> tensor<1x1x1x96xf32> {
//CHECK-NEXT: %0 = "vhlo.rsqrt_v1"(%arg0) : (tensor<1x1x1x96xf32>) -> tensor<1x1x1x96xf32>
//CHECK-NEXT: return %0 : tensor<1x1x1x96xf32>
//CHECK-NEXT:}

func.func @select(%arg0: tensor<1x30x1xi1>, %arg1: tensor<1x30x1xi32>, %arg2: tensor<1x30x1xi32>) -> tensor<1x30x1xi32> {
 %0 = "vhlo.select_v1" (%arg0, %arg1, %arg2) : (tensor<1x30x1xi1>, tensor<1x30x1xi32>, tensor<1x30x1xi32>) -> tensor<1x30x1xi32>
 func.return %0 : tensor<1x30x1xi32>
}

//CHECK:func.func private @select(%arg0: tensor<1x30x1xi1>, %arg1: tensor<1x30x1xi32>, %arg2: tensor<1x30x1xi32>) -> tensor<1x30x1xi32> {
//CHECK-NEXT: %0 = "vhlo.select_v1"(%arg0, %arg1, %arg2) : (tensor<1x30x1xi1>, tensor<1x30x1xi32>, tensor<1x30x1xi32>) -> tensor<1x30x1xi32>
//CHECK-NEXT: return %0 : tensor<1x30x1xi32>
//CHECK-NEXT:}

func.func @sub(%arg0: tensor<1x30x1xi32>, %arg1: tensor<1x30x1xi32>) -> tensor<1x30x1xi32> {
 %0 = "vhlo.subtract_v1" (%arg0, %arg1) : (tensor<1x30x1xi32>, tensor<1x30x1xi32>) -> tensor<1x30x1xi32>
 func.return %0 : tensor<1x30x1xi32>
}

//CHECK:func.func private @sub(%arg0: tensor<1x30x1xi32>, %arg1: tensor<1x30x1xi32>) -> tensor<1x30x1xi32> {
//CHECK-NEXT: %0 = "vhlo.subtract_v1"(%arg0, %arg1) : (tensor<1x30x1xi32>, tensor<1x30x1xi32>) -> tensor<1x30x1xi32>
//CHECK-NEXT: return %0 : tensor<1x30x1xi32>
//CHECK-NEXT:}

func.func @tanh(%arg0: tensor<1x1x1x96xf32>) -> tensor<1x1x1x96xf32> {
 %0 = "vhlo.tanh_v1" (%arg0) : (tensor<1x1x1x96xf32>) -> tensor<1x1x1x96xf32>
 func.return %0 : tensor<1x1x1x96xf32>
}

//CHECK:func.func private @tanh(%arg0: tensor<1x1x1x96xf32>) -> tensor<1x1x1x96xf32> {
//CHECK-NEXT: %0 = "vhlo.tanh_v1"(%arg0) : (tensor<1x1x1x96xf32>) -> tensor<1x1x1x96xf32>
//CHECK-NEXT: return %0 : tensor<1x1x1x96xf32>
//CHECK-NEXT:}

func.func @iota() -> tensor<3x4xf32> {
 %0 = "vhlo.iota_v1" () <{iota_dimension = #vhlo.integer_v1<0 : i64>}> : () -> tensor<3x4xf32>
 return %0 : tensor<3x4xf32>
}

//CHECK:func.func private @iota() -> tensor<3x4xf32> {
//CHECK-NEXT: %0 = "vhlo.iota_v1"() <{iota_dimension = #vhlo.integer_v1<0 : i64>}> : () -> tensor<3x4xf32>
//CHECK-NEXT: return %0 : tensor<3x4xf32>
//CHECK-NEXT:}

func.func @compare(%arg0: tensor<i64>, %arg1: tensor<i64>) -> tensor<i1> {
 %0 = "vhlo.compare_v1" (%arg0, %arg1) <{compare_type = #vhlo<comparison_type_v1 SIGNED>, comparison_direction = #vhlo<comparison_direction_v1 EQ>}> : (tensor<i64>, tensor<i64>) -> tensor<i1>
 func.return %0 : tensor<i1>
}

//CHECK:func.func private @compare(%arg0: tensor<i64>, %arg1: tensor<i64>) -> tensor<i1> {
//CHECK-NEXT: %0 = "vhlo.compare_v1"(%arg0, %arg1) <{compare_type = #vhlo<comparison_type_v1 SIGNED>, comparison_direction = #vhlo<comparison_direction_v1 EQ>}> : (tensor<i64>, tensor<i64>) -> tensor<i1>
//CHECK-NEXT: return %0 : tensor<i1>
//CHECK-NEXT:}

func.func @dynamic_update_slice(%arg0: tensor<4x4xi64>, %arg1: tensor<2x3xi64>, %arg2: tensor<i64>, %arg3: tensor<i64>) -> tensor<4x4xi64> {
  %0 = "vhlo.dynamic_update_slice_v1"(%arg0, %arg1, %arg2, %arg3) : (tensor<4x4xi64>, tensor<2x3xi64>, tensor<i64>, tensor<i64>) -> tensor<4x4xi64>
  return %0 : tensor<4x4xi64>
}

//CHECK:func.func private @dynamic_update_slice(%arg0: tensor<4x4xi64>, %arg1: tensor<2x3xi64>, %arg2: tensor<i64>, %arg3: tensor<i64>) -> tensor<4x4xi64> {
//CHECK-NEXT: %0 = "vhlo.dynamic_update_slice_v1"(%arg0, %arg1, %arg2, %arg3) : (tensor<4x4xi64>, tensor<2x3xi64>, tensor<i64>, tensor<i64>) -> tensor<4x4xi64>
//CHECK-NEXT: return %0 : tensor<4x4xi64>
//CHECK-NEXT:}

func.func @dyanmic_slice(%arg0: tensor<3x3xi64>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> tensor<3x3xi64> {
  %0 = "vhlo.dynamic_slice_v1"(%arg0, %arg1, %arg2) <{
    slice_sizes = #vhlo.tensor_v1<dense<[3, 3]> : tensor<2xi64>>
  }> : (tensor<3x3xi64>, tensor<i64>, tensor<i64>) -> tensor<3x3xi64>
  return %0 : tensor<3x3xi64>
}

//CHECK:func.func private @dyanmic_slice(%arg0: tensor<3x3xi64>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> tensor<3x3xi64> {
//CHECK-NEXT: %0 = "vhlo.dynamic_slice_v1"(%arg0, %arg1, %arg2) <{
//CHECK-SAME:   slice_sizes = #vhlo.tensor_v1<dense<3> : tensor<2xi64>>
//CHECK-SAME: }> : (tensor<3x3xi64>, tensor<i64>, tensor<i64>) -> tensor<3x3xi64>
//CHECK-NEXT: return %0 : tensor<3x3xi64>
//CHECK-NEXT:}

func.func @pad(%arg0: tensor<1x160x1xf32>, %arg1: tensor<f32>) -> tensor<1x161x1xf32> {
  %0 = "vhlo.pad_v1" (%arg0, %arg1) <{edge_padding_low = #vhlo.tensor_v1<dense<[0, 1, 0]> : tensor<3xi64>>,
                                      edge_padding_high = #vhlo.tensor_v1<dense<0> : tensor<3xi64>>,
                                      interior_padding = #vhlo.tensor_v1<dense<0> : tensor<3xi64>>}> : (tensor<1x160x1xf32>, tensor<f32>) -> tensor<1x161x1xf32>
  return %0 : tensor<1x161x1xf32>
}

//CHECK:func.func private @pad(%arg0: tensor<1x160x1xf32>, %arg1: tensor<f32>) -> tensor<1x161x1xf32> {
//CHECK-NEXT: %0 = "vhlo.pad_v1"(%arg0, %arg1) <{edge_padding_high = #vhlo.tensor_v1<dense<0> : tensor<3xi64>>,
//CHECK-SAME:                                     edge_padding_low = #vhlo.tensor_v1<dense<[0, 1, 0]> : tensor<3xi64>>,
//CHECK-SAME:                                     interior_padding = #vhlo.tensor_v1<dense<0> : tensor<3xi64>>}> : (tensor<1x160x1xf32>, tensor<f32>) -> tensor<1x161x1xf32>
//CHECK-NEXT: return %0 : tensor<1x161x1xf32>
//CHECK-NEXT:}

func.func @convert(%arg0: tensor<2xf64>) -> tensor<2xf32> {
  %0 = "vhlo.convert_v1" (%arg0) : (tensor<2xf64>) -> tensor<2xf32>
  return %0 : tensor<2xf32>
}

//CHECK:func.func private @convert(%arg0: tensor<2xf64>) -> tensor<2xf32> {
//CHECK-NEXT: %0 = "vhlo.convert_v1"(%arg0) : (tensor<2xf64>) -> tensor<2xf32>
//CHECK-NEXT: return %0 : tensor<2xf32>
//CHECK-NEXT:}

func.func @reduce_window(%arg0: tensor<1x160x1xf32>, %arg1: tensor<f32>) -> tensor<1x160x1xf32> {
  %0 = "vhlo.reduce_window_v1"(%arg0, %arg1) <{base_dilations = #vhlo.tensor_v1<dense<1> : tensor<3xi64>>,
        padding = #vhlo.tensor_v1<dense<[[0, 0], [159, 0], [0, 0]]> : tensor<3x2xi64>>,
        window_dilations = #vhlo.tensor_v1<dense<1> : tensor<3xi64>>,
        window_dimensions = #vhlo.tensor_v1<dense<[1, 160, 1]> : tensor<3xi64>>,
        window_strides = #vhlo.tensor_v1<dense<1> : tensor<3xi64>>}> ({
    ^bb0(%arg23: tensor<f32>, %arg24: tensor<f32>):
      %1112 = "vhlo.add_v1" (%arg23, %arg24) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "vhlo.return_v1" (%1112) : (tensor<f32>) -> ()
    }) : (tensor<1x160x1xf32>, tensor<f32>) -> tensor<1x160x1xf32>
  return %0 : tensor<1x160x1xf32>
}

//CHECK:func.func private @reduce_window(%arg0: tensor<1x160x1xf32>, %arg1: tensor<f32>) -> tensor<1x160x1xf32> {
//CHECK-NEXT: %0 = "vhlo.reduce_window_v1"(%arg0, %arg1) <{base_dilations = #vhlo.tensor_v1<dense<1> : tensor<3xi64>>,
//CHECK-SAME{LITERAL}:        padding = #vhlo.tensor_v1<dense<[[0, 0], [159, 0], [0, 0]]> : tensor<3x2xi64>>,
//CHECK-SAME:                 window_dilations = #vhlo.tensor_v1<dense<1> : tensor<3xi64>>,
//CHECK-SAME:                 window_dimensions = #vhlo.tensor_v1<dense<[1, 160, 1]> : tensor<3xi64>>,
//CHECK-SAME:                 window_strides = #vhlo.tensor_v1<dense<1> : tensor<3xi64>>}> ({
//CHECK-NEXT:    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
//CHECK-NEXT:      %1 = "vhlo.add_v1"(%arg2, %arg3) : (tensor<f32>, tensor<f32>) -> tensor<f32>
//CHECK-NEXT:      "vhlo.return_v1"(%1) : (tensor<f32>) -> ()
//CHECK-NEXT:    }) : (tensor<1x160x1xf32>, tensor<f32>) -> tensor<1x160x1xf32>
//CHECK-NEXT: return %0 : tensor<1x160x1xf32>
//CHECK-NEXT:}

func.func @dot_general(%arg0: tensor<1x1x167xf32>, %arg1: tensor<167x64xf32>) -> tensor<1x1x64xf32> {
  %0 = "vhlo.dot_general_v1"(%arg0, %arg1) <{
    lhs_batching_dimensions = #vhlo.tensor_v1<dense<0> : tensor<1xi64>>,
    lhs_contracting_dimensions = #vhlo.tensor_v1<dense<2> : tensor<1xi64>>,
    rhs_batching_dimensions = #vhlo.tensor_v1<dense<0> : tensor<1xi64>>,
    rhs_contracting_dimensions = #vhlo.tensor_v1<dense<0> : tensor<1xi64>>,
    precision_config = #vhlo.array_v1<[#vhlo<precision_v1 DEFAULT>, #vhlo<precision_v1 DEFAULT>]>}> : (tensor<1x1x167xf32>, tensor<167x64xf32>) -> tensor<1x1x64xf32>
  return %0 : tensor<1x1x64xf32>
}

//CHECK:func.func private @dot_general(%arg0: tensor<1x1x167xf32>, %arg1: tensor<167x64xf32>) -> tensor<1x1x64xf32> {
//CHECK-NEXT: %0 = "vhlo.dot_general_v1"(%arg0, %arg1) <{
//CHECK-SAME:    lhs_batching_dimensions = #vhlo.tensor_v1<dense<0> : tensor<1xi64>>,
//CHECK-SAME:    lhs_contracting_dimensions = #vhlo.tensor_v1<dense<2> : tensor<1xi64>>,
//CHECK-SAME:    precision_config = #vhlo.array_v1<[#vhlo<precision_v1 DEFAULT>, #vhlo<precision_v1 DEFAULT>]>,
//CHECK-SAME:    rhs_batching_dimensions = #vhlo.tensor_v1<dense<0> : tensor<1xi64>>,
//CHECK-SAME:    rhs_contracting_dimensions = #vhlo.tensor_v1<dense<0> : tensor<1xi64>>}> : (tensor<1x1x167xf32>, tensor<167x64xf32>) -> tensor<1x1x64xf32>
//CHECK-NEXT: return %0 : tensor<1x1x64xf32>
//CHECK-NEXT:}

func.func @sort(%arg0: tensor<448xf32>, %arg1: tensor<448xi32>) -> tensor<448xf32> {
  %0, %1 = "vhlo.sort_v1"(%arg0, %arg1) <{dimension = #vhlo.integer_v1<0 : i64>, is_stable = #vhlo.bool_v1<true>}> ({
    ^bb0(%arg23: tensor<f32>, %arg24: tensor<f32>, %arg25: tensor<i32>, %arg26: tensor<i32>):
      %1112 = "vhlo.compare_v1"(%arg23, %arg24) <{compare_type = #vhlo<comparison_type_v1 TOTALORDER>, comparison_direction = #vhlo<comparison_direction_v1 GT>}> : (tensor<f32>, tensor<f32>) -> tensor<i1>
      "vhlo.return_v1"(%1112) : (tensor<i1>) -> ()
    }) : (tensor<448xf32>, tensor<448xi32>) -> (tensor<448xf32>, tensor<448xi32>)
  return %0 : tensor<448xf32>
}

//CHECK:func.func private @sort(%arg0: tensor<448xf32>, %arg1: tensor<448xi32>) -> tensor<448xf32> {
//CHECK-NEXT:  %0:2 = "vhlo.sort_v1"(%arg0, %arg1) <{dimension = #vhlo.integer_v1<0 : i64>, is_stable = #vhlo.bool_v1<true>}> ({
//CHECK-NEXT:    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>, %arg4: tensor<i32>, %arg5: tensor<i32>):
//CHECK-NEXT:      %1 = "vhlo.compare_v1"(%arg2, %arg3) <{compare_type = #vhlo<comparison_type_v1 TOTALORDER>, comparison_direction = #vhlo<comparison_direction_v1 GT>}> : (tensor<f32>, tensor<f32>) -> tensor<i1>
//CHECK-NEXT:      "vhlo.return_v1"(%1) : (tensor<i1>) -> ()
//CHECK-NEXT:    }) : (tensor<448xf32>, tensor<448xi32>) -> (tensor<448xf32>, tensor<448xi32>) 
//CHECK-NEXT: return %0#0 : tensor<448xf32>
//CHECK-NEXT:}

func.func @while(%init_i: tensor<i64>, %init_sum: tensor<i64>) -> tensor<i64>{
  %0, %1 = "vhlo.while_v1"(%init_i, %init_sum) ({
    ^bb0(%arg0: tensor<i64>, %arg1: tensor<i64>):
      %cond = "vhlo.compare_v1"(%arg0, %arg1) {
        comparison_direction = #vhlo<comparison_direction_v1 LT>,
        compare_type = #vhlo<comparison_type_v1 NOTYPE>
      } : (tensor<i64>, tensor<i64>) -> tensor<i1>
      "vhlo.return_v1"(%cond) : (tensor<i1>) -> ()
    }, {
    ^bb0(%arg0: tensor<i64>, %arg1: tensor<i64>):
      %new_sum = "vhlo.add_v1"(%arg1, %arg1) : (tensor<i64>, tensor<i64>) -> tensor<i64>
      %new_i = "vhlo.add_v1" (%arg0, %arg1) : (tensor<i64>, tensor<i64>) -> tensor<i64>
      "vhlo.return_v1"(%new_i, %new_sum) : (tensor<i64>, tensor<i64>) ->()
  }) : (tensor<i64>, tensor<i64>) -> (tensor<i64>, tensor<i64>)
  return %0 : tensor<i64>
}

//CHECK:func.func private @while(%arg0: tensor<i64>, %arg1: tensor<i64>) -> tensor<i64> {
//CHECK-NEXT: %0:2 = "vhlo.while_v1"(%arg0, %arg1) ({
//CHECK-NEXT:    ^bb0(%arg2: tensor<i64>, %arg3: tensor<i64>):
//CHECK-NEXT:    %1 = "vhlo.compare_v1"(%arg2, %arg3) <{compare_type = #vhlo<comparison_type_v1 NOTYPE>, comparison_direction = #vhlo<comparison_direction_v1 LT>}> : (tensor<i64>, tensor<i64>) -> tensor<i1> 
//CHECK-NEXT:    "vhlo.return_v1"(%1) : (tensor<i1>) -> ()
//CHECK-NEXT:  }, {
//CHECK-NEXT:  ^bb0(%arg2: tensor<i64>, %arg3: tensor<i64>):
//CHECK-NEXT:    %1 = "vhlo.add_v1"(%arg3, %arg3) : (tensor<i64>, tensor<i64>) -> tensor<i64>
//CHECK-NEXT:    %2 = "vhlo.add_v1"(%arg2, %arg3) : (tensor<i64>, tensor<i64>) -> tensor<i64>
//CHECK-NEXT:    "vhlo.return_v1"(%2, %1) : (tensor<i64>, tensor<i64>) -> ()
//CHECK-NEXT:  }) : (tensor<i64>, tensor<i64>) -> (tensor<i64>, tensor<i64>)
//CHECK-NEXT: return %0#0 : tensor<i64>
//CHECK-NEXT:}

func.func @gather(%operand: tensor<3x4x2xi32>, %start_indices: tensor<2x3x2xi64>) -> tensor<2x3x2x2xi32>{
  %result = "vhlo.gather_v1"(%operand, %start_indices) <{
    offset_dims = #vhlo.tensor_v1<dense<[2, 3]> : tensor<2xi64>>,
    collapsed_slice_dims = #vhlo.tensor_v1<dense<0> : tensor<1xi64>>,
    start_index_map = #vhlo.tensor_v1<dense<[1, 0]> : tensor<2xi64>>,
    index_vector_dim = #vhlo.integer_v1<2 : i64>,
    slice_sizes = #vhlo.tensor_v1<dense<[1, 2, 2]> : tensor<3xi64>>,
    indices_are_sorted = #vhlo.bool_v1<false>
  }> : (tensor<3x4x2xi32>, tensor<2x3x2xi64>) -> tensor<2x3x2x2xi32>
  return %result : tensor<2x3x2x2xi32>
}


// CHECK: func.func private @gather(%arg0: tensor<3x4x2xi32>, %arg1: tensor<2x3x2xi64>) -> tensor<2x3x2x2xi32> {
// CHECK-NEXT: %0 = "vhlo.gather_v1"(%arg0, %arg1) <{collapsed_slice_dims = #vhlo.tensor_v1<dense<0> : tensor<1xi64>>, index_vector_dim = #vhlo.integer_v1<2 : i64>, indices_are_sorted = #vhlo.bool_v1<false>, offset_dims = #vhlo.tensor_v1<dense<[2, 3]> : tensor<2xi64>>, slice_sizes = #vhlo.tensor_v1<dense<[1, 2, 2]> : tensor<3xi64>>, start_index_map = #vhlo.tensor_v1<dense<[1, 0]> : tensor<2xi64>>}> : (tensor<3x4x2xi32>, tensor<2x3x2xi64>) -> tensor<2x3x2x2xi32> 
// CHECK-NEXT: return %0 : tensor<2x3x2x2xi32>
// CHECK-NEXT:}

func.func @transpose(%arg0: tensor<2x3x2xi32>) -> tensor<2x3x2xi32> {
  %0 = "vhlo.transpose_v1"(%arg0) <{permutation = #vhlo.tensor_v1<dense<[2, 1, 0]> : tensor<3xi64>>}> : (tensor<2x3x2xi32>) -> tensor<2x3x2xi32>
  return %0 : tensor<2x3x2xi32>
}

// CHECK:func.func private @transpose(%arg0: tensor<2x3x2xi32>) -> tensor<2x3x2xi32> {
// CHECK-NEXT:  %0 = "vhlo.transpose_v1"(%arg0) <{permutation = #vhlo.tensor_v1<dense<[2, 1, 0]> : tensor<3xi64>>}> : (tensor<2x3x2xi32>) -> tensor<2x3x2xi32>
// CHECK-NEXT:  return %0 : tensor<2x3x2xi32>
// CHECK-NEXT:}

func.func @rng_bit_generator(%arg0: tensor<2xui64>) -> (tensor<2xui64>, tensor<10x12xui32>) {
  %output_state, %output = "vhlo.rng_bit_generator_v1"(%arg0) <{rng_algorithm = #vhlo<rng_algorithm_v1 DEFAULT>}> : (tensor<2xui64>) -> (tensor<2xui64>, tensor<10x12xui32>)
  func.return %output_state, %output : tensor<2xui64>, tensor<10x12xui32>
}

// CHECK:func.func private @rng_bit_generator(%arg0: tensor<2xui64>) -> (tensor<2xui64>, tensor<10x12xui32>) {
// CHECK-NEXT:  %output_state, %output = "vhlo.rng_bit_generator_v1"(%arg0) <{rng_algorithm = #vhlo<rng_algorithm_v1 DEFAULT>}> : (tensor<2xui64>) -> (tensor<2xui64>, tensor<10x12xui32>)
// CHECK-NEXT:  return %output_state, %output : tensor<2xui64>, tensor<10x12xui32>
// CHECK-NEXT:}

func.func @scatter(%input_tensor: tensor<200x100x300xf32>,
    %scatter_indices: tensor<10x2xi32>, %updates: tensor<10x300xf32>) ->
      tensor<200x100x300xf32> {
  %0 = "vhlo.scatter_v1" (%input_tensor, %scatter_indices, %updates) <{
    update_window_dims = #vhlo.tensor_v1<dense<1> : tensor<1xi64>>,
    inserted_window_dims = #vhlo.tensor_v1<dense<[0, 1]> : tensor<2xi64>>,
    scatter_dims_to_operand_dims = #vhlo.tensor_v1<dense<[0, 1]> : tensor<2xi64>>,
    index_vector_dim = #vhlo.integer_v1<1 : i64>,
    indices_are_sorted = #vhlo.bool_v1<true>,
    unique_indices = #vhlo.bool_v1<true>}> ({
  ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):
    "vhlo.return_v1"(%lhs) : (tensor<f32>) -> ()
  }): (tensor<200x100x300xf32>, tensor<10x2xi32>, tensor<10x300xf32>) ->
      tensor<200x100x300xf32>
  func.return %0 : tensor<200x100x300xf32>
}

// CHECK-LABEL: func.func private @scatter(%arg0: tensor<200x100x300xf32>, %arg1: tensor<10x2xi32>, %arg2: tensor<10x300xf32>) -> tensor<200x100x300xf32> {
// CHECK-NEXT:  %0 = "vhlo.scatter_v1"(%arg0, %arg1, %arg2) <{index_vector_dim = #vhlo.integer_v1<1 : i64>, indices_are_sorted = #vhlo.bool_v1<true>, inserted_window_dims = #vhlo.tensor_v1<dense<[0, 1]> : tensor<2xi64>>, scatter_dims_to_operand_dims = #vhlo.tensor_v1<dense<[0, 1]> : tensor<2xi64>>, unique_indices = #vhlo.bool_v1<true>, update_window_dims = #vhlo.tensor_v1<dense<1> : tensor<1xi64>>}> ({
// CHECK-NEXT:  ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
// CHECK-NEXT:     "vhlo.return_v1"(%arg3) : (tensor<f32>) -> ()
// CHECK-NEXT:  }) : (tensor<200x100x300xf32>, tensor<10x2xi32>, tensor<10x300xf32>) -> tensor<200x100x300xf32>
// CHECK-NEXT:  return %0 : tensor<200x100x300xf32>
// CHECK-NEXT: }