// Copyright 2022 The TensorFlow Runtime Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Quantization as a function library.
//
// Internal functions should be marked as private. They will be inlined and
// deleted in `InsertQuantizedFunctionsPass`.
//
// Function template can generate functions with different parameters. Ex:
// ```
// parameters[
//   {"key1": "value11", "key2": "value21"},
//   {"key1": "value12", "key2": "value22"},
// ]
// func.func func_name_${key1}_fn (...) {
//   ...${key2}...
// }
// ```
// The above template with generate two functions by substituting `key1` and
// `key2` with given values.

module {
  // Rescales to the output scale and zero point.
  func.func private @internal_rescale_fn(%accumulation : tensor<*xi32>,
                         %input_scale : tensor<*xf32>, %input_zp : tensor<*xi32>,
                         %filter_scale : tensor<*xf32>, %filter_zp : tensor<*xi32>,
                         %out_scale : tensor<*xf32>, %out_zp : tensor<*xi32>) -> tensor<*xf32> {
    %scale_prod = "tf.Mul"(%input_scale, %filter_scale) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
    %rescale_factor = "tf.Div"(%scale_prod, %out_scale) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
    %float_out_zp = "tf.Cast"(%out_zp) {Truncate = false} : (tensor<*xi32>) -> tensor<*xf32>

    %cast = "tf.Cast"(%accumulation) {Truncate = false} : (tensor<*xi32>) -> tensor<*xf32>
    %mul = "tf.Mul"(%cast, %rescale_factor) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
    %add = "tf.AddV2"(%mul, %float_out_zp) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
    func.return %add : tensor<*xf32>
  }

  func.func private @internal_dequantize_i8_fn(%input : tensor<*xi8>, %scale : tensor<*xf32>, %zp : tensor<*xi32>) -> tensor<*xf32> {
    %input_i32 = "tf.Cast"(%input) : (tensor<*xi8>) -> tensor<*xi32>
    %output = "tf.Sub"(%input_i32, %zp) : (tensor<*xi32>, tensor<*xi32>) -> tensor<*xi32>
    %cast = "tf.Cast"(%output) : (tensor<*xi32>) -> tensor<*xf32>
    %mul = "tf.Mul"(%cast, %scale) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
    func.return %mul : tensor<*xf32>
  }

  // Requantizes and clips to the range of quantized type if there is no specific activation.
  func.func private @internal_requantize_no_activation_fn(%accumulation : tensor<*xi32>,
                         %input_scale : tensor<*xf32>, %input_zp : tensor<*xi32>,
                         %filter_scale : tensor<*xf32>, %filter_zp : tensor<*xi32>,
                         %out_scale : tensor<*xf32>, %out_zp : tensor<*xi32>) -> tensor<*xi8> {
    %rescale = "tf.PartitionedCall"(%accumulation, %input_scale, %input_zp, %filter_scale, %filter_zp,
                                %out_scale, %out_zp) {
        config = "", config_proto = "", executor_type = "", f=@internal_rescale_fn
      } : (tensor<*xi32>, tensor<*xf32>, tensor<*xi32>, tensor<*xf32>, tensor<*xi32>,
             tensor<*xf32>, tensor<*xi32>) -> tensor<*xf32>
    %i8_min = "tf.Const"() {value = dense<-128.0> : tensor<f32>} : () -> tensor<f32>
    %i8_max = "tf.Const"() {value = dense<127.0> : tensor<f32>} : () -> tensor<f32>

    %clamp_max = "tf.Maximum"(%rescale, %i8_min) : (tensor<*xf32>, tensor<f32>) -> tensor<*xf32>
    %clamp_min = "tf.Minimum"(%clamp_max, %i8_max) : (tensor<*xf32>, tensor<f32>) -> tensor<*xf32>
    %round = "tf.Round"(%clamp_min) : (tensor<*xf32>) -> tensor<*xf32>
    %cast = "tf.Cast"(%round) {Truncate = false} : (tensor<*xf32>) -> tensor<*xi8>
    func.return %cast : tensor<*xi8>
  }

  // Requantizes and applies quantized Relu by clipping.
  func.func private @internal_requantize_and_relu_fn(%accumulation : tensor<*xi32>,
                         %input_scale : tensor<*xf32>, %input_zp : tensor<*xi32>,
                         %filter_scale : tensor<*xf32>, %filter_zp : tensor<*xi32>,
                         %out_scale : tensor<*xf32>, %out_zp : tensor<*xi32>) -> tensor<*xi8> {
    %rescale = "tf.PartitionedCall"(%accumulation, %input_scale, %input_zp, %filter_scale, %filter_zp,
                                %out_scale, %out_zp) {
        config = "", config_proto = "", executor_type = "", f=@internal_rescale_fn
      } : (tensor<*xi32>, tensor<*xf32>, tensor<*xi32>, tensor<*xf32>, tensor<*xi32>,
             tensor<*xf32>, tensor<*xi32>) -> tensor<*xf32>
    %i8_min = "tf.Const"() {value = dense<-128.0> : tensor<f32>} : () -> tensor<f32>
    %i8_max = "tf.Const"() {value = dense<127.0> : tensor<f32>} : () -> tensor<f32>
    %float_out_zp = "tf.Cast"(%out_zp) {Truncate = false} : (tensor<*xi32>) -> tensor<*xf32>
    %clip_min = "tf.Maximum"(%i8_min, %float_out_zp) : (tensor<f32>, tensor<*xf32>) -> tensor<*xf32>

    %clamp_max = "tf.Maximum"(%rescale, %clip_min) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
    %clamp_min = "tf.Minimum"(%clamp_max, %i8_max) : (tensor<*xf32>, tensor<f32>) -> tensor<*xf32>
    %round = "tf.Round"(%clamp_min) : (tensor<*xf32>) -> tensor<*xf32>
    %cast = "tf.Cast"(%round) {Truncate = false} : (tensor<*xf32>) -> tensor<*xi8>
    func.return %cast : tensor<*xi8>
  }

  // Requantizes and applies quantized Relu6 by clipping.
  func.func private @internal_requantize_and_relu6_fn(%accumulation : tensor<*xi32>,
                         %input_scale : tensor<*xf32>, %input_zp : tensor<*xi32>,
                         %filter_scale : tensor<*xf32>, %filter_zp : tensor<*xi32>,
                         %out_scale : tensor<*xf32>, %out_zp : tensor<*xi32>) -> tensor<*xi8> {
    %rescale = "tf.PartitionedCall"(%accumulation, %input_scale, %input_zp, %filter_scale, %filter_zp,
                                %out_scale, %out_zp) {
        config = "", config_proto = "", executor_type = "", f=@internal_rescale_fn
      } : (tensor<*xi32>, tensor<*xf32>, tensor<*xi32>, tensor<*xf32>, tensor<*xi32>,
             tensor<*xf32>, tensor<*xi32>) -> tensor<*xf32>
    %i8_min = "tf.Const"() {value = dense<-128.0> : tensor<f32>} : () -> tensor<f32>
    %i8_max = "tf.Const"() {value = dense<127.0> : tensor<f32>} : () -> tensor<f32>
    %act_max =  "tf.Const"() {value = dense<6.0> : tensor<f32>} : () -> tensor<f32>
    %i8_act_max_0 = "tf.PartitionedCall"(%act_max, %out_scale, %out_zp) {
        config = "", config_proto = "", executor_type = "", f=@quantize_i8
      } : (tensor<f32>, tensor<*xf32>, tensor<*xi32>) -> tensor<*xi8>
    %i8_act_max_1 = "tf.Cast"(%i8_act_max_0) {Truncate = false} : (tensor<*xi8>) -> tensor<*xf32>
    %float_out_zp = "tf.Cast"(%out_zp) {Truncate = false} : (tensor<*xi32>) -> tensor<*xf32>
    %clip_min = "tf.Maximum"(%i8_min, %float_out_zp) : (tensor<f32>, tensor<*xf32>) -> tensor<*xf32>
    %clip_max = "tf.Minimum"(%i8_max, %i8_act_max_1) : (tensor<f32>, tensor<*xf32>) -> tensor<*xf32>

    %clamp_max = "tf.Maximum"(%rescale, %clip_min) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
    %clamp_min = "tf.Minimum"(%clamp_max, %clip_max) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
    %round = "tf.Round"(%clamp_min) : (tensor<*xf32>) -> tensor<*xf32>
    %cast = "tf.Cast"(%round) {Truncate = false} : (tensor<*xf32>) -> tensor<*xi8>
    func.return %cast : tensor<*xi8>
  }

  // Dequantizes and clips to the range of quantized type if there is no specific activation.
  func.func private @internal_dequantize_no_activation_fn(%accumulation : tensor<*xi32>,
                         %input_scale : tensor<*xf32>, %input_zp : tensor<*xi32>,
                         %filter_scale : tensor<*xf32>, %filter_zp : tensor<*xi32>,
                         %out_scale : tensor<*xf32>, %out_zp : tensor<*xi32>) -> tensor<*xf32> {
    %accumulation_scale = "tf.Mul"(%input_scale, %filter_scale) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
    %cast = "tf.Cast"(%accumulation) {Truncate = false} : (tensor<*xi32>) -> tensor<*xf32>
    %dequantize = "tf.Mul"(%cast, %accumulation_scale) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>

    %i8_min = "tf.Const"() {value = dense<-128> : tensor<i8>} : () -> tensor<i8>
    %i8_max = "tf.Const"() {value = dense<127> : tensor<i8>} : () -> tensor<i8>

    %clip_min = "tf.PartitionedCall"(%i8_min, %out_scale, %out_zp) {
        config = "", config_proto = "", executor_type = "", f=@internal_dequantize_i8_fn
      } : (tensor<i8>, tensor<*xf32>, tensor<*xi32>) -> tensor<*xf32>
    %clip_max = "tf.PartitionedCall"(%i8_max, %out_scale, %out_zp) {
        config = "", config_proto = "", executor_type = "", f=@internal_dequantize_i8_fn
      } : (tensor<i8>, tensor<*xf32>, tensor<*xi32>) -> tensor<*xf32>

    %clamp_max = "tf.Maximum"(%dequantize, %clip_min) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
    %clamp_min = "tf.Minimum"(%clamp_max, %clip_max) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>

    func.return %clamp_min : tensor<*xf32>
  }

  // Dequantizes and applies quantized Relu by clipping.
  func.func private @internal_dequantize_and_relu_fn(%accumulation : tensor<*xi32>,
                         %input_scale : tensor<*xf32>, %input_zp : tensor<*xi32>,
                         %filter_scale : tensor<*xf32>, %filter_zp : tensor<*xi32>,
                         %out_scale : tensor<*xf32>, %out_zp : tensor<*xi32>) -> tensor<*xf32> {
    %accumulation_scale = "tf.Mul"(%input_scale, %filter_scale) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
    %cast = "tf.Cast"(%accumulation) {Truncate = false} : (tensor<*xi32>) -> tensor<*xf32>
    %dequantize = "tf.Mul"(%cast, %accumulation_scale) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>

    %i8_min = "tf.Const"() {value = dense<-128> : tensor<i8>} : () -> tensor<i8>
    %i8_max = "tf.Const"() {value = dense<127> : tensor<i8>} : () -> tensor<i8>

    %clip_min_0 = "tf.PartitionedCall"(%i8_min, %out_scale, %out_zp) {
        config = "", config_proto = "", executor_type = "", f=@internal_dequantize_i8_fn
      } : (tensor<i8>, tensor<*xf32>, tensor<*xi32>) -> tensor<*xf32>
    %clip_max = "tf.PartitionedCall"(%i8_max, %out_scale, %out_zp) {
        config = "", config_proto = "", executor_type = "", f=@internal_dequantize_i8_fn
      } : (tensor<i8>, tensor<*xf32>, tensor<*xi32>) -> tensor<*xf32>

    %clip_min = "tf.Relu"(%clip_min_0) : (tensor<*xf32>) -> tensor<*xf32>

    %clamp_max = "tf.Maximum"(%dequantize, %clip_min) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
    %clamp_min = "tf.Minimum"(%clamp_max, %clip_max) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>

    func.return %clamp_min : tensor<*xf32>
  }

  // Dequantizes and applies quantized Relu6 by clipping.
  func.func private @internal_dequantize_and_relu6_fn(%accumulation : tensor<*xi32>,
                         %input_scale : tensor<*xf32>, %input_zp : tensor<*xi32>,
                         %filter_scale : tensor<*xf32>, %filter_zp : tensor<*xi32>,
                         %out_scale : tensor<*xf32>, %out_zp : tensor<*xi32>) -> tensor<*xf32> {
    %accumulation_scale = "tf.Mul"(%input_scale, %filter_scale) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
    %cast = "tf.Cast"(%accumulation) {Truncate = false} : (tensor<*xi32>) -> tensor<*xf32>
    %dequantize = "tf.Mul"(%cast, %accumulation_scale) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>

    %i8_min = "tf.Const"() {value = dense<-128> : tensor<i8>} : () -> tensor<i8>
    %i8_max = "tf.Const"() {value = dense<127> : tensor<i8>} : () -> tensor<i8>
    %relu6_upper = "tf.Const"() {value = dense<6.0>: tensor<f32>} : () -> tensor<f32>

    %clip_min_0 = "tf.PartitionedCall"(%i8_min, %out_scale, %out_zp) {
        config = "", config_proto = "", executor_type = "", f=@internal_dequantize_i8_fn
      } : (tensor<i8>, tensor<*xf32>, tensor<*xi32>) -> tensor<*xf32>
    %clip_max_0 = "tf.PartitionedCall"(%i8_max, %out_scale, %out_zp) {
        config = "", config_proto = "", executor_type = "", f=@internal_dequantize_i8_fn
      } : (tensor<i8>, tensor<*xf32>, tensor<*xi32>) -> tensor<*xf32>

    %clip_min = "tf.Relu"(%clip_min_0) : (tensor<*xf32>) -> tensor<*xf32>
    %clip_max = "tf.Minimum"(%clip_max_0, %relu6_upper) : (tensor<*xf32>, tensor<f32>) -> tensor<*xf32>

    %clamp_max = "tf.Maximum"(%dequantize, %clip_min) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
    %clamp_min = "tf.Minimum"(%clamp_max, %clip_max) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>

    func.return %clamp_min : tensor<*xf32>
  }

  // Conv2D with int32 accumulation.
  func.func private @internal_conv2d_fn(
                         %input : tensor<*xi8>, %filter : tensor<*xi8>,
                         %input_scale : tensor<*xf32>, %input_zp : tensor<*xi32>,
                         %filter_scale : tensor<*xf32>, %filter_zp : tensor<*xi32>) -> tensor<*xi32> {
    %0 = "tf.Cast"(%input) {Truncate = false} : (tensor<*xi8>) -> tensor<*xi32>
    %1 = "tf.Sub"(%0, %input_zp) : (tensor<*xi32>, tensor<*xi32>) -> tensor<*xi32>

    // Use identity op to avoid the filter being constant-folded.
    %identity = "tf.Identity"(%filter) : (tensor<*xi8>) -> tensor<*xi8>
    %2 = "tf.Cast"(%identity) {Truncate = false} : (tensor<*xi8>) -> tensor<*xi32>
    %3 = "tf.Sub"(%2, %filter_zp) : (tensor<*xi32>, tensor<*xi32>) -> tensor<*xi32>

    %5 = "tf.Conv2D"(%1, %3) {
      padding = "VALID", strides = [1, 1, 1, 1],
      attr_map = "strides:0,use_cudnn_on_gpu:1,padding:2,explicit_paddings:3,dilations:4"
    } : (tensor<*xi32>, tensor<*xi32>) -> tensor<*xi32>
    func.return %5 : tensor<*xi32>
  }

  // DepthwiseConv2D with (simulated) int32 accumulation.
  func.func private @internal_depthwise_conv2d_fn(
                         %input : tensor<*xi8>, %filter : tensor<*xi8>,
                         %input_scale : tensor<*xf32>, %input_zp : tensor<*xi32>,
                         %filter_scale : tensor<*xf32>, %filter_zp : tensor<*xi32>) -> tensor<*xi32> {
    %0 = "tf.Cast"(%input) {Truncate = false} : (tensor<*xi8>) -> tensor<*xi32>
    %1 = "tf.Sub"(%0, %input_zp) : (tensor<*xi32>, tensor<*xi32>) -> tensor<*xi32>

    // Use identity op to avoid the filter being constant-folded.
    %identity = "tf.Identity"(%filter) : (tensor<*xi8>) -> tensor<*xi8>
    %2 = "tf.Cast"(%identity) {Truncate = false} : (tensor<*xi8>) -> tensor<*xi32>
    %3 = "tf.Sub"(%2, %filter_zp) : (tensor<*xi32>, tensor<*xi32>) -> tensor<*xi32>

    %cast_1_f32 = "tf.Cast"(%1) {Truncate = false} : (tensor<*xi32>) -> tensor<*xf32>
    %cast_3_f32 = "tf.Cast"(%3) {Truncate = false} : (tensor<*xi32>) -> tensor<*xf32>

    %5 = "tf.DepthwiseConv2dNative"(%cast_1_f32, %cast_3_f32) {
      padding = "VALID", strides = [1, 1, 1, 1],
      attr_map = "strides:0,padding:1,explicit_paddings:2,dilations:3"
    } : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
    %6 = "tf.Cast"(%5) : (tensor<*xf32>) -> tensor<*xi32>
    func.return %6 : tensor<*xi32>
  }

  // Matmul with int32 accumulation.
  func.func private @internal_matmul_fn(
                         %input : tensor<*xi8>, %weight : tensor<*xi8>,
                         %input_scale : tensor<*xf32>, %input_zp : tensor<*xi32>,
                         %weight_scale : tensor<*xf32>, %weight_zp : tensor<*xi32>) -> tensor<*xi32> {
    %0 = "tf.Cast"(%input) {Truncate = false} : (tensor<*xi8>) -> tensor<*xi32>
    %1 = "tf.Sub"(%0, %input_zp) : (tensor<*xi32>, tensor<*xi32>) -> tensor<*xi32>

    // Use identity op to avoid the weight being constant-folded.
    %identity = "tf.Identity"(%weight) : (tensor<*xi8>) -> tensor<*xi8>
    %2 = "tf.Cast"(%identity) {Truncate = false} : (tensor<*xi8>) -> tensor<*xi32>
    %3 = "tf.Sub"(%2, %weight_zp) : (tensor<*xi32>, tensor<*xi32>) -> tensor<*xi32>

    %5 = "tf.MatMul"(%1, %3) {
      attr_map = "transpose_a:0,transpose_b:1"
    } : (tensor<*xi32>, tensor<*xi32>) -> tensor<*xi32>
    func.return %5 : tensor<*xi32>
  }

  // Conv3D with int32 accumulation.
  func.func private @internal_conv3d_fn(
                         %input : tensor<*xi8>, %filter : tensor<*xi8>,
                         %input_scale : tensor<*xf32>, %input_zp : tensor<*xi32>,
                         %filter_scale : tensor<*xf32>, %filter_zp : tensor<*xi32>) -> tensor<*xi32> {
    %0 = "tf.Cast"(%input) {Truncate = false} : (tensor<*xi8>) -> tensor<*xi32>
    %1 = "tf.Sub"(%0, %input_zp) : (tensor<*xi32>, tensor<*xi32>) -> tensor<*xi32>

    // Use identity op to avoid the filter being constant-folded.
    %identity = "tf.Identity"(%filter) : (tensor<*xi8>) -> tensor<*xi8>
    %2 = "tf.Cast"(%identity) {Truncate = false} : (tensor<*xi8>) -> tensor<*xi32>
    %3 = "tf.Sub"(%2, %filter_zp) : (tensor<*xi32>, tensor<*xi32>) -> tensor<*xi32>

    %cast_1_f32 = "tf.Cast"(%1) {Truncate = false} : (tensor<*xi32>) -> tensor<*xf32>
    %cast_3_f32 = "tf.Cast"(%3) {Truncate = false} : (tensor<*xi32>) -> tensor<*xf32>

    %5 = "tf.Conv3D"(%cast_1_f32, %cast_3_f32) {
      padding = "VALID", strides = [1, 1, 1, 1, 1],
      attr_map = "strides:0,padding:1,dilations:2"
    } : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
    %6 = "tf.Cast"(%5) : (tensor<*xf32>) -> tensor<*xi32>
    func.return %6 : tensor<*xi32>
  }

  // BatchMatMul with int32 accumulation.
  func.func private @internal_batch_matmul_fn(
                         %input : tensor<*xi8>, %weight : tensor<*xi8>,
                         %input_scale : tensor<*xf32>, %input_zp : tensor<*xi32>,
                         %weight_scale : tensor<*xf32>, %weight_zp : tensor<*xi32>) -> tensor<*xi32> {
    %0 = "tf.Cast"(%input) {Truncate = false} : (tensor<*xi8>) -> tensor<*xi32>
    %1 = "tf.Sub"(%0, %input_zp) : (tensor<*xi32>, tensor<*xi32>) -> tensor<*xi32>

    // Use identity op to avoid the weight being constant-folded.
    %identity = "tf.Identity"(%weight) : (tensor<*xi8>) -> tensor<*xi8>
    %2 = "tf.Cast"(%identity) {Truncate = false} : (tensor<*xi8>) -> tensor<*xi32>
    %3 = "tf.Sub"(%2, %weight_zp) : (tensor<*xi32>, tensor<*xi32>) -> tensor<*xi32>

    %5 = "tf.BatchMatMulV2"(%1, %3) {
      attr_map = "adj_x:0,adj_y:1"
    } : (tensor<*xi32>, tensor<*xi32>) -> tensor<*xi32>
    func.return %5 : tensor<*xi32>
  }

   // Einsum with int32 accumulation.
  func.func private @internal_einsum_fn(
                         %input : tensor<*xi8>, %weight : tensor<*xi8>,
                         %input_scale : tensor<*xf32>, %input_zp : tensor<*xi32>,
                         %weight_scale : tensor<*xf32>, %weight_zp : tensor<*xi32>) -> tensor<*xi32> {
    // For Einsum, the input could also be constant, since we do the argument swapping.
    %identity_input = "tf.Identity"(%input) : (tensor<*xi8>) -> tensor<*xi8>
    %0 = "tf.Cast"(%identity_input) {Truncate = false} : (tensor<*xi8>) -> tensor<*xi32>
    %1 = "tf.Sub"(%0, %input_zp) : (tensor<*xi32>, tensor<*xi32>) -> tensor<*xi32>

    // Use identity op to avoid the weight being constant-folded.
    %identity = "tf.Identity"(%weight) : (tensor<*xi8>) -> tensor<*xi8>
    %2 = "tf.Cast"(%identity) {Truncate = false} : (tensor<*xi8>) -> tensor<*xi32>
    %3 = "tf.Sub"(%2, %weight_zp) : (tensor<*xi32>, tensor<*xi32>) -> tensor<*xi32>

    %4 = "tf.Einsum"(%1, %3) {
      // Equation placeholder to prevent error during op creation.
      equation = "",
      attr_map = "equation:0"
    } : (tensor<*xi32>, tensor<*xi32>) -> tensor<*xi32>

    func.return %4 : tensor<*xi32>
  }

  for main_op in ["Conv2D", "DepthwiseConv2D", "MatMul", "Conv3D", "BatchMatMul", "Einsum"] {
    parameters[
      {"quantized_ops": ["${main_op}", "BiasAdd"], "act_func": "internal_requantize_no_activation_fn", "output_type": "i8"},
      {"quantized_ops": ["${main_op}", "BiasAdd", "Relu"], "act_func": "internal_requantize_and_relu_fn", "output_type": "i8"},
      {"quantized_ops": ["${main_op}", "BiasAdd", "Relu6"], "act_func": "internal_requantize_and_relu6_fn", "output_type": "i8"},
      {"quantized_ops": ["${main_op}", "BiasAdd"], "act_func": "internal_dequantize_no_activation_fn", "output_type": "f32"},
      {"quantized_ops": ["${main_op}", "BiasAdd", "Relu"], "act_func": "internal_dequantize_and_relu_fn", "output_type": "f32"},
      {"quantized_ops": ["${main_op}", "BiasAdd", "Relu6"], "act_func": "internal_dequantize_and_relu6_fn", "output_type": "f32"},
    ]
    func.func @GenerateQuantizedFunctionName(${quantized_ops}, "${output_type}")(%input : tensor<*xi8>,
                           %filter : tensor<*xi8>, %bias : tensor<*xi32>,
                           %input_scale : tensor<*xf32>, %input_zp : tensor<*xi32>,
                           %filter_scale : tensor<*xf32>, %filter_zp : tensor<*xi32>,
                           %bias_scale : tensor<*xf32>, %bias_zp : tensor<*xi32>,
                           %out_scale : tensor<*xf32>, %out_zp : tensor<*xi32>) -> tensor<*x${output_type}>
        attributes {tf_quant.quantized_ops = ${quantized_ops}} {
      %0 = "tf.PartitionedCall"(%input, %filter, %input_scale, %input_zp,
                                  %filter_scale, %filter_zp) {
          config = "", config_proto = "", executor_type = "", f=@GenerateImplFunctionName(${main_op})
        } : (tensor<*xi8>, tensor<*xi8>, tensor<*xf32>, tensor<*xi32>,
               tensor<*xf32>, tensor<*xi32>) -> tensor<*xi32>
      %1 = "tf.AddV2"(%0, %bias) : (tensor<*xi32>, tensor<*xi32>) -> tensor<*xi32>
      %2 = "tf.PartitionedCall"(%1, %input_scale, %input_zp, %filter_scale, %filter_zp,
                                  %out_scale, %out_zp) {
          config = "", config_proto = "", executor_type = "", f=@${act_func}
        } : (tensor<*xi32>, tensor<*xf32>, tensor<*xi32>, tensor<*xf32>, tensor<*xi32>,
               tensor<*xf32>, tensor<*xi32>) -> tensor<*x${output_type}>
      func.return %2 : tensor<*x${output_type}>
    }

    parameters[
      {"quantized_ops": ["${main_op}"], "act_func": "internal_requantize_no_activation_fn", "output_type": "i8"},
      {"quantized_ops": ["${main_op}", "Relu"], "act_func": "internal_requantize_and_relu_fn", "output_type": "i8"},
      {"quantized_ops": ["${main_op}", "Relu6"], "act_func": "internal_requantize_and_relu6_fn", "output_type": "i8"},
      {"quantized_ops": ["${main_op}"], "act_func": "internal_dequantize_no_activation_fn", "output_type": "f32"},
      {"quantized_ops": ["${main_op}", "Relu"], "act_func": "internal_dequantize_and_relu_fn", "output_type": "f32"},
      {"quantized_ops": ["${main_op}", "Relu6"], "act_func": "internal_dequantize_and_relu6_fn", "output_type": "f32"},
    ]
    func.func @GenerateQuantizedFunctionName(${quantized_ops}, "${output_type}")(
                           %input : tensor<*xi8>, %filter : tensor<*xi8>,
                           %input_scale : tensor<*xf32>, %input_zp : tensor<*xi32>,
                           %filter_scale : tensor<*xf32>, %filter_zp : tensor<*xi32>,
                           %out_scale : tensor<*xf32>, %out_zp : tensor<*xi32>) -> tensor<*x${output_type}>
        attributes {tf_quant.quantized_ops = ${quantized_ops}} {
      %0 = "tf.PartitionedCall"(%input, %filter, %input_scale, %input_zp,
                                  %filter_scale, %filter_zp) {
          config = "", config_proto = "", executor_type = "", f=@GenerateImplFunctionName(${main_op})
        } : (tensor<*xi8>, tensor<*xi8>, tensor<*xf32>, tensor<*xi32>,
               tensor<*xf32>, tensor<*xi32>) -> tensor<*xi32>
      %1 = "tf.PartitionedCall"(%0, %input_scale, %input_zp, %filter_scale, %filter_zp,
                                  %out_scale, %out_zp) {
          config = "", config_proto = "", executor_type = "", f=@${act_func}
        } : (tensor<*xi32>, tensor<*xf32>, tensor<*xi32>, tensor<*xf32>, tensor<*xi32>,
               tensor<*xf32>, tensor<*xi32>) -> tensor<*x${output_type}>
      func.return %1 : tensor<*x${output_type}>
    }
  } // end for

  // TODO(b/278493977): Create generic implementation of lifting any fused op
  // with any reshaping op
  for main_op in ["MatMul"] {
    parameters[
      {"quantized_ops": ["${main_op}", "Reshape", "BiasAdd"], "act_func": "internal_requantize_no_activation_fn", "output_type": "i8"},
      {"quantized_ops": ["${main_op}", "Reshape", "BiasAdd"], "act_func": "internal_dequantize_no_activation_fn", "output_type": "f32"},
    ]
    func.func @GenerateQuantizedFunctionName(${quantized_ops}, "${output_type}")(%input : tensor<*xi8>,
                           %filter : tensor<*xi8>, %bias : tensor<*xi32>, %shape : tensor<*xi32>,
                           %input_scale : tensor<*xf32>, %input_zp : tensor<*xi32>,
                           %filter_scale : tensor<*xf32>, %filter_zp : tensor<*xi32>,
                           %bias_scale : tensor<*xf32>, %bias_zp : tensor<*xi32>,
                           %out_scale : tensor<*xf32>, %out_zp : tensor<*xi32>) -> tensor<*x${output_type}>
        attributes {tf_quant.quantized_ops = ${quantized_ops}} {
      %0 = "tf.PartitionedCall"(%input, %filter, %input_scale, %input_zp,
                                  %filter_scale, %filter_zp) {
          config = "", config_proto = "", executor_type = "", f=@GenerateImplFunctionName(${main_op})
        } : (tensor<*xi8>, tensor<*xi8>, tensor<*xf32>, tensor<*xi32>,
               tensor<*xf32>, tensor<*xi32>) -> tensor<*xi32>
      %1 = "tf.Reshape"(%0, %shape) : (tensor<*xi32>, tensor<*xi32>) -> tensor<*xi32>
      %2 = "tf.AddV2"(%1, %bias) : (tensor<*xi32>, tensor<*xi32>) -> tensor<*xi32>
      %3 = "tf.PartitionedCall"(%2, %input_scale, %input_zp, %filter_scale, %filter_zp,
                                  %out_scale, %out_zp) {
          config = "", config_proto = "", executor_type = "", f=@${act_func}
        } : (tensor<*xi32>, tensor<*xf32>, tensor<*xi32>, tensor<*xf32>, tensor<*xi32>,
               tensor<*xf32>, tensor<*xi32>) -> tensor<*x${output_type}>
      func.return %3 : tensor<*x${output_type}>
    }
  } // end for

  func.func @quantize_i8(%input : tensor<*xf32>, %scale : tensor<*xf32>, %zp : tensor<*xi32>) -> tensor<*xi8> {
    %float_zp = "tf.Cast"(%zp) {Truncate = false} : (tensor<*xi32>) -> tensor<*xf32>
    %div = "tf.Div"(%input, %scale) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
    %add = "tf.AddV2"(%div, %float_zp) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>

    %i8_min = "tf.Const"() {value = dense<-128.0> : tensor<f32>} : () -> tensor<f32>
    %i8_max = "tf.Const"() {value = dense<127.0> : tensor<f32>} : () -> tensor<f32>
    %clamp_max = "tf.Maximum"(%add, %i8_min) : (tensor<*xf32>, tensor<f32>) -> tensor<*xf32>
    %clamp_min = "tf.Minimum"(%clamp_max, %i8_max) : (tensor<*xf32>, tensor<f32>) -> tensor<*xf32>
    %round = "tf.Round"(%clamp_min) : (tensor<*xf32>) -> tensor<*xf32>
    %i8 = "tf.Cast"(%round) : (tensor<*xf32>) -> tensor<*xi8>
    func.return %i8 : tensor<*xi8>
  }

  func.func @dequantize_i8(%input : tensor<*xi8>, %scale : tensor<*xf32>, %zp : tensor<*xi32>) -> tensor<*xf32> {
    // Use identity op to avoid the weight being constant-folded.
    %identity = "tf.Identity"(%input) : (tensor<*xi8>) -> tensor<*xi8>
    %input_i32 = "tf.Cast"(%identity) : (tensor<*xi8>) -> tensor<*xi32>
    %output = "tf.Sub"(%input_i32, %zp) : (tensor<*xi32>, tensor<*xi32>) -> tensor<*xi32>
    %cast = "tf.Cast"(%output) : (tensor<*xi32>) -> tensor<*xf32>
    %mul = "tf.Mul"(%cast, %scale) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
    func.return %mul : tensor<*xf32>
  }

  //===----------------------------------------------------------------------===//
  // Weight-only functions.
  //===----------------------------------------------------------------------===//

  func.func private @internal_dequantize_i8_in_f32_fn(
                           %input : tensor<*xi8>, %weight_scale : tensor<*xf32>) -> tensor<*xf32> {
    %input_f32 = "tf.Cast"(%input) : (tensor<*xi8>) -> tensor<*xf32>
    %mul = "tf.Mul"(%input_f32, %weight_scale) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
    func.return %mul : tensor<*xf32>
  }

  // Note that input i64 type is also supported by this.
  // As the output is quantized type, output scale/zp is required for the arguments.
  parameters[
    {"quantized_ops": ["Gather"], "act_func": "internal_identity_fn", "output_type": "i8"}
  ]
  func.func @GenerateQuantizedFunctionName(${quantized_ops}, "${output_type}")(
                         %weight : tensor<*xi8>, %input : tensor<*xi32>, %axis : tensor<i32>,
                         %weight_scale : tensor<*xf32>, %weight_zp : tensor<*xi32>,
                         %out_scale : tensor<*xf32>, %out_zp : tensor<*xi32>) -> tensor<*x${output_type}>
      attributes {tf_quant.quantized_ops = ${quantized_ops}} {

    %out = "tf.GatherV2"(%weight, %input, %axis) {
      batch_dims = 0 : i64, attr_map = "batch_dims:0"} : (tensor<*xi8>, tensor<*xi32>, tensor<i32>) -> tensor<*xi8>

    func.return %out : tensor<*x${output_type}>
  }

  // Note that input i64 type is also supported by this.
  // The dequantization is merged to the quantized function.
  // As the output type is specified to f32, the quantized function has "_float_output_fn" tag at the end.
  parameters[
    {"quantized_ops": ["Gather"], "act_func": "internal_dequantize_i8_in_f32_fn", "output_type": "f32"}
  ]
  func.func @GenerateQuantizedFunctionName(${quantized_ops}, "${output_type}")(
                         %weight : tensor<*xi8>, %input : tensor<*xi32>, %axis : tensor<i32>,
                         %weight_scale : tensor<*xf32>, %weight_zp : tensor<*xi32>) -> tensor<*x${output_type}>
      attributes {tf_quant.quantized_ops = ${quantized_ops}} {

    %accum_out = "tf.GatherV2"(%weight, %input, %axis) {
      batch_dims = 0 : i64, attr_map = "batch_dims:0"} : (tensor<*xi8>, tensor<*xi32>, tensor<i32>) -> tensor<*xi8>

    %out = "tf.PartitionedCall"(%accum_out, %weight_scale) {
        config = "", config_proto = "", executor_type = "", f=@${act_func}
      } : (tensor<*xi8>, tensor<*xf32>) -> tensor<*x${output_type}>

    func.return %out : tensor<*x${output_type}>
  }

}
