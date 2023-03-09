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

// Quantization as a function library with TF Ops for Dynamic PTQ
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

  // Note: following functions won't handle per-channel quantization for now.
  func.func private @internal_quantize_i8(%input : tensor<*xf32>, %scale : tensor<*xf32>, %zp : tensor<*xi32>) -> tensor<*xi8> {
    // Uses tf.floor(x + 0.5) instead of tf.round(x) since tf.round generates
    // a very expensive pattern.
    %round_cst = "tf.Const"() {value = dense<0.5> : tensor<f32>} : () -> tensor<f32>
    %float_zp = "tf.Cast"(%zp) {Truncate = false} : (tensor<*xi32>) -> tensor<*xf32>
    %zp_plus_round_cst = "tf.AddV2"(%float_zp, %round_cst) : (tensor<*xf32>, tensor<f32>) -> tensor<*xf32>

    %div = "tf.Div"(%input, %scale) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
    %add = "tf.AddV2"(%div, %zp_plus_round_cst) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
    %round = "tf.Floor"(%add) : (tensor<*xf32>) -> tensor<*xf32>

    %i8_min = "tf.Const"() {value = dense<-128.0> : tensor<f32>} : () -> tensor<f32>
    %i8_max = "tf.Const"() {value = dense<127.0> : tensor<f32>} : () -> tensor<f32>
    %clip = "tf.ClipByValue"(%round, %i8_min, %i8_max) : (tensor<*xf32>, tensor<f32>, tensor<f32>) -> tensor<*xf32>
    %i8 = "tf.Cast"(%clip) : (tensor<*xf32>) -> tensor<*xi8>
    func.return %i8 : tensor<*xi8>
  }

  func.func private @internal_dequantize_i32(%input : tensor<*xi32>,
                                    %input_scale : tensor<*xf32>,
                                    %weight_scale : tensor<*xf32>) -> tensor<*xf32> {
    %scale_prod = "tf.Mul"(%input_scale, %weight_scale) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>

    %cast = "tf.Cast"(%input) : (tensor<*xi32>) -> tensor<*xf32>
    %mul = "tf.Mul"(%cast, %scale_prod) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
    func.return %mul : tensor<*xf32>
  }

  // TODO(b/263199401): Support quantization options for activation quantization for DRQ
  // Note: following function supports per-tensor, asymmetric, non_narrow_range.
  func.func private @internal_calculate_quant_params(%input : tensor<*xf32>) -> (tensor<1xf32>, tensor<1xi32>) {
    %zero = "tf.Const"() {value = dense<0.0> : tensor<1xf32>} : () -> tensor<1xf32>
    %shape = "tf.Const"() {value = dense<[-1]> : tensor<1xi32>} : () -> tensor<1xi32>
    %dim = "tf.Const"() { value = dense<0> : tensor<1xi64> } : () -> tensor<1xi64>

    // Check and include zero in the range so that zero value can be correctly
    // represented.
    %input_1d = "tf.Reshape"(%input, %shape) : (tensor<*xf32>, tensor<1xi32>) -> tensor<?xf32>
    %r_max_without_zero = "tf.Max"(%input_1d, %dim) { keep_dims = true }: (tensor<?xf32>, tensor<1xi64>) -> tensor<1xf32>
    %r_max = "tf.Maximum"(%zero, %r_max_without_zero) : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>

    %r_min_without_zero = "tf.Min"(%input_1d, %dim) { keep_dims = true }: (tensor<?xf32>, tensor<1xi64>) -> tensor<1xf32>
    %r_min = "tf.Minimum"(%zero, %r_min_without_zero) : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>

    %r_max_f64 = "tf.Cast"(%r_max) : (tensor<1xf32>) -> tensor<1xf64>
    %r_min_f64 = "tf.Cast"(%r_min) : (tensor<1xf32>) -> tensor<1xf64>

    %i8_min = "tf.Const"() {value = dense<-128.0> : tensor<f32>} : () -> tensor<f32>
    %i8_max = "tf.Const"() {value = dense<127.0> : tensor<f32>} : () -> tensor<f32>
    %i8_min_f64 = "tf.Cast"(%i8_min) : (tensor<f32>) -> tensor<f64>
    %i8_max_f64 = "tf.Cast"(%i8_max) : (tensor<f32>) -> tensor<f64>

    %range_nume = "tf.Sub"(%r_max_f64, %r_min_f64) : (tensor<1xf64>, tensor<1xf64>) -> tensor<1xf64>
    %range_deno = "tf.Sub"(%i8_max_f64, %i8_min_f64) : (tensor<f64>, tensor<f64>) -> tensor<f64>

    %scale_f64 = "tf.Div"(%range_nume, %range_deno) : (tensor<1xf64>, tensor<f64>) -> tensor<1xf64>
    %scale = "tf.Cast"(%scale_f64) : (tensor<1xf64>) -> tensor<1xf32>

    // Add comparison with minimum if needed
    %intermediate_val = "tf.Div"(%r_max_f64, %scale_f64) : (tensor<1xf64>, tensor<1xf64>) -> tensor<1xf64>
    %zp_from_max = "tf.Sub"(%i8_max_f64, %intermediate_val) : (tensor<f64>, tensor<1xf64>) -> tensor<1xf64>
    %zp_fp32 = "tf.Cast"(%zp_from_max) : (tensor<1xf64>) -> tensor<1xf32>
    %zp = "tf.Cast"(%zp_fp32) : (tensor<1xf32>) -> tensor<1xi32>

    func.return %scale, %zp : tensor<1xf32>, tensor<1xi32>
  }

  // Matmul with int32 accumulation
  func.func private @internal_matmul_fn(
                         %input : tensor<*xi8>, %filter : tensor<*xi8>,
                         %input_scale : tensor<*xf32>, %input_zp : tensor<*xi32>,
                         %weight_scale : tensor<*xf32>, %weight_zp : tensor<*xi32>) -> tensor<*xi32> {
    %0 = "tf.Cast"(%input) {Truncate = false} : (tensor<*xi8>) -> tensor<*xi32>
    %1 = "tf.Sub"(%0, %input_zp) : (tensor<*xi32>, tensor<*xi32>) -> tensor<*xi32>

    // Use identity op to avoid the weight being constant-folded.
    %identity = "tf.Identity"(%filter) : (tensor<*xi8>) -> tensor<*xi8>
    %2 = "tf.Cast"(%identity) {Truncate = false} : (tensor<*xi8>) -> tensor<*xi32>
    %3 = "tf.Sub"(%2, %weight_zp) : (tensor<*xi32>, tensor<*xi32>) -> tensor<*xi32>

    %5 = "tf.MatMul"(%1, %3) {
      attr_map = "transpose_a:0,transpose_b:1"
    } : (tensor<*xi32>, tensor<*xi32>) -> tensor<*xi32>
    func.return %5 : tensor<*xi32>
  }

  // Conv2D with int32 accumulation
  func.func private @internal_conv2d_fn(
                         %input : tensor<*xi8>, %filter : tensor<*xi8>,
                         %input_scale : tensor<*xf32>, %input_zp : tensor<*xi32>,
                         %filter_scale : tensor<*xf32>, %filter_zp : tensor<*xi32>) -> tensor<*xi32> {
    %0 = "tf.Cast"(%input) {Truncate = false} : (tensor<*xi8>) -> tensor<*xi32>
    %1 = "tf.Sub"(%0, %input_zp) : (tensor<*xi32>, tensor<*xi32>) -> tensor<*xi32>

    // Use identity op to avoid the weight being constant-folded.
    %identity = "tf.Identity"(%filter) : (tensor<*xi8>) -> tensor<*xi8>
    %2 = "tf.Cast"(%identity) {Truncate = false} : (tensor<*xi8>) -> tensor<*xi32>
    %3 = "tf.Sub"(%2, %filter_zp) : (tensor<*xi32>, tensor<*xi32>) -> tensor<*xi32>

    %5 = "tf.Conv2D"(%1, %3) {
      padding = "VALID", strides = [1, 1, 1, 1],
      attr_map = "strides:0,use_cudnn_on_gpu:1,padding:2,explicit_paddings:3,dilations:4"
    } : (tensor<*xi32>, tensor<*xi32>) -> tensor<*xi32>
    func.return %5 : tensor<*xi32>
  }

  // DepthwiseConv2D with float computation
  func.func private @internal_depthwise_conv2d_fn(
                         %input : tensor<*xi8>, %filter : tensor<*xi8>,
                         %input_scale : tensor<*xf32>, %input_zp : tensor<*xi32>,
                         %filter_scale : tensor<*xf32>, %filter_zp : tensor<*xi32>) -> tensor<*xi32> {
    %0 = "tf.Cast"(%input) {Truncate = false} : (tensor<*xi8>) -> tensor<*xi32>
    %1 = "tf.Sub"(%0, %input_zp) : (tensor<*xi32>, tensor<*xi32>) -> tensor<*xi32>

    // Use identity op to avoid the weight being constant-folded.
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
  
  parameters[
    {"quantized_ops": ["MatMul"], "internal_func_name": "internal_matmul_fn"},
    {"quantized_ops": ["Conv2D"], "internal_func_name": "internal_conv2d_fn"},
    {"quantized_ops": ["DepthwiseConv2D"], "internal_func_name": "internal_depthwise_conv2d_fn"}
  ]
  func.func @GenerateQuantizedFunctionName(${quantized_ops})(
                         %input : tensor<*xf32>, %weight : tensor<*xi8>,
                         %weight_scale : tensor<*xf32>, %weight_zp : tensor<*xi32>) -> tensor<*xf32>
      attributes {tf_quant.quantized_ops = ${quantized_ops}} {

    %input_scale, %input_zp = "tf.PartitionedCall"(%input) {
        config = "", config_proto = "", executor_type = "", f=@internal_calculate_quant_params
      } : (tensor<*xf32>) -> (tensor<*xf32>, tensor<*xi32>)

    %quantized_input = "tf.PartitionedCall"(%input, %input_scale, %input_zp) {
        config = "", config_proto = "", executor_type = "", f=@internal_quantize_i8
      } : (tensor<*xf32>, tensor<*xf32>, tensor<*xi32>) -> tensor<*xi8>

    %accum_out = "tf.PartitionedCall"(%quantized_input, %weight, %input_scale, %input_zp,
                                %weight_scale, %weight_zp) {
        config = "", config_proto = "", executor_type = "", f=@${internal_func_name}
      } : (tensor<*xi8>, tensor<*xi8>, tensor<*xf32>, tensor<*xi32>,
             tensor<*xf32>, tensor<*xi32>) -> tensor<*xi32>

    %out = "tf.PartitionedCall"(%accum_out, %input_scale, %weight_scale) {
        config = "", config_proto = "", executor_type = "", f=@internal_dequantize_i32
      } : (tensor<*xi32>, tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>

    func.return %out : tensor<*xf32>
  }

  // For weight-only
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

  func.func private @internal_dequantize_f32(
                           %input : tensor<*xf32>, %weight_scale : tensor<*xf32>) -> tensor<*xf32> {
    %mul = "tf.Mul"(%input, %weight_scale) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
    func.return %mul : tensor<*xf32>
  }

  // Note that input i64 type is also supported by this.
  parameters[
    {"quantized_ops": ["Gather"]}
  ]
  func.func @GenerateQuantizedFunctionName(${quantized_ops})(
                         %weight : tensor<*xi8>, %input : tensor<*xi32>, %axis : tensor<i32>,
                         %weight_scale : tensor<*xf32>, %weight_zp : tensor<*xi32>) -> tensor<*xf32>
      attributes {tf_quant.quantized_ops = ${quantized_ops}}
  {
    %accum_out = "tf.GatherV2"(%weight, %input, %axis) {
      batch_dims = 0 : i64, attr_map = "batch_dims:0"} : (tensor<*xi8>, tensor<*xi32>, tensor<i32>) -> tensor<*xi8>

    %accum_out_new = "tf.Cast"(%accum_out) : (tensor<*xi8>) -> tensor<*xf32>

    %out = "tf.PartitionedCall"(%accum_out_new, %weight_scale) {
        config = "", config_proto = "", executor_type = "", f=@internal_dequantize_f32
      } : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>

    func.return %out : tensor<*xf32>
  }
}
