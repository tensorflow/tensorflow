// Copyright 2023 The TensorFlow Authors
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

  func.func private @internal_dequantize_f32(
                           %input : tensor<*xf32>, %weight_scale : tensor<*xf32>) -> tensor<*xf32> {
    %mul = "tf.Mul"(%input, %weight_scale) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
    func.return %mul : tensor<*xf32>
  }

  func.func private @internal_conv3d_fn(
                         %input : tensor<*xf32>, %filter : tensor<*xi8>) -> tensor<*xf32> {

   // Use identity op to avoid the filter being constant-folded.
    %identity = "tf.Identity"(%filter) : (tensor<*xi8>) -> tensor<*xi8>
    %2 = "tf.Cast"(%identity) {Truncate = false} : (tensor<*xi8>) -> tensor<*xf32>
    %3 = "tf.Conv3D"(%input, %2) {
      padding = "VALID", strides = [1, 1, 1, 1, 1],
      attr_map = "strides:0,padding:1,dilations:2"
    } : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
    func.return %3 : tensor<*xf32>
  }

  func.func private @internal_batch_matmul_fn(
                         %input : tensor<*xf32>, %filter : tensor<*xi8>) -> tensor<*xf32> {

   // Use identity op to avoid the filter being constant-folded.
    %identity = "tf.Identity"(%filter) : (tensor<*xi8>) -> tensor<*xi8>
    %2 = "tf.Cast"(%identity) {Truncate = false} : (tensor<*xi8>) -> tensor<*xf32>
    %3 = "tf.BatchMatMulV2"(%input, %2) {
      attr_map = "adj_x:0,adj_y:1"
    } : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
    func.return %3 : tensor<*xf32>
  }

  // DepthwiseConv2D with float computation
  func.func private @internal_depthwise_conv2d_fn(
                         %input : tensor<*xf32>, %filter : tensor<*xi8>) -> tensor<*xf32> {

   // Use identity op to avoid the filter being constant-folded.
    %identity = "tf.Identity"(%filter) : (tensor<*xi8>) -> tensor<*xi8>
    %2 = "tf.Cast"(%identity) {Truncate = false} : (tensor<*xi8>) -> tensor<*xf32>
    %3 = "tf.DepthwiseConv2dNative"(%input, %2) {
      padding = "VALID", strides = [1, 1, 1, 1],
      attr_map = "strides:0,padding:1,explicit_paddings:2,dilations:3"
    } : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
    func.return %3 : tensor<*xf32>
  }

  func.func private @internal_matmul_fn(
                         %input : tensor<*xf32>, %filter : tensor<*xi8>) -> tensor<*xf32> {

   // Use identity op to avoid the filter being constant-folded.
    %identity = "tf.Identity"(%filter) : (tensor<*xi8>) -> tensor<*xi8>
    %2 = "tf.Cast"(%identity) {Truncate = false} : (tensor<*xi8>) -> tensor<*xf32>
    %3 = "tf.MatMul"(%input, %2) {
      attr_map = "transpose_a:0,transpose_b:1"
    } : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
    func.return %3 : tensor<*xf32>
  }

  func.func private @internal_conv2d_fn(
                         %input : tensor<*xf32>, %filter : tensor<*xi8>) -> tensor<*xf32> {

   // Use identity op to avoid the filter being constant-folded.
    %identity = "tf.Identity"(%filter) : (tensor<*xi8>) -> tensor<*xi8>
    %2 = "tf.Cast"(%identity) {Truncate = false} : (tensor<*xi8>) -> tensor<*xf32>
    %3 = "tf.Conv2D"(%input, %2) {
      padding = "VALID", strides = [1, 1, 1, 1],
      attr_map = "strides:0,use_cudnn_on_gpu:1,padding:2,explicit_paddings:3,dilations:4"
    } : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
    func.return %3 : tensor<*xf32>
  }

  // TODO(b/269387785): Support asymmetric quantization for weight with weight-only scheme.
  parameters[
    {"quantized_ops": ["MatMul"], "internal_func_name": "internal_matmul_fn"},
    {"quantized_ops": ["Conv2D"], "internal_func_name": "internal_conv2d_fn"},
    {"quantized_ops": ["DepthwiseConv2D"], "internal_func_name": "internal_depthwise_conv2d_fn"},
    {"quantized_ops": ["Conv3D"], "internal_func_name": "internal_conv3d_fn"},
    {"quantized_ops": ["BatchMatMul"], "internal_func_name": "internal_batch_matmul_fn"}
  ]
  func.func @GenerateQuantizedFunctionName(${quantized_ops})(
                         %input : tensor<*xf32>, %weight : tensor<*xi8>,
                         %weight_scale : tensor<*xf32>, %weight_zp : tensor<*xi32>) -> tensor<*xf32>
      attributes {tf_quant.quantized_ops = ${quantized_ops}} {

    %accum_out = "tf.PartitionedCall"(%input, %weight) {
        config = "", config_proto = "", executor_type = "", f=@${internal_func_name}
      } : (tensor<*xf32>, tensor<*xi8>) -> tensor<*xf32>

    %out = "tf.PartitionedCall"(%accum_out, %weight_scale) {
        config = "", config_proto = "", executor_type = "", f=@internal_dequantize_f32
      } : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>

    func.return %out : tensor<*xf32>
  }

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
