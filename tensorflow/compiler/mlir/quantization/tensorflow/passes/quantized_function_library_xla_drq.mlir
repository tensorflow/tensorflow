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

// Quantization as a function library with XLA Ops for Dynamic PTQ
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
    %div = "tf.Div"(%input, %scale) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
    %round = "tf.Round"(%div) : (tensor<*xf32>) -> tensor<*xf32>
    %cast = "tf.Cast"(%round) : (tensor<*xf32>) -> tensor<*xi32>
    %add = "tf.AddV2"(%cast, %zp) : (tensor<*xi32>, tensor<*xi32>) -> tensor<*xi32>
    %i8 = "tf.Cast"(%add) : (tensor<*xi32>) -> tensor<*xi8>
    func.return %i8 : tensor<*xi8>
  }

  func.func private @internal_dequantize_i32(%input : tensor<*xi32>,
                                    %input_scale : tensor<*xf32>,
                                    %weight_scale : tensor<*xf32>) -> tensor<*xf32> {
    %scale_prod = "tf.Mul"(%input_scale, %weight_scale) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
    %0 = "tf.Cast"(%input) {Truncate = false} : (tensor<*xi32>) -> tensor<*xf32>
    %1 = "tf.Mul"(%0, %scale_prod) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
    func.return %1 : tensor<*xf32>
  }

  // Note: following function supports per-tensor, symmetric, none narrow_range.
  func.func private @internal_calculate_quant_params(%input : tensor<*xf32>) -> (tensor<1xf32>, tensor<1xi32>) {
    %zp = "tf.Const"() {value = dense<0> : tensor<1xi32>} : () -> tensor<1xi32>
    %shape = "tf.Const"() {value = dense<[-1]> : tensor<1xi32>} : () -> tensor<1xi32>
    %dim = "tf.Const"() { value = dense<0> : tensor<1xi64> } : () -> tensor<1xi64>

    %input_1d = "tf.Reshape"(%input, %shape) : (tensor<*xf32>, tensor<1xi32>) -> tensor<?xf32>
    %r_max = "tf.Max"(%input_1d, %dim) { keep_dims = true }: (tensor<?xf32>, tensor<1xi64>) -> tensor<1xf32>
    %r_min = "tf.Min"(%input_1d, %dim) { keep_dims = true }: (tensor<?xf32>, tensor<1xi64>) -> tensor<1xf32>
    %r_max_abs = "tf.Abs"(%r_max) : (tensor<1xf32>) -> tensor<1xf32>
    %r_min_abs = "tf.Abs"(%r_min) : (tensor<1xf32>) -> tensor<1xf32>
    %r_abs_max = "tf.Maximum"(%r_max_abs, %r_min_abs) : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
    %r_abs_max_cast = "tf.Cast"(%r_abs_max) : (tensor<1xf32>) -> tensor<1xf64>

    %i8_min = "tf.Const"() {value = dense<-128> : tensor<i32>} : () -> tensor<i32>
    %i8_max = "tf.Const"() {value = dense<127> : tensor<i32>} : () -> tensor<i32>
    %i8_min_cast = "tf.Cast"(%i8_min) : (tensor<i32>) -> tensor<f64>
    %i8_max_cast = "tf.Cast"(%i8_max) : (tensor<i32>) -> tensor<f64>

    %range_nume = "tf.AddV2"(%r_abs_max_cast, %r_abs_max_cast) : (tensor<1xf64>, tensor<1xf64>) -> tensor<1xf64>
    %range_deno = "tf.Sub"(%i8_max_cast, %i8_min_cast) : (tensor<f64>, tensor<f64>) -> tensor<f64>

    %scale_double = "tf.Div"(%range_nume, %range_deno) : (tensor<1xf64>, tensor<f64>) -> tensor<1xf64>
    %scale = "tf.Cast"(%scale_double) : (tensor<1xf64>) -> tensor<1xf32>

    func.return %scale, %zp : tensor<1xf32>, tensor<1xi32>
  }

  // Matmul with int32 accumulation
  func.func private @internal_matmul_fn(
                         %input : tensor<*xi8>, %weight : tensor<*xi8>,
                         %input_scale : tensor<*xf32>, %input_zp : tensor<*xi32>,
                         %weight_scale : tensor<*xf32>, %weight_zp : tensor<*xi32>) -> tensor<*xi32> {
    %0 = "tf.Cast"(%input) {Truncate = false} : (tensor<*xi8>) -> tensor<*xi32>
    %1 = "tf.Sub"(%0, %input_zp) : (tensor<*xi32>, tensor<*xi32>) -> tensor<*xi32>

    %2 = "tf.Cast"(%weight) {Truncate = false} : (tensor<*xi8>) -> tensor<*xi32>
    %3 = "tf.Sub"(%2, %weight_zp) : (tensor<*xi32>, tensor<*xi32>) -> tensor<*xi32>

    // TODO(b/215633216): Optimize this function with the XLA Dot op.
    %5 = "tf.MatMul"(%1, %3) {
      attr_map = "transpose_a:0,transpose_b:1"
    } : (tensor<*xi32>, tensor<*xi32>) -> tensor<*xi32>
    func.return %5 : tensor<*xi32>
  }

  func.func @quantized_matmul_fn(
                         %input : tensor<*xf32>, %weight : tensor<*xi8>,
                         %weight_scale : tensor<*xf32>, %weight_zp : tensor<*xi32>) -> tensor<*xf32> {

    %input_scale, %input_zp = "tf.PartitionedCall"(%input) {
        config = "", config_proto = "", executor_type = "", f=@internal_calculate_quant_params
      } : (tensor<*xf32>) -> (tensor<*xf32>, tensor<*xi32>)

    %quantized_input = "tf.PartitionedCall"(%input, %input_scale, %input_zp) {
        config = "", config_proto = "", executor_type = "", f=@internal_quantize_i8
      } : (tensor<*xf32>, tensor<*xf32>, tensor<*xi32>) -> tensor<*xi8>

    %accum_out = "tf.PartitionedCall"(%quantized_input, %weight, %input_scale, %input_zp,
                                %weight_scale, %weight_zp) {
        config = "", config_proto = "", executor_type = "", f=@internal_matmul_fn
      } : (tensor<*xi8>, tensor<*xi8>, tensor<*xf32>, tensor<*xi32>,
             tensor<*xf32>, tensor<*xi32>) -> tensor<*xi32>

    %out = "tf.PartitionedCall"(%accum_out, %input_scale, %weight_scale) {
        config = "", config_proto = "", executor_type = "", f=@internal_dequantize_i32
      } : (tensor<*xi32>, tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>

    func.return %out : tensor<*xf32>
  }

}
