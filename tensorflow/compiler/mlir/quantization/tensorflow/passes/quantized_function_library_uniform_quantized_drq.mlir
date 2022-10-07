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

// Quantization as a function library with Uniform Quantized Ops for Dynamic
// PTQ
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

  // TODO(b/238600711): Populate attributes for quantized_function_library_uniform_quantized
  func.func @quantized_matmul_fn(
                         %input : tensor<*xf32>, %weight : tensor<*x!tf_type.qint8>,
                         %weight_scale : tensor<*xf32>, %weight_zp : tensor<*xi32>) -> tensor<*xf32> {

    %out = "tf.UniformQuantizedDotHybrid"(%input, %weight,
                                %weight_scale, %weight_zp) {
        Tlhs = "tfdtype$DT_FLOAT",
        Trhs = "tfdtype$DT_QINT8",
        Tout = "tfdtype$DT_FLOAT",
        rhs_quantization_axis = -1,
        rhs_quantization_min_val = -128,
        rhs_quantization_max_val = 127,
        attr_map = "0:Tlhs,1:Trhs,2:Tout,3:rhs_quantization_axis,4:rhs_quantization_min_val,5:rhs_quantization_max_val"
      } : (tensor<*xf32>, tensor<*x!tf_type.qint8>, tensor<*xf32>, tensor<*xi32>) -> tensor<*xf32>

    func.return %out : tensor<*xf32>
  }
}
