/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_SHLO_OPS_BINARY_ELEMENTWISE_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_SHLO_OPS_BINARY_ELEMENTWISE_H_

#include "tensorflow/lite/experimental/shlo/data_type.h"
#include "tensorflow/lite/experimental/shlo/quantize.h"
#include "tensorflow/lite/experimental/shlo/quantized_tensor_element_type.h"
#include "tensorflow/lite/experimental/shlo/shape.h"
#include "tensorflow/lite/experimental/shlo/tensor.h"

namespace shlo_ref {

namespace detail {

template <DataType storage_type, DataType expressed_type, typename F>
void DequantizeOpQuantizePerTensor(F&& func, const Tensor& lhs,
                                   const Tensor& rhs, Tensor& output) {
  using StorageT = StorageType<storage_type>;
  using ExpressedT = StorageType<expressed_type>;
  const DimensionSize num_elements = lhs.NumElements();
  const StorageT lhs_zero_point =
      lhs.quantized_per_tensor_element_type().ZeroPointAs<storage_type>();
  const ExpressedT lhs_scale =
      lhs.quantized_per_tensor_element_type().ScaleAs<expressed_type>();
  const StorageT rhs_zero_point =
      rhs.quantized_per_tensor_element_type().ZeroPointAs<storage_type>();
  const ExpressedT rhs_scale =
      rhs.quantized_per_tensor_element_type().ScaleAs<expressed_type>();
  const StorageT output_zero_point =
      output.quantized_per_tensor_element_type().ZeroPointAs<storage_type>();
  const ExpressedT output_scale =
      output.quantized_per_tensor_element_type().ScaleAs<expressed_type>();
  const StorageT* lhs_data = lhs.GetDataAs<storage_type>();
  const StorageT* rhs_data = rhs.GetDataAs<storage_type>();
  StorageT* output_data = output.GetDataAs<storage_type>();
  const ExpressedT inv_scale = static_cast<ExpressedT>(1) / output_scale;
  for (DimensionSize i = 0; i < num_elements;
       ++i, ++lhs_data, ++rhs_data, ++output_data) {
    const ExpressedT dequantized_lhs =
        Dequantize(*lhs_data, lhs_zero_point, lhs_scale);
    const ExpressedT dequantized_rhs =
        Dequantize(*rhs_data, rhs_zero_point, rhs_scale);
    const ExpressedT dequantized_res = func(dequantized_lhs, dequantized_rhs);
    *output_data = Quantize<storage_type, expressed_type>(
        dequantized_res, output_zero_point, inv_scale);
  }
}

template <DataType data_type, class F>
void EvaluateNoQuantization(F&& func, const Tensor& lhs, const Tensor& rhs,
                            Tensor& output) {
  using T = StorageType<data_type>;
  const T* lhs_data = lhs.GetDataAs<data_type>();
  const T* rhs_data = rhs.GetDataAs<data_type>();
  T* output_data = output.GetDataAs<data_type>();
  const DimensionSize num_elements = lhs.NumElements();
  for (DimensionSize i = 0; i < num_elements;
       ++i, ++output_data, ++lhs_data, ++rhs_data) {
    *output_data = static_cast<T>(func(*lhs_data, *rhs_data));
  }
}

}  // namespace detail

}  // namespace shlo_ref

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_SHLO_OPS_BINARY_ELEMENTWISE_H_
