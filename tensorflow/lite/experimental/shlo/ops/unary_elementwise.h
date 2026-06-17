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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_SHLO_OPS_UNARY_ELEMENTWISE_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_SHLO_OPS_UNARY_ELEMENTWISE_H_

#include <cstddef>

#include "absl/algorithm/container.h"
#include "absl/status/status.h"
#include "absl/types/span.h"
#include "tensorflow/lite/experimental/shlo/data_type.h"
#include "tensorflow/lite/experimental/shlo/dispatch.h"
#include "tensorflow/lite/experimental/shlo/ops/util.h"
#include "tensorflow/lite/experimental/shlo/quantize.h"
#include "tensorflow/lite/experimental/shlo/quantized_tensor_element_type.h"
#include "tensorflow/lite/experimental/shlo/shape.h"
#include "tensorflow/lite/experimental/shlo/tensor.h"

namespace shlo_ref {

namespace detail {

template <typename StorageT, typename ExpressedT, typename F>
void DequantizeOpQuantizePerAxisImpl(
    F& op, const Shape& shape, const Axis quantization_dimension,
    const StorageT quantization_min, const StorageT quantization_max,
    const absl::Span<const StorageT> input_zero_points,
    const absl::Span<const ExpressedT> input_scales,
    const absl::Span<const StorageT> output_zero_points,
    const absl::Span<const ExpressedT> output_scales, const Strides& strides,
    const StorageT* input_data, StorageT* output_data, const size_t depth,
    size_t quantization_index) {
  const DimensionSize dim = shape.Dim(depth);
  if (depth + 1 >= shape.Rank()) {
    for (DimensionSize i = 0; i < dim; ++i) {
      if (depth == quantization_dimension) {
        quantization_index = i;
      }
      const ExpressedT dequantized_input =
          Dequantize(*input_data, input_zero_points[quantization_index],
                     input_scales[quantization_index]);
      const ExpressedT dequantized_res = op(dequantized_input);
      *output_data = Quantize<StorageT, ExpressedT>(
          dequantized_res, output_zero_points[quantization_index],
          static_cast<ExpressedT>(1) / output_scales[quantization_index],
          quantization_min, quantization_max);
      output_data += strides[depth];
      input_data += strides[depth];
    }
  } else {
    for (DimensionSize i = 0; i < dim; ++i) {
      if (depth == quantization_dimension) {
        quantization_index = i;
      }
      DequantizeOpQuantizePerAxisImpl(
          op, shape, quantization_dimension, quantization_min, quantization_max,
          input_zero_points, input_scales, output_zero_points, output_scales,
          strides, input_data, output_data, depth + 1, quantization_index);
      output_data += strides[depth];
      input_data += strides[depth];
    }
  }
}

template <DataType storage_type, DataType expressed_type, typename F>
void DequantizeOpQuantizePerAxis(F&& func, const Tensor& input,
                                 Tensor& output) {
  using StorageT = StorageType<storage_type>;
  using ExpressedT = StorageType<expressed_type>;
  const Shape& shape = input.shape();
  const Axis quantization_dimension =
      input.quantized_per_axis_element_type().QuantizedDimension();
  const absl::Span<const StorageT> input_zero_points =
      input.quantized_per_axis_element_type().ZeroPointsAs<storage_type>();
  const absl::Span<const ExpressedT> input_scales =
      input.quantized_per_axis_element_type().ScalesAs<expressed_type>();
  const absl::Span<const StorageT> output_zero_points =
      output.quantized_per_axis_element_type().ZeroPointsAs<storage_type>();
  const absl::Span<const ExpressedT> output_scales =
      output.quantized_per_axis_element_type().ScalesAs<expressed_type>();
  const Strides& strides = ComputeStrides(shape);
  const StorageT* input_data = input.GetDataAs<storage_type>();
  StorageT* output_data = output.GetDataAs<storage_type>();
  DequantizeOpQuantizePerAxisImpl(
      func, shape, quantization_dimension, Storage<storage_type>::kMinValue,
      Storage<storage_type>::kMaxValue, input_zero_points, input_scales,
      output_zero_points, output_scales, strides, input_data, output_data,
      /*depth=*/0, /*quantization_index=*/0);
}

template <DataType storage_type, DataType expressed_type, typename F>
void DequantizeOpQuantizePerTensor(F& func, const Tensor& input,
                                   Tensor& output) {
  using StorageT = StorageType<storage_type>;
  using ExpressedT = StorageType<expressed_type>;
  const DimensionSize num_elements = input.NumElements();
  const StorageT input_zero_point =
      input.quantized_per_tensor_element_type().ZeroPointAs<storage_type>();
  const ExpressedT input_scale =
      input.quantized_per_tensor_element_type().ScaleAs<expressed_type>();
  const StorageT output_zero_point =
      output.quantized_per_tensor_element_type().ZeroPointAs<storage_type>();
  const ExpressedT output_scale =
      output.quantized_per_tensor_element_type().ScaleAs<expressed_type>();
  const StorageT* input_data = input.GetDataAs<storage_type>();
  StorageT* output_data = output.GetDataAs<storage_type>();
  const ExpressedT inv_scale = static_cast<ExpressedT>(1) / output_scale;
  for (DimensionSize i = 0; i < num_elements;
       ++i, ++input_data, ++output_data) {
    const ExpressedT dequantized_input =
        Dequantize(*input_data, input_zero_point, input_scale);
    const ExpressedT dequantized_res = func(dequantized_input);
    *output_data = Quantize<storage_type, expressed_type>(
        dequantized_res, output_zero_point, inv_scale);
  }
}

template <DataType data_type, class F>
void EvaluateNoQuantization(F&& func, const Tensor& input, Tensor& output) {
  absl::c_transform(input.Flat<data_type>(), output.GetDataAs<data_type>(),
                    static_cast<F&&>(func));
}

}  // namespace detail

// The following structures and functions are examples to implement unary ops.

template <class F>
struct UnaryElementwiseOp {
  struct Attributes {};
  F func;
};

// Creates the op structure and initializes the functor if it has a state.
template <class F>
UnaryElementwiseOp<F> Create(typename UnaryElementwiseOp<F>::Attributes,
                             const F& func) {
  return UnaryElementwiseOp<F>{func};
}

template <class F>
UnaryElementwiseOp<F> Create(typename UnaryElementwiseOp<F>::Attributes,
                             F&& func) {
  return UnaryElementwiseOp<F>{static_cast<F&&>(func)};
}

// Checks the op constraints and propagates the output shape if needed.
template <class F>
absl::Status Prepare(UnaryElementwiseOp<F>& op, const Tensor& input,
                     Tensor& output) {
  return Propagate(input.shape(), output.shape());
}

// Runs the op over the input tensor.
template <class F>
absl::Status Evaluate(UnaryElementwiseOp<F>& op, const Tensor& input,
                      Tensor& output) {
  if (input.IsPerAxisQuantized()) {
    DISPATCH_QUANTIZED(detail::DequantizeOpQuantizePerAxis,
                       input.quantized_per_axis_element_type().StorageType(),
                       input.quantized_per_axis_element_type().ExpressedType(),
                       op.func, input, output);
  } else if (input.IsPerTensorQuantized()) {
    DISPATCH_QUANTIZED(
        detail::DequantizeOpQuantizePerTensor,
        input.quantized_per_tensor_element_type().StorageType(),
        input.quantized_per_tensor_element_type().ExpressedType(), op.func,
        input, output)
  } else {
    DISPATCH_BOOL_INT_FLOAT(detail::EvaluateNoQuantization,
                            input.tensor_element_type(), op.func, input,
                            output);
  }
  return absl::OkStatus();
}

}  // namespace shlo_ref

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_SHLO_OPS_UNARY_ELEMENTWISE_H_
