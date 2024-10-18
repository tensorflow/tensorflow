/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_UTIL_QUANTIZATION_UNIFORM_QUANT_OPS_PARAMS_H_
#define TENSORFLOW_CORE_UTIL_QUANTIZATION_UNIFORM_QUANT_OPS_PARAMS_H_

#include <string>
#include <vector>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/util/quantization/uniform_quant_ops_attr.pb.h"

namespace tensorflow {

// Utility class to load and retrieve params for
// UniformQuantizedConvolution{Hybrid} Op kernels.
//
// NOTE: This class instance is fully loaded and valid, only after (1) One of
// LoadFromAttrs() is called (2) ValidateOrFillParamsAndValidateShape is called.
// Member retrieve and CalculateOutputShape() can be used only after both are
// called.
class UniformQuantizedConvolutionParams {
 public:
  UniformQuantizedConvolutionParams() = default;
  // Only for unit tests.
  UniformQuantizedConvolutionParams(
      const std::vector<int>& window_strides,
      const std::vector<int>& lhs_dilation,
      const std::vector<int>& rhs_dilation,
      const UniformQuantizedConvolutionDimensionNumbersAttr& dimension_numbers,
      int feature_group_count, int batch_group_count,
      const std::string& padding, const std::vector<int>& padding_list = {})
      : window_strides_(window_strides),
        lhs_dilation_(lhs_dilation),
        rhs_dilation_(rhs_dilation),
        dimension_numbers_(dimension_numbers),
        feature_group_count_(feature_group_count),
        batch_group_count_(batch_group_count),
        padding_(padding),
        padding_list_(padding_list) {}

  const std::vector<int>& window_strides() const { return window_strides_; }
  const std::vector<int>& lhs_dilation() const { return lhs_dilation_; }
  const std::vector<int>& rhs_dilation() const { return rhs_dilation_; }
  const UniformQuantizedConvolutionDimensionNumbersAttr& dimension_numbers()
      const {
    return dimension_numbers_;
  }
  int batch_group_count() const { return batch_group_count_; }

  const std::vector<int>& padding_list() const { return padding_list_; }
  int feature_group_count() const { return feature_group_count_; }

  // Load UniformQuantizedConvolutionParams members by reading op attrs.
  absl::Status LoadFromAttrs(const OpKernelConstruction& context);
  absl::Status LoadFromAttrs(const shape_inference::InferenceContext& context);

  // Check if UniformQuantizedConvolutionParams members loaded from Attr are
  // valid regarding the lhs_shape and rhs_shape, and fill param values if
  // required. (Set default of empty optional Attrs, and fill padding_list_ if
  // required.)
  // Then, validate given lhs_shape and rhs_shape.
  //
  // NOTE: This method should be called only after calling one of
  // LoadFromAttrs().
  absl::Status ValidateOrFillParamsAndValidateShape(
      const TensorShape& lhs_shape, const TensorShape& rhs_shape);

  // Calculate output shape using lhs_shape, rhs_shape, and the params.
  //
  // NOTE: this method can be used only after calling both LoadFromAttrs() and
  // ValidateOrFillParamsAndValidateShape().
  // Reference:
  // https://github.com/google/jax/blob/0584c6a1c405b23317deb1596c2c161eb5709c84/jax/_src/lax/convolution.py#L349
  absl::StatusOr<TensorShape> CalculateOutputShape(
      const TensorShape& lhs_shape, const TensorShape& rhs_shape) const;

  // Given the original size of a dimension and a dilation, calculate the
  // resulting size after dilation is applied.
  inline static int64_t DilatedSize(int64_t size, int dilation) {
    return size == 0 ? 0 : size + (dilation - 1) * (size - 1);
  }

 private:
  template <typename ContextT>
  absl::Status LoadFromAttrsInternal(const ContextT& context);
  absl::Status ValidateShape(const TensorShape& lhs_shape,
                             const TensorShape& rhs_shape);
  absl::Status ValidateOrFillPaddingList(const TensorShape& lhs_shape,
                                         const TensorShape& rhs_shape);

  // Params from Attrs.
  std::vector<int> window_strides_;
  std::vector<int> lhs_dilation_;
  std::vector<int> rhs_dilation_;
  UniformQuantizedConvolutionDimensionNumbersAttr dimension_numbers_;
  int feature_group_count_;
  int batch_group_count_;
  std::string padding_;

  // Params derived from Attrs and Inputs.
  std::vector<int> padding_list_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_UTIL_QUANTIZATION_UNIFORM_QUANT_OPS_PARAMS_H_
