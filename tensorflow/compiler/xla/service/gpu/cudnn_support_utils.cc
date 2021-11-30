/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/gpu/cudnn_support_utils.h"

#include <functional>

#include "tensorflow/compiler/xla/primitive_util.h"
#include "tensorflow/compiler/xla/service/gpu/cublas_cudnn.h"
#include "tensorflow/core/platform/status.h"

namespace xla {
namespace gpu {

StatusOr<bool> CudnnSupportsOptimizedIntegerConvolution(
    const se::CudaComputeCapability& compute_capability,
    HloCustomCallInstruction& conv, int vector_size) {
  TF_ASSIGN_OR_RETURN(auto kind, GetCudnnConvKind(&conv));
  const Shape& input_shape = conv.operand(0)->shape();
  const Shape& kernel_shape = conv.operand(1)->shape();
  const Shape& result_shape = conv.shape().tuple_shapes(0);
  const auto& dnums = conv.convolution_dimension_numbers();

  // Only vectorization/padding of 4 or 32 for integers is supported.
  if (vector_size != 4 && vector_size != 32) {
    VLOG(3) << "Unsupported vector size for integer convolution: "
            << vector_size;
    return false;
  }

  // Require cc6.1+ for any vectorized integer convolutions
  // Require cc7.5+ for any IMMA convolutions
  if ((vector_size == 32 && !compute_capability.IsAtLeast(7, 5)) ||
      !compute_capability.IsAtLeast(6, 1)) {
    VLOG(3) << "Compute capability " << compute_capability.ToString()
            << " is not sufficent for int8x" << vector_size
            << " vectorization.";
    return false;
  }

  // kForward and kForwardActivation only
  if (kind != CudnnConvKind::kForward &&
      kind != CudnnConvKind::kForwardActivation) {
    VLOG(3) << "Convolution kind is not forward or foward-activation: "
            << conv.ToString();
    return false;
  }

  // Integer inputs/weights only
  if (!primitive_util::IsIntegralType(input_shape.element_type()) ||
      !primitive_util::IsIntegralType(kernel_shape.element_type())) {
    VLOG(3) << "Convolution does not accept integer inputs/weights: "
            << conv.ToString();
    return false;
  }

  // 2D convolutions only
  if (dnums.input_spatial_dimensions().size() != 2 ||
      dnums.kernel_spatial_dimensions().size() != 2 ||
      dnums.output_spatial_dimensions().size() != 2) {
    VLOG(3) << "Convolution is not 2D: " << conv.ToString();
    return false;
  }

  // Only allow for int8x32 when output is also integer
  if (vector_size == 32 &&
      !primitive_util::IsIntegralType(result_shape.element_type())) {
    VLOG(3) << "int8x32 convolutions only support integer output: "
            << conv.ToString();
    return false;
  }

  // For int8x32 convolution check to see if the input/filter size are
  // consistent with the limitation for cuDNN algo1. Per cuDNN release notes:
  // "In INT8x32 Tensor Core cases, the parameters supported by cuDNN v7.6 are
  // limited to W >= (R-1) * dilationW && H >= (S-1) * dilationH, whereas, in
  // cuDNN v8.0.x, W == (R-1) * dilationW || H == (S-1) * dilationH cases are no
  // longer supported."
  //
  // This check is more strict than necessary for cuDNN v7 (allowed for
  // equality) to avoid checking the version of cuDNN explicitly.
  if (vector_size == 32) {
    int64_t W = input_shape.dimensions(dnums.input_spatial_dimensions()[0]);
    int64_t H = input_shape.dimensions(dnums.input_spatial_dimensions()[1]);
    int64_t R = kernel_shape.dimensions(dnums.kernel_spatial_dimensions()[0]);
    int64_t S = kernel_shape.dimensions(dnums.kernel_spatial_dimensions()[1]);
    const int64_t dilationW = conv.window().dimensions()[0].base_dilation();
    const int64_t dilationH = conv.window().dimensions()[1].base_dilation();
    if ((W <= (R - 1) * dilationW) || (H <= (S - 1) * dilationH)) {
      VLOG(3) << "Conv spatial filter/input dimensions are too small for "
                 "vecotrized int8x32 convolution: "
              << conv.ToString();
      return false;
    }
  }

  // Dilation is not supported with integer convs.
  if (window_util::HasDilation(conv.window())) {
    VLOG(3) << "Vectorized integer convolutions do not support dilation: "
            << conv.ToString();
    return false;
  }

  return true;
}
}  // namespace gpu
}  // namespace xla
