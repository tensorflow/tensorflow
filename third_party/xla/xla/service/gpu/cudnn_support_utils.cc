/* Copyright 2021 The OpenXLA Authors.

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

#include "xla/service/gpu/cudnn_support_utils.h"

#include <cstdint>
#include <vector>

#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/primitive_util.h"
#include "xla/service/gpu/cublas_cudnn.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_description.h"
#include "xla/util.h"
#include "xla/window_util.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace gpu {

absl::StatusOr<bool> CudnnSupportsOptimizedIntegerConvolution(
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

absl::StatusOr<CudnnReorderTransposeConfig>
CudnnInferTransposeForFilterReordering(
    const Shape& shape, const ConvolutionDimensionNumbers& dimension_numbers) {
  // A normal filter should have four dimensions: [O, I, H, W]
  // An already vectorized filter will have five: [O, I/k, H, W, k]; k=4|32
  if (shape.dimensions_size() != 4 && shape.dimensions_size() != 5) {
    return Internal("Filter shape has unexpected rank.");
  }

  // Get convolution dimension numbers.
  const int64_t dO = dimension_numbers.kernel_output_feature_dimension();
  const int64_t dI = dimension_numbers.kernel_input_feature_dimension();
  const int64_t dH = dimension_numbers.kernel_spatial_dimensions().at(0);
  const int64_t dW = dimension_numbers.kernel_spatial_dimensions().at(1);
  // In case of re-vectorization (rank=5), the missing dimension can be
  // calculated as Σi(i=0..4)-(dO+dI+dH+dW)
  bool revectorize = shape.dimensions_size() == 5;
  const int64_t dZ = revectorize ? 10 - dO - dI - dH - dW : -1;
  const int64_t vsize = revectorize ? shape.dimensions(dZ) : 1;

  // Verify convolution dimensions (should be vectorizable).
  if (shape.dimensions(dO) % 32 != 0 ||
      shape.dimensions(dI) % (32 / vsize) != 0 ||
      (revectorize && vsize != 4 && vsize != 32)) {
    return Internal("Filter shape is not vectorizable.");
  }

  // Build the resulting shape: [O, I/32, H, W, 32]
  std::vector<int64_t> output = {
      shape.dimensions(dO), shape.dimensions(dI) / (32 / vsize),
      shape.dimensions(dH), shape.dimensions(dW), 32};
  Shape output_shape = ShapeUtil::MakeShape(shape.element_type(), output);

  // Compute the positions of filter components in the transposable shape.
  // Every dimension preceding the given one moves it to the right, and
  // feature dimensions are split (into 2 or 3 components).
  auto calc_index = [&](int dim) {
    bool split_v = vsize == 32;
    return (revectorize
                ? (dI < dim ? 2 - split_v : 0) + (dZ < dim ? 1 + split_v : 0)
                : (dI < dim ? 3 : 0)) +
           (dO < dim ? 3 : 0) + (dH < dim) + (dW < dim);
  };
  int idx_O = calc_index(dO);
  int idx_I = calc_index(dI);
  int idx_H = calc_index(dH);
  int idx_W = calc_index(dW);
  // Position of input features split dimensions (8 and 4).
  int idx_Y = vsize == 32 ? calc_index(dZ) : idx_I + 1;
  int idx_Z = vsize == 4 ? calc_index(dZ) : vsize == 32 ? idx_Y + 1 : idx_I + 2;

  // Build the transposable shape: [O/8, 4, 2, I/32, 8, 4, H, W]
  std::vector<int64_t> dims(8);
  dims[idx_O] = shape.dimensions(dO) / 8;
  dims[idx_O + 1] = 4;
  dims[idx_O + 2] = 2;
  dims[idx_I] = shape.dimensions(dI) / (32 / vsize);
  dims[idx_Y] = 8;
  dims[idx_Z] = 4;
  dims[idx_H] = shape.dimensions(dH);
  dims[idx_W] = shape.dimensions(dW);
  Shape split_shape = ShapeUtil::MakeShape(shape.element_type(), dims);

  // Build the transposition permutation: [I/32, H, W, O/8, 2, 8, 4, 4]
  std::vector<int64_t> permutation = {idx_I,     idx_H, idx_W,     idx_O,
                                      idx_O + 2, idx_Y, idx_O + 1, idx_Z};
  return CudnnReorderTransposeConfig{split_shape, output_shape, permutation};
}

absl::StatusOr<CudnnReorderTransposeConfig>
CudnnInferTransposeForBiasReordering(const Shape& shape) {
  // Expected bias has one dimension: [O]
  if (shape.dimensions_size() != 1) {
    return Internal("Bias shape has unexpected rank.");
  }
  if (shape.dimensions(0) % 32 != 0) {
    return Internal("Bias shape is not vectorizable.");
  }

  // Build the transposable shape: [O/32, 4, 2, 4]
  std::vector<int64_t> dims = {shape.dimensions(0) / 32, 4, 2, 4};
  Shape split_shape = ShapeUtil::MakeShape(shape.element_type(), dims);

  // Build the transposition permutation: [O/32, 2, 4, 4]
  std::vector<int64_t> permutation = {0, 2, 1, 3};
  return CudnnReorderTransposeConfig{split_shape, shape, permutation};
}

bool IsWorkspaceAllocationRoot(const HloInstruction& root) {
  return root.IsRoot() && root.opcode() == HloOpcode::kTuple &&
         root.operand_count() == 2 &&
         root.operand(1)->IsCustomCall(kWorkspaceAllocationCustomCallTarget) &&
         root.operand(1)->operand_count() == 0;
}

}  // namespace gpu
}  // namespace xla
