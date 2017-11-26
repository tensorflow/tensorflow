/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/cpu/ir_emission_utils.h"

#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/service/cpu/cpu_runtime.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/window_util.h"

namespace xla {
namespace cpu {

bool PotentiallyImplementedAsEigenConvolution(
    const HloInstruction& convolution) {
  // The following conditions are necessary (but not sufficient) for
  // implementing `convolution` with Eigen convolution:
  // - the input and kernel have a non-zero number of elements.
  // - the input is in NHWC or NWHC order.
  // - the kernel is in HWIO or WHIO order.
  // - the spatial dimensions are in the same relative order in the input,
  //   kernel and output.
  //
  // To be sufficient, certain layout constraints need to be satisfied as well.
  const Shape& input_shape = convolution.operand(0)->shape();
  const Shape& kernel_shape = convolution.operand(0)->shape();
  if (ShapeUtil::HasZeroElements(input_shape) ||
      ShapeUtil::HasZeroElements(kernel_shape)) {
    return false;
  }
  // TODO(b/65408531): Explore using Eigen dot for complex64 type.
  if (ShapeUtil::ElementIsComplex(input_shape) ||
      ShapeUtil::ElementIsComplex(kernel_shape)) {
    return false;
  }

  const ConvolutionDimensionNumbers& dnums =
      convolution.convolution_dimension_numbers();
  // Only 1D and 2D convolutions are supported at the moment.
  // TODO(b/32897908): add an optimized implementation for 3D convolution.
  if (dnums.spatial_dimensions_size() > 2) {
    return false;
  }

  bool input_spatial_dims_ascending = std::is_sorted(
      dnums.spatial_dimensions().begin(), dnums.spatial_dimensions().end());
  bool kernel_spatial_dims_ascending =
      std::is_sorted(dnums.kernel_spatial_dimensions().begin(),
                     dnums.kernel_spatial_dimensions().end());

  const Shape& output_shape = convolution.shape();
  return dnums.input_batch_dimension() == 0 &&
         dnums.input_feature_dimension() == input_shape.dimensions_size() - 1 &&
         dnums.output_batch_dimension() == 0 &&
         dnums.output_feature_dimension() ==
             output_shape.dimensions_size() - 1 &&
         input_spatial_dims_ascending == kernel_spatial_dims_ascending &&
         dnums.kernel_input_feature_dimension() ==
             kernel_shape.dimensions_size() - 2 &&
         dnums.kernel_output_feature_dimension() ==
             kernel_shape.dimensions_size() - 1;
}

}  // namespace cpu
}  // namespace xla
