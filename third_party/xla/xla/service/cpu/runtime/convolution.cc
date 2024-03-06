// Copyright 2023 The OpenXLA Authors.
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

#include "xla/service/cpu/runtime/convolution.h"

#include <cstdint>
#include <memory>
#include <vector>

#include "absl/status/status.h"
#include "absl/types/span.h"
#include "Eigen/Core"  // from @eigen_archive
#include "xla/executable_run_options.h"
#include "xla/runtime/memref_view.h"
#include "xla/service/cpu/runtime_conv2d.h"
#include "xla/service/cpu/runtime_conv3d.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace cpu {

using ::xla::runtime::MemrefView;

absl::Status XlaConvolution::operator()(
    const ExecutableRunOptions* run_options, MemrefView input,
    MemrefView kernel, MemrefView output, int64_t inputBatchDimension,
    absl::Span<const int64_t> inputSpatialDimensions,
    int64_t inputFeatureDimension,
    absl::Span<const int64_t> kernelSpatialDimensions,
    int64_t kernelInputFeatureDimension, int64_t kernelOutputFeatureDimension,
    absl::Span<const int64_t> outputSpatialDimensions,
    absl::Span<const int64_t> window_strides, absl::Span<const int64_t> padding,
    absl::Span<const int64_t> lhs_dilation,
    absl::Span<const int64_t> rhs_dilation, int64_t feature_group_count) const {
  auto size = inputSpatialDimensions.size();
  if (size < 1 || size > 3) {
    return absl::InvalidArgumentError(
        "Only 1D, 2D and 3D convolutions are supported");
  }

  if (size != kernelSpatialDimensions.size() ||
      size != outputSpatialDimensions.size() || size != window_strides.size() ||
      size * 2 != padding.size() || size != lhs_dilation.size() ||
      size != rhs_dilation.size()) {
    return absl::InvalidArgumentError("Number of attributes mismatched");
  }

  // We lower 1D convolutions into calls to the same Eigen function as 2D
  // convolutions, except that we pretend that the 1D convolution is really a 2D
  // convolution with the missing dimension set to 1.  We also adjust the
  // padding, dilation parameters as needed.
  std::vector<int64_t> input_dims;
  std::vector<int64_t> kernel_dims;
  std::vector<int64_t> output_dims;
  std::vector<int64_t> strides;
  std::vector<int64_t> pad;
  std::vector<int64_t> base_dilation;
  std::vector<int64_t> window_dilation;
  if (size == 1) {
    input_dims.push_back(1);
    kernel_dims.push_back(1);
    output_dims.push_back(1);
    strides.push_back(1);
    pad.insert(pad.end(), {0, 0});
    base_dilation.push_back(1);
    window_dilation.push_back(1);
  }
  for (auto dim : inputSpatialDimensions) {
    input_dims.push_back(input.sizes[dim]);
  }
  for (auto dim : kernelSpatialDimensions) {
    kernel_dims.push_back(kernel.sizes[dim]);
  }
  for (auto dim : outputSpatialDimensions) {
    output_dims.push_back(output.sizes[dim]);
  }
  strides.insert(strides.end(), window_strides.begin(), window_strides.end());
  pad.insert(pad.end(), padding.begin(), padding.end());
  base_dilation.insert(base_dilation.end(), lhs_dilation.begin(),
                       lhs_dilation.end());
  window_dilation.insert(window_dilation.end(), rhs_dilation.begin(),
                         rhs_dilation.end());

  if (output.dtype == PrimitiveType::F16) {
    auto* out = reinterpret_cast<Eigen::half*>(output.data);
    auto* lhs = reinterpret_cast<Eigen::half*>(input.data);
    auto* rhs = reinterpret_cast<Eigen::half*>(kernel.data);
    if (size != 3) {
      __xla_cpu_runtime_EigenConv2DF16(
          run_options, out, lhs, rhs,
          /*input_batch*/ input.sizes[inputBatchDimension],
          /*input_rows*/ input_dims[0],
          /*input_cols*/ input_dims[1],
          /*input_channels*/ input.sizes[inputFeatureDimension],
          /*kernel_rows*/ kernel_dims[0],
          /*kernel_cols*/ kernel_dims[1],
          /*kernel_channels*/ kernel.sizes[kernelInputFeatureDimension],
          /*kernel_filters*/ kernel.sizes[kernelOutputFeatureDimension],
          /*output_rows*/ output_dims[0],
          /*output_cols*/ output_dims[1],
          /*row_stride*/ strides[0],
          /*col_stride*/ strides[1],
          /*padding_top*/ pad[0],
          /*padding_bottom*/ pad[1],
          /*padding_left*/ pad[2],
          /*padding_right*/ pad[3],
          /*lhs_row_dilation*/ base_dilation[0],
          /*lhs_col_dilation*/ base_dilation[1],
          /*rhs_row_dilation*/ window_dilation[0],
          /*rhs_col_dilation*/ window_dilation[1], feature_group_count);
    } else {
      __xla_cpu_runtime_EigenConv3DF16(
          run_options, out, lhs, rhs,
          /*input_batch*/ input.sizes[inputBatchDimension],
          /*input_x*/ input_dims[0],
          /*input_y*/ input_dims[1],
          /*input_z*/ input_dims[2],
          /*input_channels*/ input.sizes[inputFeatureDimension],
          /*kernel_x*/ kernel_dims[0],
          /*kernel_y*/ kernel_dims[1],
          /*kernel_z*/ kernel_dims[2],
          /*kernel_channels*/ kernel.sizes[kernelInputFeatureDimension],
          /*kernel_filters*/ kernel.sizes[kernelOutputFeatureDimension],
          /*output_x*/ output_dims[0],
          /*output_y*/ output_dims[1],
          /*output_z*/ output_dims[2],
          /*x_stride*/ strides[0],
          /*y_stride*/ strides[1],
          /*z_stride*/ strides[2],
          /*padding_x_before*/ pad[0],
          /*padding_x_after*/ pad[1],
          /*padding_y_before*/ pad[2],
          /*padding_y_after*/ pad[3],
          /*padding_z_before*/ pad[4],
          /*padding_z_after*/ pad[5],
          /*lhs_x_dilation*/ base_dilation[0],
          /*lhs_y_dilation*/ base_dilation[1],
          /*lhs_z_dilation*/ base_dilation[2],
          /*rhs_x_dilation*/ window_dilation[0],
          /*rhs_y_dilation*/ window_dilation[1],
          /*rhs_z_dilation*/ window_dilation[2], feature_group_count);
    }
  } else {
    auto* out = reinterpret_cast<float*>(output.data);
    auto* lhs = reinterpret_cast<float*>(input.data);
    auto* rhs = reinterpret_cast<float*>(kernel.data);
    if (size != 3) {
      __xla_cpu_runtime_EigenConv2DF32(
          run_options, out, lhs, rhs,
          /*input_batch*/ input.sizes[inputBatchDimension],
          /*input_rows*/ input_dims[0],
          /*input_cols*/ input_dims[1],
          /*input_channels*/ input.sizes[inputFeatureDimension],
          /*kernel_rows*/ kernel_dims[0],
          /*kernel_cols*/ kernel_dims[1],
          /*kernel_channels*/ kernel.sizes[kernelInputFeatureDimension],
          /*kernel_filters*/ kernel.sizes[kernelOutputFeatureDimension],
          /*output_rows*/ output_dims[0],
          /*output_cols*/ output_dims[1],
          /*row_stride*/ strides[0],
          /*col_stride*/ strides[1],
          /*padding_top*/ pad[0],
          /*padding_bottom*/ pad[1],
          /*padding_left*/ pad[2],
          /*padding_right*/ pad[3],
          /*lhs_row_dilation*/ base_dilation[0],
          /*lhs_col_dilation*/ base_dilation[1],
          /*rhs_row_dilation*/ window_dilation[0],
          /*rhs_col_dilation*/ window_dilation[1], feature_group_count);
    } else {
      __xla_cpu_runtime_EigenConv3DF32(
          run_options, out, lhs, rhs,
          /*input_batch*/ input.sizes[inputBatchDimension],
          /*input_x*/ input_dims[0],
          /*input_y*/ input_dims[1],
          /*input_z*/ input_dims[2],
          /*input_channels*/ input.sizes[inputFeatureDimension],
          /*kernel_x*/ kernel_dims[0],
          /*kernel_y*/ kernel_dims[1],
          /*kernel_z*/ kernel_dims[2],
          /*kernel_channels*/ kernel.sizes[kernelInputFeatureDimension],
          /*kernel_filters*/ kernel.sizes[kernelOutputFeatureDimension],
          /*output_x*/ output_dims[0],
          /*output_y*/ output_dims[1],
          /*output_z*/ output_dims[2],
          /*x_stride*/ strides[0],
          /*y_stride*/ strides[1],
          /*z_stride*/ strides[2],
          /*padding_x_before*/ pad[0],
          /*padding_x_after*/ pad[1],
          /*padding_y_before*/ pad[2],
          /*padding_y_after*/ pad[3],
          /*padding_z_before*/ pad[4],
          /*padding_z_after*/ pad[5],
          /*lhs_x_dilation*/ base_dilation[0],
          /*lhs_y_dilation*/ base_dilation[1],
          /*lhs_z_dilation*/ base_dilation[2],
          /*rhs_x_dilation*/ window_dilation[0],
          /*rhs_y_dilation*/ window_dilation[1],
          /*rhs_z_dilation*/ window_dilation[2], feature_group_count);
    }
  }

  return absl::OkStatus();
}

}  // namespace cpu
}  // namespace xla
