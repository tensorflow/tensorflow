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
#ifndef XLA_SERVICE_CPU_RUNTIME_CONVOLUTION_H_
#define XLA_SERVICE_CPU_RUNTIME_CONVOLUTION_H_

#include <cstdint>

#include "absl/types/span.h"
#include "xla/executable_run_options.h"
#include "xla/runtime/memref_view.h"

namespace xla {
namespace cpu {

struct XlaConvolution {
  absl::Status operator()(
      const ExecutableRunOptions* run_options, xla::runtime::MemrefView input,
      xla::runtime::MemrefView kernel, xla::runtime::MemrefView output,
      int64_t inputBatchDimension,
      absl::Span<const int64_t> inputSpatialDimensions,
      int64_t inputFeatureDimension,
      absl::Span<const int64_t> kernelSpatialDimensions,
      int64_t kernelInputFeatureDimension, int64_t kernelOutputFeatureDimension,
      absl::Span<const int64_t> outputSpatialDimensions,
      absl::Span<const int64_t> window_strides,
      absl::Span<const int64_t> padding, absl::Span<const int64_t> lhs_dilation,
      absl::Span<const int64_t> rhs_dilation,
      int64_t feature_group_count) const;
  static XlaConvolution Handler() { return XlaConvolution(); }
};

}  // namespace cpu
}  // namespace xla

#endif  // XLA_SERVICE_CPU_RUNTIME_CONVOLUTION_H_
