/* Copyright 2024 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_CPU_RUNTIME_XNNPACK_XNN_DOT_THUNK_H_
#define XLA_BACKENDS_CPU_RUNTIME_XNNPACK_XNN_DOT_THUNK_H_

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/backends/cpu/runtime/dot_lib.h"
#include "xla/backends/cpu/runtime/thunk.h"
#include "xla/backends/cpu/runtime/xnnpack/xnn_fusion_thunk.h"
#include "xla/service/buffer_assignment.h"
#include "xla/shape.h"

namespace xla::cpu {

// Dot operation implemented on top of XNNPACK.
class XnnDotThunk final : public XnnFusionThunk {
 public:
  static absl::StatusOr<std::unique_ptr<XnnDotThunk>> Create(
      Options options, Info info, DotDimensionNumbers dot_dimensions,
      BufferAllocation::Slice lhs_buffer, Shape lhs_shape,
      BufferAllocation::Slice rhs_buffer, Shape rhs_shape,
      BufferAllocation::Slice out_buffer, Shape out_shape);

  DotDimensionNumbers dot_dimensions() const { return dot_dimensions_; }
  DotSlices dot_slices() const { return dot_slices_; }

 protected:
  std::string fusion_kind() const final;
  std::string fusion_description() const final;

  bool has_fusion_details() const final { return true; }
  std::vector<std::string> fusion_details() const final;

  std::string argument_name(size_t index) const final;
  std::string result_name(size_t index) const final;

 private:
  XnnDotThunk(Options options, Info info, DotDimensionNumbers dot_dimensions,
              DotSlices dot_slices, DotShape dot_shape,
              DotCanonicalDims dot_canonical_dims);

  absl::StatusOr<xnn_subgraph_t> BuildDotSubgraph(
      absl::Span<const Argument> arguments, absl::Span<const Result> results);

  DotDimensionNumbers dot_dimensions_;
  DotSlices dot_slices_;
  DotShape dot_shape_;
  DotCanonicalDims dot_canonical_dims_;
};

}  // namespace xla::cpu

#endif  // XLA_BACKENDS_CPU_RUNTIME_XNNPACK_XNN_DOT_THUNK_H_
