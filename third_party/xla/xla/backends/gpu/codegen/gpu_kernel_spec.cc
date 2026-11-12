/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/backends/gpu/codegen/gpu_kernel_spec.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/codegen/gpu_kernel_spec.pb.h"
#include "xla/codegen/emitters/kernel_arguments.h"
#include "xla/codegen/emitters/kernel_arguments.pb.h"
#include "xla/codegen/kernel_spec.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/gpu/launch_dimensions.pb.h"
#include "xla/stream_executor/gpu/tma_metadata.h"
#include "xla/stream_executor/gpu/tma_metadata.pb.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/launch_dim.pb.h"

namespace xla::gpu {
namespace {
absl::flat_hash_set<int64_t> GetInvariantArguments(
    const emitters::KernelArguments& kernel_arguments) {
  absl::flat_hash_set<int64_t> res;
  const std::vector<emitters::KernelArgument>& args = kernel_arguments.args();
  for (int i = 0; i < args.size(); i++) {
    if (args[i].invariant()) {
      res.insert(i);
    }
  }
  return res;
}
}  // namespace

GpuKernelSpec::GpuKernelSpec(absl::string_view name,
                             const LaunchDimensions& launch_dimensions,
                             const emitters::KernelArguments& kernel_arguments,
                             std::optional<size_t> scratch_bytes, bool use_pdl,
                             stream_executor::gpu::TmaMetadata tma_metadata,
                             std::optional<se::ClusterDim> cluster_dim)
    : KernelSpec(name, launch_dimensions.AsWorkDimensions(),
                 kernel_arguments.GetArgumentShapedSlices(), {},
                 GetInvariantArguments(kernel_arguments), scratch_bytes),
      launch_dimensions_(launch_dimensions),
      kernel_arguments_(kernel_arguments),
      use_pdl_(use_pdl),
      tma_metadata_(tma_metadata),
      cluster_dim_(cluster_dim) {}

absl::StatusOr<GpuKernelSpec> GpuKernelSpec::FromProto(
    const GpuKernelSpecProto& proto,
    absl::Span<const BufferAllocation> buffer_allocations) {
  ABSL_ASSIGN_OR_RETURN(LaunchDimensions launch_dimensions,
                   LaunchDimensions::FromProto(proto.launch_dimensions()));
  ABSL_ASSIGN_OR_RETURN(emitters::KernelArguments kernel_arguments,
                   emitters::KernelArguments::FromProto(
                       proto.kernel_arguments(), buffer_allocations));
  std::optional<size_t> scratch_bytes;
  if (proto.has_scratch_bytes()) {
    if (proto.scratch_bytes() < 0) {
      return absl::InvalidArgumentError(absl::StrCat(
          "scratch_bytes must be non-negative, got ", proto.scratch_bytes()));
    }
    scratch_bytes = static_cast<size_t>(proto.scratch_bytes());
  }
  stream_executor::gpu::TmaMetadata tma_metadata;
  if (proto.has_tma_metadata()) {
    ABSL_ASSIGN_OR_RETURN(tma_metadata, stream_executor::gpu::TmaMetadata::FromProto(
                                       proto.tma_metadata()));
  }
  std::optional<stream_executor::ClusterDim> cluster_dim;
  if (proto.has_cluster_dim()) {
    ABSL_ASSIGN_OR_RETURN(
        cluster_dim.emplace(),
        stream_executor::ClusterDim::FromProto(proto.cluster_dim()));
  }
  return GpuKernelSpec(proto.name(), launch_dimensions, kernel_arguments,
                       scratch_bytes, proto.use_pdl(), std::move(tma_metadata),
                       cluster_dim);
}

absl::StatusOr<GpuKernelSpecProto> GpuKernelSpec::ToProto() const {
  GpuKernelSpecProto proto;
  proto.set_name(name());
  *proto.mutable_launch_dimensions() = launch_dimensions_.ToProto();
  ABSL_ASSIGN_OR_RETURN(*proto.mutable_kernel_arguments(),
                   kernel_arguments_.ToProto());
  if (scratch_bytes().has_value()) {
    if (*scratch_bytes() >
        static_cast<size_t>(std::numeric_limits<int64_t>::max())) {
      return absl::InvalidArgumentError(absl::StrCat(
          "scratch_bytes exceeds int64_t max: ", *scratch_bytes()));
    }
    proto.set_scratch_bytes(static_cast<int64_t>(*scratch_bytes()));
  }
  proto.set_use_pdl(use_pdl_);
  if (!tma_metadata_.arg_index_to_tma_info.empty()) {
    *proto.mutable_tma_metadata() = tma_metadata_.ToProto();
  }
  if (cluster_dim_.has_value()) {
    *proto.mutable_cluster_dim() = cluster_dim_->ToProto();
  }
  return proto;
}

}  // namespace xla::gpu
