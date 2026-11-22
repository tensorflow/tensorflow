/* Copyright 2023 The OpenXLA Authors.

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

#include "xla/backends/gpu/codegen/kernels/custom_kernel.h"

#include <cstddef>
#include <optional>
#include <string>
#include <utility>

#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/codegen/gpu_kernel_spec.h"
#include "xla/backends/gpu/codegen/kernels/custom_kernel.pb.h"
#include "xla/codegen/emitters/kernel_arguments.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/stream_executor/kernel_spec.h"
#include "xla/stream_executor/launch_dim.h"

namespace xla::gpu {

CustomKernel::CustomKernel(const GpuKernelSpec& spec,
                           se::KernelLoaderSpec kernel_spec)
    : loader_spec_(std::move(kernel_spec)), spec_(spec) {}

absl::string_view CustomKernel::name() const { return spec_.name(); }

const se::BlockDim& CustomKernel::block_dims() const {
  return spec_.launch_dimensions().block_counts();
}

const se::ThreadDim& CustomKernel::thread_dims() const {
  return spec_.launch_dimensions().thread_counts_per_block();
}

const std::optional<se::ClusterDim>& CustomKernel::cluster_dims() const {
  return spec_.launch_dimensions().cluster_dims();
}

size_t CustomKernel::shared_memory_bytes() const { return spec_.shmem_bytes(); }

std::string CustomKernel::ToString() const {
  std::string cluster_dims_str =
      cluster_dims().has_value()
          ? absl::StrFormat("cluster: [%d, %d, %d]", cluster_dims()->x,
                            cluster_dims()->y, cluster_dims()->z)
          : "";
  return absl::StrFormat(
      "%s grid: [%d, %d, %d] threads: [%d, %d, %d] %s "
      "shared_memory: %d bytes",
      spec_.name(), block_dims().x, block_dims().y, block_dims().z,
      thread_dims().x, thread_dims().y, thread_dims().z, cluster_dims_str,
      shared_memory_bytes());
}

absl::StatusOr<CustomKernelProto> CustomKernel::ToProto() const {
  CustomKernelProto proto;
  ABSL_ASSIGN_OR_RETURN(*proto.mutable_kernel_spec(), loader_spec_.ToProto());

  // TODO(b/578658607) remove forward compatibility code after Nov 30.
  proto.set_name(spec_.name());
  *proto.mutable_block_dims() =
      spec_.launch_dimensions().block_counts().ToProto();
  *proto.mutable_thread_dims() =
      spec_.launch_dimensions().thread_counts_per_block().ToProto();
  if (cluster_dims().has_value()) {
    *proto.mutable_cluster_dim() = cluster_dims()->ToProto();
  }
  proto.set_shared_memory_bytes(spec_.shmem_bytes());

  ABSL_ASSIGN_OR_RETURN(*proto.mutable_spec(), spec_.ToProto());
  return proto;
}

absl::StatusOr<CustomKernel> CustomKernel::FromProto(
    const CustomKernelProto& proto,
    absl::Span<const BufferAllocation> buffer_allocations,
    const std::optional<se::KernelLoaderSpec::SymbolResolver>&
        symbol_resolver) {
  ABSL_ASSIGN_OR_RETURN(
      se::KernelLoaderSpec kernel_spec,
      se::KernelLoaderSpec::FromProto(proto.kernel_spec(), symbol_resolver));
  if (proto.has_spec()) {
    ABSL_ASSIGN_OR_RETURN(GpuKernelSpec spec, GpuKernelSpec::FromProto(
                                             proto.spec(), buffer_allocations));
    return CustomKernel(std::move(spec), std::move(kernel_spec));
  }

  // TODO(b/578658607) remove backward compatibility code after Nov 13 2026 +
  // AOT compatibility window duration.
  ABSL_ASSIGN_OR_RETURN(se::BlockDim block_dims,
                   se::BlockDim::FromProto(proto.block_dims()));
  ABSL_ASSIGN_OR_RETURN(se::ThreadDim thread_dims,
                   se::ThreadDim::FromProto(proto.thread_dims()));
  std::optional<se::ClusterDim> cluster_dim;
  if (proto.has_cluster_dim()) {
    ABSL_ASSIGN_OR_RETURN(cluster_dim,
                     se::ClusterDim::FromProto(proto.cluster_dim()));
  }

  GpuKernelSpec spec{
      proto.name(), LaunchDimensions{block_dims, thread_dims, cluster_dim},
      emitters::KernelArguments({}), proto.shared_memory_bytes()};

  return CustomKernel(std::move(spec), std::move(kernel_spec));
}

}  // namespace xla::gpu
