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

#include "xla/service/gpu/kernels/custom_kernel.h"

#include <cstddef>
#include <optional>
#include <string>
#include <utility>

#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "xla/service/gpu/kernels/custom_kernel.pb.h"
#include "xla/stream_executor/kernel_spec.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/tsl/platform/statusor.h"

namespace xla::gpu {

CustomKernel::CustomKernel(std::string name, se::KernelLoaderSpec kernel_spec,
                           se::BlockDim block_dims, se::ThreadDim thread_dims,
                           size_t shared_memory_bytes)
    : name_(std::move(name)),
      kernel_spec_(std::move(kernel_spec)),
      block_dims_(block_dims),
      thread_dims_(thread_dims),
      cluster_dims_(std::nullopt),
      shared_memory_bytes_(shared_memory_bytes) {}

CustomKernel::CustomKernel(std::string name, se::KernelLoaderSpec kernel_spec,
                           se::BlockDim block_dims, se::ThreadDim thread_dims,
                           se::ClusterDim cluster_dims,
                           size_t shared_memory_bytes)
    : name_(std::move(name)),
      kernel_spec_(std::move(kernel_spec)),
      block_dims_(block_dims),
      thread_dims_(thread_dims),
      cluster_dims_(cluster_dims),
      shared_memory_bytes_(shared_memory_bytes) {}

absl::string_view CustomKernel::name() const { return name_; }

const se::KernelLoaderSpec& CustomKernel::kernel_spec() const {
  return kernel_spec_;
}

se::BlockDim CustomKernel::block_dims() const { return block_dims_; }

se::ThreadDim CustomKernel::thread_dims() const { return thread_dims_; }

std::optional<se::ClusterDim> CustomKernel::cluster_dims() const {
  return cluster_dims_;
}

size_t CustomKernel::shared_memory_bytes() const {
  return shared_memory_bytes_;
}

std::string CustomKernel::ToString() const {
  std::string cluster_dims_str =
      cluster_dims_.has_value()
          ? absl::StrFormat("cluster: [%d, %d, %d]", cluster_dims_->x,
                            cluster_dims_->y, cluster_dims_->z)
          : "";
  return absl::StrFormat(
      "%s grid: [%d, %d, %d] threads: [%d, %d, %d] %s "
      "shared_memory: %d bytes",
      name_, block_dims_.x, block_dims_.y, block_dims_.z, thread_dims_.x,
      thread_dims_.y, thread_dims_.z, cluster_dims_str, shared_memory_bytes_);
}

absl::StatusOr<CustomKernelProto> CustomKernel::ToProto() const {
  CustomKernelProto proto;
  proto.set_name(name_);
  TF_ASSIGN_OR_RETURN(*proto.mutable_kernel_spec(), kernel_spec_.ToProto());
  *proto.mutable_block_dims() = block_dims_.ToProto();
  *proto.mutable_thread_dims() = thread_dims_.ToProto();
  if (cluster_dims_.has_value()) {
    *proto.mutable_cluster_dim() = cluster_dims_->ToProto();
  }
  proto.set_shared_memory_bytes(shared_memory_bytes_);
  return proto;
}

absl::StatusOr<CustomKernel> CustomKernel::FromProto(
    const CustomKernelProto& proto,
    const std::optional<se::KernelLoaderSpec::SymbolResolver>&
        symbol_resolver) {
  TF_ASSIGN_OR_RETURN(
      se::KernelLoaderSpec kernel_spec,
      se::KernelLoaderSpec::FromProto(proto.kernel_spec(), symbol_resolver));
  TF_ASSIGN_OR_RETURN(se::BlockDim block_dims,
                      se::BlockDim::FromProto(proto.block_dims()));
  TF_ASSIGN_OR_RETURN(se::ThreadDim thread_dims,
                      se::ThreadDim::FromProto(proto.thread_dims()));
  if (proto.has_cluster_dim()) {
    TF_ASSIGN_OR_RETURN(se::ClusterDim cluster_dims,
                        se::ClusterDim::FromProto(proto.cluster_dim()));
    return CustomKernel(proto.name(), std::move(kernel_spec), block_dims,
                        thread_dims, cluster_dims, proto.shared_memory_bytes());
  }

  return CustomKernel(proto.name(), std::move(kernel_spec), block_dims,
                      thread_dims, proto.shared_memory_bytes());
}

}  // namespace xla::gpu
