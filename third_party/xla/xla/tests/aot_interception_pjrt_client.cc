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

#include "xla/tests/aot_interception_pjrt_client.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>

#include "absl/functional/any_invocable.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/layout.h"
#include "xla/literal.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/runtime/device_id.h"
#include "xla/service/computation_placer.h"
#include "xla/xla_data.pb.h"

namespace xla {

absl::StatusOr<std::unique_ptr<PjRtExecutable>>
AOTInterceptionPjrtClient::Compile(const XlaComputation& computation,
                                   CompileOptions options) {
  // Skeleton implementation: directly delegate to the underlying client.
  return inner_client_->Compile(computation, std::move(options));
}

absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>>
AOTInterceptionPjrtClient::CompileAndLoad(const XlaComputation& computation,
                                          CompileOptions options) {
  // Skeleton implementation: directly delegate to the underlying client.
  return inner_client_->CompileAndLoad(computation, std::move(options));
}

absl::StatusOr<PjRtDevice*> AOTInterceptionPjrtClient::LookupDevice(
    GlobalDeviceId global_device_id) const {
  return inner_client_->LookupDevice(global_device_id);
}

absl::StatusOr<PjRtDevice*> AOTInterceptionPjrtClient::LookupAddressableDevice(
    LocalDeviceId local_device_id) const {
  return inner_client_->LookupAddressableDevice(local_device_id);
}

absl::StatusOr<std::unique_ptr<PjRtBuffer>>
AOTInterceptionPjrtClient::BufferFromHostLiteral(
    const LiteralSlice& literal, PjRtMemorySpace* memory_space) {
  return inner_client_->BufferFromHostLiteral(literal, memory_space);
}

absl::StatusOr<std::unique_ptr<PjRtBuffer>>
AOTInterceptionPjrtClient::BufferFromHostLiteral(const LiteralSlice& literal,
                                                 PjRtMemorySpace* memory_space,
                                                 const Layout* device_layout) {
  return inner_client_->BufferFromHostLiteral(literal, memory_space,
                                              device_layout);
}

absl::StatusOr<std::unique_ptr<PjRtBuffer>>
AOTInterceptionPjrtClient::BufferFromHostBuffer(
    const void* data, PrimitiveType type, absl::Span<int64_t const> dims,
    std::optional<absl::Span<int64_t const>> byte_strides,
    PjRtClient::HostBufferSemantics host_buffer_semantics,
    absl::AnyInvocable<void() &&> on_done_with_host_buffer,
    PjRtMemorySpace* memory_space, const Layout* device_layout) {
  return inner_client_->BufferFromHostBuffer(
      data, type, dims, byte_strides, host_buffer_semantics,
      std::move(on_done_with_host_buffer), memory_space, device_layout);
}

absl::StatusOr<std::unique_ptr<PjRtBuffer>>
AOTInterceptionPjrtClient::BufferFromHostBuffer(
    const void* data, PrimitiveType type, absl::Span<int64_t const> dims,
    std::optional<absl::Span<int64_t const>> byte_strides,
    PjRtClient::HostBufferSemantics host_buffer_semantics,
    absl::AnyInvocable<void() &&> on_done_with_host_buffer,
    PjRtBuffer* donated_dst, const Layout* device_layout) {
  return inner_client_->BufferFromHostBuffer(
      data, type, dims, byte_strides, host_buffer_semantics,
      std::move(on_done_with_host_buffer), donated_dst, device_layout);
}

absl::StatusOr<DeviceAssignment>
AOTInterceptionPjrtClient::GetDefaultDeviceAssignment(
    int num_replicas, int num_partitions) const {
  return inner_client_->GetDefaultDeviceAssignment(num_replicas,
                                                   num_partitions);
}

absl::StatusOr<DeviceAssignment>
AOTInterceptionPjrtClient::GetDefaultDeviceAssignment(
    int num_replicas, std::optional<int> num_replicas_per_slice,
    int num_partitions, const MultiSliceConfig* multi_slice_config) const {
  return inner_client_->GetDefaultDeviceAssignment(
      num_replicas, num_replicas_per_slice, num_partitions, multi_slice_config);
}

absl::StatusOr<std::unique_ptr<PjRtExecutable>>
AOTInterceptionPjrtClient::DeserializeExecutable(
    absl::string_view serialized, std::optional<CompileOptions> options) {
  return inner_client_->DeserializeExecutable(serialized, std::move(options));
}

absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>>
AOTInterceptionPjrtClient::Load(std::shared_ptr<PjRtExecutable> executable,
                                const LoadOptions& load_options) {
  return inner_client_->Load(std::move(executable), load_options);
}

}  // namespace xla
