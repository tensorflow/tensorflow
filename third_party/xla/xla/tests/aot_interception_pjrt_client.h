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

#ifndef XLA_TESTS_AOT_INTERCEPTION_PJRT_CLIENT_H_
#define XLA_TESTS_AOT_INTERCEPTION_PJRT_CLIENT_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "absl/functional/any_invocable.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/layout.h"
#include "xla/literal.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_common.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/runtime/device_id.h"
#include "xla/service/computation_placer.h"
#include "xla/xla_data.pb.h"

namespace xla {

// The interception mode dictated by the testing macro.
enum class AOTTestMode {
  kGoldenVerification,
  kBackwardsCompatibility,
};

// A wrapper around an existing PjRtClient that intercepts compilation
// and loading calls to enforce AOT compatibility verification.
class AOTInterceptionPjrtClient : public PjRtClient {
 public:
  AOTInterceptionPjrtClient(std::unique_ptr<PjRtClient> inner_client,
                            AOTTestMode mode, std::string artifact_path)
      : inner_client_(std::move(inner_client)),
        mode_(mode),
        artifact_path_(std::move(artifact_path)) {}

  ~AOTInterceptionPjrtClient() override = default;

  // Intercepted compilation methods.
  absl::StatusOr<std::unique_ptr<PjRtExecutable>> Compile(
      const XlaComputation& computation, CompileOptions options) override;

  absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>> CompileAndLoad(
      const XlaComputation& computation, CompileOptions options) override;

  // Delegates to the inner client.
  int process_index() const override { return inner_client_->process_index(); }
  int device_count() const override { return inner_client_->device_count(); }
  int addressable_device_count() const override {
    return inner_client_->addressable_device_count();
  }
  absl::Span<PjRtDevice* const> devices() const override {
    return inner_client_->devices();
  }
  absl::Span<PjRtDevice* const> addressable_devices() const override {
    return inner_client_->addressable_devices();
  }
  absl::Span<PjRtMemorySpace* const> memory_spaces() const override {
    return inner_client_->memory_spaces();
  }
  PjRtPlatformId platform_id() const override {
    return inner_client_->platform_id();
  }
  absl::string_view platform_name() const override {
    return inner_client_->platform_name();
  }
  absl::string_view platform_version() const override {
    return inner_client_->platform_version();
  }

  absl::StatusOr<PjRtDevice*> LookupDevice(
      GlobalDeviceId global_device_id) const override;

  absl::StatusOr<PjRtDevice*> LookupAddressableDevice(
      LocalDeviceId local_device_id) const override;

  // Expose the inner client for any testing needs.
  PjRtClient* inner_client() const { return inner_client_.get(); }

  absl::StatusOr<std::unique_ptr<PjRtBuffer>> BufferFromHostLiteral(
      const LiteralSlice& literal, PjRtMemorySpace* memory_space) override;

  absl::StatusOr<std::unique_ptr<PjRtBuffer>> BufferFromHostLiteral(
      const LiteralSlice& literal, PjRtMemorySpace* memory_space,
      const Layout* device_layout) override;

  absl::StatusOr<std::unique_ptr<PjRtBuffer>> BufferFromHostBuffer(
      const void* data, PrimitiveType type, absl::Span<int64_t const> dims,
      std::optional<absl::Span<int64_t const>> byte_strides,
      PjRtClient::HostBufferSemantics host_buffer_semantics,
      absl::AnyInvocable<void() &&> on_done_with_host_buffer,
      PjRtMemorySpace* memory_space, const Layout* device_layout) override;

  absl::StatusOr<std::unique_ptr<PjRtBuffer>> BufferFromHostBuffer(
      const void* data, PrimitiveType type, absl::Span<int64_t const> dims,
      std::optional<absl::Span<int64_t const>> byte_strides,
      PjRtClient::HostBufferSemantics host_buffer_semantics,
      absl::AnyInvocable<void() &&> on_done_with_host_buffer,
      PjRtBuffer* donated_dst, const Layout* device_layout) override;

  absl::StatusOr<DeviceAssignment> GetDefaultDeviceAssignment(
      int num_replicas, int num_partitions) const override;

  absl::StatusOr<DeviceAssignment> GetDefaultDeviceAssignment(
      int num_replicas, std::optional<int> num_replicas_per_slice,
      int num_partitions,
      const MultiSliceConfig* multi_slice_config) const override;

  absl::StatusOr<std::unique_ptr<PjRtExecutable>> DeserializeExecutable(
      absl::string_view serialized,
      std::optional<CompileOptions> options) override;

  absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>> Load(
      std::shared_ptr<PjRtExecutable> executable,
      const LoadOptions& load_options) override;

 private:
  std::unique_ptr<PjRtClient> inner_client_;
  AOTTestMode mode_;
  std::string artifact_path_;
};

}  // namespace xla

#endif  // XLA_TESTS_AOT_INTERCEPTION_PJRT_CLIENT_H_
