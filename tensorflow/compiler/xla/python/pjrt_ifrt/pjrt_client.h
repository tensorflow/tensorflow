/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_PYTHON_PJRT_IFRT_PJRT_CLIENT_H_
#define TENSORFLOW_COMPILER_XLA_PYTHON_PJRT_IFRT_PJRT_CLIENT_H_

#include <functional>
#include <memory>
#include <optional>
#include <utility>

#include "absl/types/span.h"
#include "llvm/Support/ExtensibleRTTI.h"
#include "tensorflow/compiler/xla/pjrt/pjrt_client.h"
#include "tensorflow/compiler/xla/python/ifrt/client.h"
#include "tensorflow/compiler/xla/python/pjrt_ifrt/pjrt_compiler.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace xla {
namespace ifrt {

// `Client` implementation that wraps `xla::PjRtClient`.
class PjRtClient final : public llvm::RTTIExtends<PjRtClient, Client> {
 public:
  static std::unique_ptr<ifrt::Client> Create(
      std::shared_ptr<xla::PjRtClient> pjrt_client);
  static std::unique_ptr<ifrt::Client> Create(
      std::unique_ptr<xla::PjRtClient> pjrt_client);

  xla::PjRtClient* pjrt_client() const { return pjrt_client_.get(); }
  std::shared_ptr<xla::PjRtClient> shared_ptr_pjrt_client() const {
    return pjrt_client_;
  }

  // Client implementation.

  ~PjRtClient() override = default;

  StatusOr<std::unique_ptr<Array>> MakeArrayFromHostBuffer(
      const void* data, DType dtype, Shape shape,
      std::optional<absl::Span<const int64_t>> byte_strides,
      std::shared_ptr<const Sharding> sharding,
      Client::HostBufferSemantics semantics,
      std::function<void()> on_done_with_host_buffer) override;

  StatusOr<std::unique_ptr<Array>> AssembleArrayFromSingleDeviceArrays(
      Shape shape, std::shared_ptr<const Sharding> sharding,
      absl::Span<Array* const> arrays, ArrayCopySemantics semantics) override;

  absl::string_view runtime_type() const override {
    DCHECK(this);
    return PjRtRuntimeTypeString(pjrt_client_->runtime_type());
  }

  absl::string_view platform_name() const override {
    DCHECK(this);
    return pjrt_client_->platform_name();
  }
  absl::string_view platform_version() const override {
    DCHECK(this);
    return pjrt_client_->platform_version();
  }
  PlatformId platform_id() const override {
    DCHECK(this);
    return pjrt_client_->platform_id();
  }

  int device_count() const override {
    DCHECK(this);
    return pjrt_client_->device_count();
  }
  int addressable_device_count() const override {
    DCHECK(this);
    return pjrt_client_->addressable_device_count();
  }
  absl::Span<Device* const> devices() const override {
    DCHECK(this);
    return pjrt_client_->devices();
  }
  absl::Span<Device* const> addressable_devices() const override {
    DCHECK(this);
    return pjrt_client_->addressable_devices();
  }
  int process_index() const override { return pjrt_client_->process_index(); }
  StatusOr<DeviceAssignment> GetDefaultDeviceAssignment(
      int num_replicas, int num_partitions) const override {
    DCHECK(this);
    return pjrt_client_->GetDefaultDeviceAssignment(num_replicas,
                                                    num_partitions);
  }
  StatusOr<Device*> LookupDevice(int device_id) const override {
    DCHECK(this);
    return pjrt_client_->LookupDevice(device_id);
  }

  StatusOr<ChannelHandle> CreateDeviceToHostChannelHandle() override {
    DCHECK(this);
    return pjrt_client_->CreateDeviceToHostChannelHandle();
  }
  StatusOr<ChannelHandle> CreateHostToDeviceChannelHandle() override {
    DCHECK(this);
    return pjrt_client_->CreateHostToDeviceChannelHandle();
  }

  Compiler* GetDefaultCompiler() const override {
    DCHECK(this);
    return const_cast<PjRtCompiler*>(&default_compiler_);
  }

  static char ID;  // NOLINT

 private:
  explicit PjRtClient(std::shared_ptr<xla::PjRtClient> pjrt_client)
      : pjrt_client_(std::move(pjrt_client)), default_compiler_(this) {}

  std::shared_ptr<xla::PjRtClient> pjrt_client_;
  PjRtCompiler default_compiler_;
};

}  // namespace ifrt
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_PYTHON_PJRT_IFRT_PJRT_CLIENT_H_
