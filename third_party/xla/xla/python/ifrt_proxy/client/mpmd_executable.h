/*
 * Copyright 2025 The OpenXLA Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef XLA_PYTHON_IFRT_PROXY_CLIENT_MPMD_EXECUTABLE_H_
#define XLA_PYTHON_IFRT_PROXY_CLIENT_MPMD_EXECUTABLE_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "llvm/Support/ExtensibleRTTI.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/pjrt_layout.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/attribute_map.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/device_list.h"
#include "xla/python/ifrt/executable.h"
#include "xla/python/ifrt/host_callback.h"
#include "xla/python/ifrt/mpmd_executable.h"
#include "xla/python/ifrt/user_context.h"
#include "xla/python/ifrt_proxy/client/executable.h"
#include "xla/python/ifrt_proxy/client/rpc_helper.h"
#include "xla/tsl/concurrency/future.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace ifrt {
namespace proxy {

class MpmdLoadedExecutable final
    : public llvm::RTTIExtends<MpmdLoadedExecutable,
                               xla::ifrt::MpmdLoadedExecutable> {
 public:
  MpmdLoadedExecutable(
      xla::ifrt::Client* client, std::shared_ptr<RpcHelper> rpc_helper,
      uint64_t handle, std::string name, int num_devices,
      std::optional<DeviceListRef> devices,
      std::vector<xla::ifrt::Device*> addressable_devices,
      absl::StatusOr<
          absl::flat_hash_map<std::string, std::vector<xla::ifrt::Device*>>>
          mpmd_addressable_devices,
      absl::StatusOr<std::optional<std::string>> fingerprint,
      std::vector<tsl::RCReference<xla::ifrt::LoadedHostCallback>>
          loaded_host_callbacks,
      std::vector<uint64_t> loaded_host_callback_handles);

  ~MpmdLoadedExecutable() override;

  xla::ifrt::Client* client() const override {
    return loaded_executable_->client();
  }
  absl::string_view name() const override { return loaded_executable_->name(); }
  absl::StatusOr<std::optional<std::string>> Fingerprint() const override {
    return loaded_executable_->Fingerprint();
  }
  absl::StatusOr<std::shared_ptr<const xla::ifrt::ExecutableVersion>>
  executable_version() const override {
    return loaded_executable_->executable_version();
  }
  absl::StatusOr<std::string> Serialize() const override {
    return loaded_executable_->Serialize();
  }
  absl::StatusOr<std::string> GetHumanReadableProgramText() const override {
    return loaded_executable_->GetHumanReadableProgramText();
  }
  xla::ifrt::UserContextRef user_context() const override {
    return loaded_executable_->user_context();
  }

  int num_devices() const override { return loaded_executable_->num_devices(); }
  int64_t SizeOfGeneratedCodeInBytes() const override {
    return loaded_executable_->SizeOfGeneratedCodeInBytes();
  }
  absl::StatusOr<CompiledMemoryStats> GetCompiledMemoryStats() const override {
    return loaded_executable_->GetCompiledMemoryStats();
  }

  std::optional<std::vector<OpSharding>> GetParameterShardings()
      const override {
    return loaded_executable_->GetParameterShardings();
  }
  absl::StatusOr<absl::Span<const int>> GetDonatableInputIndices()
      const override {
    return loaded_executable_->GetDonatableInputIndices();
  }
  std::optional<std::vector<OpSharding>> GetOutputShardings() const override {
    return loaded_executable_->GetOutputShardings();
  }
  absl::StatusOr<std::vector<std::shared_ptr<const xla::PjRtLayout>>>
  GetParameterLayouts() const override {
    return loaded_executable_->GetParameterLayouts();
  }
  absl::StatusOr<std::vector<std::shared_ptr<const xla::PjRtLayout>>>
  GetOutputLayouts() const override {
    return loaded_executable_->GetOutputLayouts();
  }
  absl::StatusOr<std::vector<std::vector<absl::string_view>>>
  GetOutputMemoryKinds() const override {
    return loaded_executable_->GetOutputMemoryKinds();
  }
  absl::StatusOr<std::vector<std::shared_ptr<HloModule>>> GetHloModules()
      const override {
    return loaded_executable_->GetHloModules();
  }

  absl::StatusOr<xla::ifrt::AttributeMap> GetCostAnalysis() const override {
    return loaded_executable_->GetCostAnalysis();
  }

  absl::StatusOr<LoadedExecutable::ExecuteResult> Execute(
      absl::Span<xla::ifrt::ArrayRef> args, const ExecuteOptions& options,
      std::optional<xla::ifrt::DeviceListRef> devices) override {
    return loaded_executable_->Execute(args, options, devices);
  }

  std::optional<DeviceListRef> devices() const override {
    return loaded_executable_->devices();
  };
  absl::Span<xla::ifrt::Device* const> addressable_devices() const override {
    return loaded_executable_->addressable_devices();
  }

  absl::StatusOr<
      absl::flat_hash_map<std::string, absl::Span<xla::ifrt::Device* const>>>
  GetMpmdAddressableDevices() const override;
  absl::StatusOr<absl::flat_hash_map<std::string, xla::CompiledMemoryStats>>
  GetMpmdCompiledMemoryStats() const override;
  absl::StatusOr<absl::flat_hash_map<
      std::string, std::vector<std::shared_ptr<xla::HloModule>>>>
  GetMpmdHloModules() const override;
  absl::StatusOr<absl::flat_hash_map<std::string, xla::ifrt::AttributeMap>>
  GetMpmdCostAnalysis() const override;

  static char ID;  // NOLINT

 private:
  struct MpmdMetadata {
    absl::StatusOr<absl::flat_hash_map<std::string, CompiledMemoryStats>>
        mpmd_compiled_memory_stats;
  };
  std::shared_ptr<RpcHelper> rpc_helper_;
  const uint64_t handle_;
  mutable tsl::Future<std::shared_ptr<MpmdMetadata>> mpmd_metadata_future_;
  std::unique_ptr<ifrt::proxy::LoadedExecutable> loaded_executable_;
  const absl::StatusOr<
      absl::flat_hash_map<std::string, std::vector<xla::ifrt::Device*>>>
      mpmd_addressable_devices_;
  mutable absl::Mutex mpmd_cost_analysis_mu_;
  mutable std::optional<
      absl::StatusOr<absl::flat_hash_map<std::string, xla::ifrt::AttributeMap>>>
      mpmd_cost_analysis_response_ ABSL_GUARDED_BY(mpmd_cost_analysis_mu_);
};

}  // namespace proxy
}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_IFRT_PROXY_CLIENT_MPMD_EXECUTABLE_H_
