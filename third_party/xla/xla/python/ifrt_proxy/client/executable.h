/*
 * Copyright 2023 The OpenXLA Authors.
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

#ifndef XLA_PYTHON_IFRT_PROXY_CLIENT_EXECUTABLE_H_
#define XLA_PYTHON_IFRT_PROXY_CLIENT_EXECUTABLE_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/node_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/Support/ExtensibleRTTI.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/layout.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/executable.h"
#include "xla/python/ifrt/future.h"
#include "xla/python/ifrt/host_callback.h"
#include "xla/python/ifrt_proxy/client/rpc_helper.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace ifrt {
namespace proxy {

class LoadedExecutable final
    : public llvm::RTTIExtends<LoadedExecutable, xla::ifrt::LoadedExecutable> {
 public:
  LoadedExecutable(xla::ifrt::Client* client,
                   std::shared_ptr<RpcHelper> rpc_helper, uint64_t handle,
                   std::string name, int num_devices,
                   std::vector<xla::ifrt::LoadedExecutable::LogicalDeviceIds>
                       addressable_device_logical_device_ids,
                   std::vector<xla::ifrt::Device*> addressable_devices,
                   absl::StatusOr<std::optional<std::string>> fingerprint,
                   Future<> ready_future,
                   std::vector<tsl::RCReference<xla::ifrt::LoadedHostCallback>>
                       loaded_host_callbacks,
                   std::vector<uint64_t> loaded_host_callback_handles);

  ~LoadedExecutable() override;

  xla::ifrt::Client* client() const override;
  absl::string_view name() const override;
  absl::StatusOr<std::optional<std::string>> Fingerprint() const override;
  absl::StatusOr<std::string> Serialize() const override;
  Future<> GetReadyFuture() const override;

  int num_devices() const override;
  int64_t SizeOfGeneratedCodeInBytes() const override;
  absl::StatusOr<CompiledMemoryStats> GetCompiledMemoryStats() const override;

  std::optional<std::vector<OpSharding>> GetParameterShardings() const override;
  std::optional<std::vector<OpSharding>> GetOutputShardings() const override;
  absl::StatusOr<std::vector<std::unique_ptr<Layout>>> GetParameterLayouts()
      const override;
  absl::StatusOr<std::vector<std::unique_ptr<Layout>>> GetOutputLayouts()
      const override;
  absl::StatusOr<std::vector<std::vector<absl::string_view>>>
  GetOutputMemoryKinds() const override;
  absl::StatusOr<std::vector<std::shared_ptr<HloModule>>> GetHloModules()
      const override;

  absl::StatusOr<absl::flat_hash_map<std::string,
                                     xla::ifrt::Executable::CostAnalysisValue>>
  GetCostAnalysis() const override;

  absl::StatusOr<ExecuteResult> Execute(
      absl::Span<tsl::RCReference<xla::ifrt::Array>> args,
      const ExecuteOptions& options,
      std::optional<DeviceList> devices) override;

  Future<> Delete() override;
  bool IsDeleted() const override;

  absl::Span<const LogicalDeviceIds> addressable_device_logical_ids()
      const override;
  absl::Span<xla::ifrt::Device* const> addressable_devices() const override;

  static char ID;  // NOLINT

 private:
  struct Metadata {
    std::optional<std::vector<xla::OpSharding>> parameter_shardings;
    std::optional<std::vector<xla::OpSharding>> output_shardings;

    absl::StatusOr<std::vector<xla::Layout>> parameter_layouts;
    absl::StatusOr<std::vector<xla::Layout>> output_layouts;

    // Elements in `output_memory_kinds` point to elements in `memory_kinds`.
    // Required since `GetOutputMemoryKinds()` returns `absl::string_view`.
    // `memory_kinds` uses `absl::node_hash_set` for pointer stability.
    absl::node_hash_set<std::string> memory_kinds;
    absl::StatusOr<std::vector<std::vector<absl::string_view>>>
        output_memory_kinds;
  };

  void PollLoadedHostCallback(
      uint64_t handle,
      tsl::RCReference<xla::ifrt::LoadedHostCallback> loaded_host_callback);

  xla::ifrt::Client* client_;
  std::shared_ptr<RpcHelper> rpc_helper_;

  const uint64_t handle_;
  const std::string name_;
  const int num_devices_;
  const std::vector<xla::ifrt::LoadedExecutable::LogicalDeviceIds>
      addressable_device_logical_device_ids_;
  const std::vector<xla::ifrt::Device*> addressable_devices_;
  const absl::StatusOr<std::optional<std::string>> fingerprint_;
  const Future<> ready_future_;

  // Metadata queried when the executable is created. Declared as `mutable`
  // since `Future::Await()` is not const.
  mutable Future<std::shared_ptr<Metadata>> metadata_future_;
};

}  // namespace proxy
}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_IFRT_PROXY_CLIENT_EXECUTABLE_H_
