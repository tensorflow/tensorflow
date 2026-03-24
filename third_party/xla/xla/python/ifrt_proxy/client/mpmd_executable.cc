// Copyright 2023 The OpenXLA Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "xla/python/ifrt_proxy/client/mpmd_executable.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/array_spec.h"
#include "xla/python/ifrt/attribute_map.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/device_list.h"
#include "xla/python/ifrt/executable.h"
#include "xla/python/ifrt/host_callback.h"
#include "xla/python/ifrt/user_context_status_util.h"
#include "xla/python/ifrt_proxy/client/executable.h"
#include "xla/python/ifrt_proxy/client/rpc_helper.h"
#include "xla/python/ifrt_proxy/common/ifrt_service.pb.h"
#include "xla/python/ifrt_proxy/common/versions.h"
#include "xla/tsl/concurrency/future.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "xla/tsl/platform/status_to_from_proto.h"
#include "xla/xla_data.pb.h"
#include "tsl/profiler/lib/traceme.h"

namespace xla {
namespace ifrt {
namespace proxy {

MpmdLoadedExecutable::MpmdLoadedExecutable(
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
    std::vector<uint64_t> loaded_host_callback_handles)
    : rpc_helper_(rpc_helper),
      handle_(handle),
      mpmd_addressable_devices_(std::move(mpmd_addressable_devices)) {
  loaded_executable_ = std::make_unique<ifrt::proxy::LoadedExecutable>(
      client, rpc_helper, handle, std::move(name), num_devices,
      std::move(devices), std::move(addressable_devices),
      std::move(fingerprint), std::move(loaded_host_callbacks),
      std::move(loaded_host_callback_handles));

  tsl::profiler::TraceMe traceme_ifrt_entrypoint(
      "IfrtProxyEntrypointMpmdLoadedExecutableCreate");
  auto [promise, future] = tsl::MakePromise<std::shared_ptr<MpmdMetadata>>();
  mpmd_metadata_future_ = std::move(future);

  auto req = std::make_unique<LoadedExecutableMpmdMetadataRequest>();
  req->set_mpmd_loaded_executable_handle(handle);

  auto on_done =
      [promise = std::move(promise)](
          absl::StatusOr<std::shared_ptr<LoadedExecutableMpmdMetadataResponse>>
              response) mutable {
        if (!response.ok()) {
          LOG(ERROR) << "LoadedExecutableMpmdMetadata: Got "
                     << response.status();
          promise.Set(response.status());
          return;
        }

        auto info = std::make_shared<MpmdMetadata>();

        if (response.value()->has_mpmd_compiled_memory_stats()) {
          absl::flat_hash_map<std::string, xla::CompiledMemoryStats>
              mpmd_compiled_memory_stats;
          for (const auto& [name, stats_proto] :
               response.value()
                   ->mpmd_compiled_memory_stats()
                   .compiled_memory_stats()) {
            mpmd_compiled_memory_stats.insert(
                {name, xla::CompiledMemoryStats::FromProto(stats_proto)});
          }
          info->mpmd_compiled_memory_stats =
              std::move(mpmd_compiled_memory_stats);
        } else if (response.value()->has_mpmd_compiled_memory_stats_error()) {
          info->mpmd_compiled_memory_stats =
              xla::ifrt::ReattachUserContextRefs(tsl::StatusFromProto(
                  response.value()->mpmd_compiled_memory_stats_error()));
        } else {
          info->mpmd_compiled_memory_stats = absl::InternalError(
              "IFRT Proxy server did not return mpmd compiled memory stats");
        }

        promise.Set(std::move(info));
      };
  rpc_helper_->LoadedExecutableMpmdMetadata(std::move(req))
      .OnReady(std::move(on_done));
}

MpmdLoadedExecutable::~MpmdLoadedExecutable() {
  tsl::profiler::TraceMe traceme_ifrt_entrypoint(
      "IfrtProxyEntrypointMpmdLoadedExecutableDestruct");
}

absl::StatusOr<
    absl::flat_hash_map<std::string, absl::Span<xla::ifrt::Device* const>>>
MpmdLoadedExecutable::GetMpmdAddressableDevices() const {
  if (rpc_helper_->protocol_version() <
      protocol_version::kMpmdLoadedExecutableMethods) {
    return absl::UnimplementedError(
        "LoadedExecutable::GetMpmdAddressableDevices() is unimplemented by "
        "IFRT proxy");
  }

  if (!mpmd_addressable_devices_.ok()) {
    return mpmd_addressable_devices_.status();
  }

  absl::flat_hash_map<std::string, absl::Span<xla::ifrt::Device* const>>
      devices_map_with_spans;
  devices_map_with_spans.reserve(mpmd_addressable_devices_->size());
  for (const auto& [mesh_name, device_vector] : *mpmd_addressable_devices_) {
    devices_map_with_spans[mesh_name] = absl::MakeConstSpan(device_vector);
  }

  return devices_map_with_spans;
}

absl::StatusOr<absl::flat_hash_map<std::string, xla::CompiledMemoryStats>>
MpmdLoadedExecutable::GetMpmdCompiledMemoryStats() const {
  if (rpc_helper_->protocol_version() <
      protocol_version::kMpmdLoadedExecutableMethods) {
    return absl::UnimplementedError(
        "LoadedExecutable::GetMpmdCompiledMemoryStats() is unimplemented by "
        "IFRT proxy");
  }
  auto info = mpmd_metadata_future_.Await();
  if (!info.ok()) {
    return info.status();
  }
  return (*info)->mpmd_compiled_memory_stats;
}

absl::StatusOr<absl::flat_hash_map<
    std::string, std::vector<std::shared_ptr<xla::HloModule>>>>
MpmdLoadedExecutable::GetMpmdHloModules() const {
  return absl::UnimplementedError(
      "IFRT service does not support LoadedExecutable::GetMpmdHloModules() "
      "since HloModule does not provide stable serialization");
}

absl::StatusOr<absl::flat_hash_map<std::string, xla::ifrt::AttributeMap>>
MpmdLoadedExecutable::GetMpmdCostAnalysis() const {
  if (rpc_helper_->protocol_version() <
      protocol_version::kMpmdLoadedExecutableMethods) {
    return absl::UnimplementedError(
        "LoadedExecutable::GetMpmdCostAnalysis() is unimplemented by IFRT "
        "proxy");
  }
  absl::MutexLock l(mpmd_cost_analysis_mu_);
  if (!mpmd_cost_analysis_response_.has_value()) {
    auto req = std::make_unique<LoadedExecutableMpmdCostAnalysisRequest>();
    req->set_loaded_executable_handle(handle_);

    absl::StatusOr<std::shared_ptr<LoadedExecutableMpmdCostAnalysisResponse>>
        response = rpc_helper_->LoadedExecutableMpmdCostAnalysis(std::move(req))
                       .Await();

    if (!response.ok()) {
      LOG(ERROR) << "LoadedExecutableMpmdCostAnalysis: Got "
                 << response.status();
      mpmd_cost_analysis_response_ = response.status();
      return *mpmd_cost_analysis_response_;
    }
    if (response.ok() && response.value()->has_attributes()) {
      absl::flat_hash_map<std::string, xla::ifrt::AttributeMap> temp_map;
      absl::Status status = absl::OkStatus();
      for (const auto& [name, attributes] :
           response.value()->attributes().attributes()) {
        absl::StatusOr<xla::ifrt::AttributeMap> attr_map =
            xla::ifrt::AttributeMap::FromProto(attributes);
        if (!attr_map.ok()) {
          LOG(ERROR) << "Failed to deserialize AttributeMap for key '" << name
                     << "': " << attr_map.status();
          status = attr_map.status();
          break;
        }
        temp_map.insert({name, *std::move(attr_map)});
      }
      if (!status.ok()) {
        mpmd_cost_analysis_response_ = status;
      } else {
        mpmd_cost_analysis_response_ = std::move(temp_map);
      }
    } else {
      mpmd_cost_analysis_response_ = xla::ifrt::ReattachUserContextRefs(
          tsl::StatusFromProto(response.value()->status()));
    }
  }
  return *mpmd_cost_analysis_response_;
}

char MpmdLoadedExecutable::ID = 0;  // NOLINT

}  // namespace proxy
}  // namespace ifrt
}  // namespace xla
