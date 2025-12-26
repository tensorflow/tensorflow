/* Copyright 2018 The OpenXLA Authors.

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

#include <memory>
#include <vector>

#include "absl/status/status.h"
#include "xla/backends/profiler/cpu/metadata_utils.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/xla_debug_info_manager.h"
#include "xla/tsl/profiler/utils/xplane_builder.h"
#include "xla/tsl/profiler/utils/xplane_schema.h"
#include "xla/tsl/profiler/utils/xplane_utils.h"
#include "tsl/profiler/lib/profiler_factory.h"
#include "tsl/profiler/lib/profiler_interface.h"
#include "tsl/profiler/protobuf/profiler_options.pb.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace xla {
namespace profiler {
namespace {

// MetadataCollector collect miscellaneous metadata for xprof, e.g. HLO protos
// from XLA runtime etc.
//
// Thread-safety: This class is go/thread-compatible.
class MetadataCollector : public tsl::profiler::ProfilerInterface {
 public:
  explicit MetadataCollector(const tensorflow::ProfileOptions& options)
      : options_(options) {}

  absl::Status Start() override {
    if (!trace_active_) {
      xla::XlaDebugInfoManager::Get()->StartTracing();
      trace_active_ = true;
    }
    return absl::OkStatus();
  }

  absl::Status Stop() override {
    if (trace_active_) {
      xla::XlaDebugInfoManager::Get()->StopTracing(&debug_info_);
      trace_active_ = false;
    }
    return absl::OkStatus();
  }

  absl::Status CollectData(tsl::profiler::XSpace* space) override {
    tsl::profiler::XPlane* plane = tsl::profiler::FindOrAddMutablePlaneWithName(
        space, tsl::profiler::kMetadataPlaneName);
    tsl::profiler::XPlaneBuilder xp(plane);

    LOG(INFO) << "MetadataCollector CollectData: JAX version: "
              << options_.jax_version()
              << ", JAXlib version: " << options_.jaxlib_version();

    if (!options_.jax_version().empty()) {
      tsl::profiler::XStatMetadata* jax_version_metadata =
          xp.GetOrCreateStatMetadata(tsl::profiler::GetStatTypeStr(
              tsl::profiler::StatType::kMetadataJaxVersion));
      xp.AddStatValue(*jax_version_metadata, options_.jax_version());
    } else {
      tsl::profiler::XStatMetadata* jax_version_metadata =
          xp.GetOrCreateStatMetadata(tsl::profiler::GetStatTypeStr(
              tsl::profiler::StatType::kMetadataJaxVersion));
      xp.AddStatValue(*jax_version_metadata, "unknown");
    }
    if (!options_.jaxlib_version().empty()) {
      tsl::profiler::XStatMetadata* jaxlib_version_metadata =
          xp.GetOrCreateStatMetadata(tsl::profiler::GetStatTypeStr(
              tsl::profiler::StatType::kMetadataJaxlibVersion));
      xp.AddStatValue(*jaxlib_version_metadata, options_.jaxlib_version());
    } else {
      tsl::profiler::XStatMetadata* jaxlib_version_metadata =
          xp.GetOrCreateStatMetadata(tsl::profiler::GetStatTypeStr(
              tsl::profiler::StatType::kMetadataJaxlibVersion));
      xp.AddStatValue(*jaxlib_version_metadata, "unknown");
    }

    if (!debug_info_.empty()) {
      MetadataXPlaneBuilder metadata_plane(plane);
      for (auto& hlo_proto : debug_info_) {
        metadata_plane.AddHloProto(hlo_proto->hlo_module().id(), *hlo_proto);
        hlo_proto.reset();
      }
      debug_info_.clear();
    }
    return absl::OkStatus();
  }

 private:
  std::vector<std::unique_ptr<xla::HloProto>> debug_info_;
  tensorflow::ProfileOptions options_;
  bool trace_active_ = false;

  MetadataCollector(const MetadataCollector&) = delete;
  void operator=(const MetadataCollector&) = delete;
};

std::unique_ptr<tsl::profiler::ProfilerInterface> CreatMetadataCollector(
    const tensorflow::ProfileOptions& options) {
  return options.enable_hlo_proto()
             ? std::make_unique<MetadataCollector>(options)
             : nullptr;
}

}  // namespace

auto register_metadata_collector_factory = [] {
  RegisterProfilerFactory(&CreatMetadataCollector);
  return 0;
}();

}  // namespace profiler
}  // namespace xla
