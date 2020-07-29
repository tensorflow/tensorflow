/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#include <string>
#include <utility>
#include <vector>

#include "tensorflow/compiler/xla/service/gpu/gpu_debug_info_manager.h"
#include "tensorflow/compiler/xla/service/hlo.pb.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/profiler/internal/profiler_factory.h"
#include "tensorflow/core/profiler/internal/profiler_interface.h"
#include "tensorflow/core/profiler/profiler_options.pb.h"
#include "tensorflow/core/profiler/protobuf/xplane.pb.h"
#include "tensorflow/core/profiler/utils/xplane_builder.h"
#include "tensorflow/core/profiler/utils/xplane_schema.h"
#include "tensorflow/core/profiler/utils/xplane_utils.h"
#include "tensorflow/core/protobuf/config.pb.h"

namespace tensorflow {
namespace profiler {
namespace {

// MetadataCollector collect miscellaneous metadata for xprof, e.g. HLO protos
// from XLA runtime etc.
//
// Thread-safety: This class is go/thread-compatible.
class MetadataCollector : public ProfilerInterface {
 public:
  MetadataCollector() = default;

  Status Start() override {
    if (!trace_active_) {
      xla::gpu::GpuDebugInfoManager::Get()->StartTracing();
      trace_active_ = true;
    }
    return Status::OK();
  }

  Status Stop() override {
    if (trace_active_) {
      xla::gpu::GpuDebugInfoManager::Get()->StopTracing(&debug_info_);
      trace_active_ = false;
    }
    return Status::OK();
  }

  Status CollectData(RunMetadata* run_metadata) override {
    return Status::OK();  // legacy session is not supported.
  }

  Status CollectData(XSpace* space) override {
    if (!debug_info_.empty()) {
      XPlane* plane = FindOrAddMutablePlaneWithName(space, kMetadataPlaneName);
      XPlaneBuilder xplane(plane);
      const XStatMetadata& hlo_proto_stat =
          *xplane.GetOrCreateStatMetadata(kHloProto);
      for (auto& p : debug_info_) {
        xplane.AddStatValue(hlo_proto_stat, *p.hlo_proto);
        p.hlo_proto.reset();
      }
      debug_info_.clear();
    }
    return Status::OK();
  }

 private:
  std::vector<xla::gpu::GpuModuleDebugInfo> debug_info_;
  bool trace_active_ = false;

  TF_DISALLOW_COPY_AND_ASSIGN(MetadataCollector);
};

std::unique_ptr<ProfilerInterface> CreatMetadataCollector(
    const ProfileOptions& options) {
  return options.enable_hlo_proto() ? absl::make_unique<MetadataCollector>()
                                    : nullptr;
}

}  // namespace

auto register_metadata_collector_factory = [] {
  RegisterProfilerFactory(&CreatMetadataCollector);
  return 0;
}();

}  // namespace profiler
}  // namespace tensorflow
