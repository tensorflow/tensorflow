/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/backends/profiler/plugin/plugin_metadata.h"

#include "absl/base/builddata.h"
#include "absl/strings/str_cat.h"
#include "xla/tsl/profiler/utils/xplane_builder.h"
#include "xla/tsl/profiler/utils/xplane_schema.h"
#include "xla/tsl/profiler/utils/xplane_utils.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace xla {
namespace profiler {

using tensorflow::profiler::XPlane;
using tensorflow::profiler::XSpace;
using tsl::profiler::FindOrAddMutablePlaneWithName;
using tsl::profiler::GetStatTypeStr;
using tsl::profiler::kMetadataPlaneName;
using tsl::profiler::StatType;
using tsl::profiler::XPlaneBuilder;
using tsl::profiler::XStatMetadata;

void AddPluginMetadata(XSpace* space) {
  XPlane* plane = FindOrAddMutablePlaneWithName(space, kMetadataPlaneName);
  XPlaneBuilder xp(plane);
  XStatMetadata* stat_metadata = xp.GetOrCreateStatMetadata(
      GetStatTypeStr(StatType::kMetadataLibtpuVersion));
  xp.AddStatValue(*stat_metadata, absl::StrCat(BuildData::Timestamp(), " cl/",
                                               BuildData::Changelist()));
}
}  // namespace profiler
}  // namespace xla
