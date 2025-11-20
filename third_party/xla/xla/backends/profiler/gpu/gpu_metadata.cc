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

#include "xla/backends/profiler/gpu/gpu_metadata.h"

#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "third_party/gpus/cuda/include/cuda.h"
#include "third_party/gpus/cuda/include/cuda_runtime_api.h"
#include "third_party/gpus/cuda/include/driver_types.h"
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

namespace {
void AddCudaVersion(XPlaneBuilder& xp, StatType stat_type, int version) {
  if (version <= 0) {
    return;
  }
  XStatMetadata* stat_metadata =
      xp.GetOrCreateStatMetadata(GetStatTypeStr(stat_type));
  xp.AddStatValue(*stat_metadata, absl::StrCat(version));
}

void AddCudaRuntimeVersion(XPlaneBuilder& xp) {
  int runtime_version = 0;
  cudaError_t err = cudaRuntimeGetVersion(&runtime_version);
  if (err == cudaSuccess) {
    AddCudaVersion(xp, StatType::kMetadataCudaRuntimeVersion, runtime_version);
  } else {
    VLOG(1) << "Could not get CUDA runtime version: "
            << cudaGetErrorString(err);
  }
}

void AddCudaDriverVersion(XPlaneBuilder& xp) {
  int driver_version = 0;
  cudaError_t err = cudaDriverGetVersion(&driver_version);
  if (err == cudaSuccess) {
    AddCudaVersion(xp, StatType::kMetadataCudaDriverVersion, driver_version);
  } else {
    VLOG(1) << "Could not get CUDA driver version: " << cudaGetErrorString(err);
  }
}
}  // namespace

void AddGpuMetadata(XSpace* space) {
  XPlane* plane = FindOrAddMutablePlaneWithName(space, kMetadataPlaneName);
  XPlaneBuilder xp(plane);
  AddCudaVersion(xp, StatType::kMetadataCudaVersion, CUDA_VERSION);
  AddCudaRuntimeVersion(xp);
  AddCudaDriverVersion(xp);
}
}  // namespace profiler
}  // namespace xla
