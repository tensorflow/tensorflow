/* Copyright 2023 The OpenXLA Authors.

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

#include "xla/backends/profiler/plugin/plugin_tracer_impl.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "xla/backends/profiler/plugin/profiler_c_api.h"
#include "xla/backends/profiler/plugin/profiler_error.h"
#include "tsl/platform/logging.h"
#include "tsl/profiler/lib/profiler_collection.h"
#include "tsl/profiler/lib/profiler_factory.h"
#include "tsl/profiler/protobuf/profiler_options.pb.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace xla {
namespace profiler {

PLUGIN_Profiler_Error* PLUGIN_Profiler_Create(
    PLUGIN_Profiler_Create_Args* args) {
  VLOG(1) << "Creating plugin profiler";
  auto profiler = std::make_unique<PLUGIN_Profiler>();
  profiler->stopped = true;
  tensorflow::ProfileOptions options;
  options.ParseFromArray(args->options, args->options_size);
  profiler->impl = std::make_unique<tsl::profiler::ProfilerCollection>(
      tsl::profiler::CreateProfilers(options));

  args->profiler = profiler.release();
  return nullptr;
}

PLUGIN_Profiler_Error* PLUGIN_Profiler_Destroy(
    PLUGIN_Profiler_Destroy_Args* args) {
  VLOG(1) << "Destroying plugin profiler";
  if (args->profiler != nullptr) {
    delete args->profiler;
  }
  return nullptr;
}

PLUGIN_Profiler_Error* PLUGIN_Profiler_Start(PLUGIN_Profiler_Start_Args* args) {
  VLOG(1) << "Starting profiler";
  if (!args->profiler->stopped) {
    VLOG(1) << "Profiler is already started";
    return nullptr;
  }
  args->profiler->byte_size = 0;
  PLUGIN_PROFILER_RETURN_IF_ERROR(args->profiler->impl->Start());
  args->profiler->stopped = false;
  return nullptr;
}

PLUGIN_Profiler_Error* PLUGIN_Profiler_Stop(PLUGIN_Profiler_Stop_Args* args) {
  VLOG(1) << "Stopping profiler";
  if (args->profiler->stopped) {
    VLOG(1) << "Profiler is already stopped";
    return nullptr;
  }
  PLUGIN_PROFILER_RETURN_IF_ERROR(args->profiler->impl->Stop());
  args->profiler->stopped = false;
  return nullptr;
}

PLUGIN_Profiler_Error* PLUGIN_Profiler_CollectData(
    PLUGIN_Profiler_CollectData_Args* args) {
  VLOG(1) << "Collecting data from profiler";
  tensorflow::profiler::XSpace space;
  if (!args->profiler->space) {
    VLOG(1) << "TpuProfiler CollectData";
    PLUGIN_PROFILER_RETURN_IF_ERROR(args->profiler->impl->CollectData(&space));
    args->profiler->byte_size = space.ByteSizeLong();
    VLOG(2) << "TpuProfiler CollectData: Number of XPlanes: "
            << space.planes_size();
  }

  const size_t profiler_data_size = space.ByteSizeLong();
  if (args->buffer == nullptr) {
    args->profiler->buffer =
        std::make_unique<std::vector<uint8_t>>(profiler_data_size + 1);
    space.SerializeToArray(args->profiler->buffer->data(), profiler_data_size);

    args->buffer_size_in_bytes = args->profiler->buffer->size();
    args->buffer = args->profiler->buffer->data();
    return nullptr;
  }
  return nullptr;
}
}  // namespace profiler
}  // namespace xla
