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

#include "absl/strings/string_view.h"
#include "xla/backends/profiler/plugin/plugin_metadata.h"
#include "xla/backends/profiler/plugin/profiler_c_api.h"
#include "xla/backends/profiler/plugin/profiler_error.h"
#include "xla/tsl/platform/logging.h"
#include "tsl/profiler/lib/profiler_collection.h"
#include "tsl/profiler/lib/profiler_factory.h"
#include "tsl/profiler/protobuf/profiler_options.pb.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace xla {
namespace profiler {
namespace {}  // namespace

PLUGIN_Profiler_Error* PLUGIN_Profiler_Create(
    PLUGIN_Profiler_Create_Args* args) {
  VLOG(1) << "Creating plugin profiler";
  auto profiler = std::make_unique<PLUGIN_Profiler>();
  profiler->stopped = true;
  tensorflow::ProfileOptions options;
  options.ParseFromString(absl::string_view(args->options, args->options_size));
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
  AddPluginMetadata();
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
        std::make_unique<std::vector<uint8_t>>(profiler_data_size);
    space.SerializeToArray(args->profiler->buffer->data(), profiler_data_size);

    args->buffer_size_in_bytes = args->profiler->buffer->size();
    args->buffer = args->profiler->buffer->data();
    return nullptr;
  }
  space.SerializeToArray(const_cast<uint8_t*>(args->buffer),
                         profiler_data_size);
  args->buffer_size_in_bytes = profiler_data_size;
  return nullptr;
}

PLUGIN_Profiler_Error* PLUGIN_Profiler_Consume(
    PLUGIN_Profiler_Consume_Args* args) {
  VLOG(1) << "Consuming data from profiler";
  auto status_or = args->profiler->impl->Consume();
  if (!status_or.ok()) {
    return new PLUGIN_Profiler_Error{status_or.status()};
  }
  auto result = std::make_unique<PLUGIN_Profiler_ConsumeResult>();
  result->consume_result = *std::move(status_or);
  args->result = result.release();
  return nullptr;
}

PLUGIN_Profiler_Error* PLUGIN_Profiler_Serialize(
    PLUGIN_Profiler_Serialize_Args* args) {
  VLOG(1) << "Serializing data from profiler";
  tensorflow::profiler::XSpace space;
  absl::Status s = args->profiler->impl->Serialize(
      std::move(args->consume_result->consume_result.data), &space);
  if (!s.ok()) {
    return new PLUGIN_Profiler_Error{s};
  }
  const size_t profiler_data_size = space.ByteSizeLong();
  args->profiler->buffer =
      std::make_unique<std::vector<uint8_t>>(profiler_data_size);
  space.SerializeToArray(args->profiler->buffer->data(), profiler_data_size);
  args->serialized_bytes =
      reinterpret_cast<const char*>(args->profiler->buffer->data());
  args->serialized_size = args->profiler->buffer->size();
  return nullptr;
}

void PLUGIN_Profiler_ConsumeResult_Destroy(
    PLUGIN_Profiler_ConsumeResult* result) {
  VLOG(1) << "Destroying consume result";
  if (result != nullptr) {
    delete result;
  }
}

}  // namespace profiler
}  // namespace xla
