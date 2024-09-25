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
#include "xla/backends/profiler/plugin/plugin_tracer.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "xla/backends/profiler/plugin/profiler_c_api.h"
#include "tsl/platform/logging.h"
#include "tsl/profiler/protobuf/profiler_options.pb.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace xla {
namespace profiler {

using tensorflow::profiler::XPlane;
using tensorflow::profiler::XSpace;

namespace {

#define RETURN_STATUS_IF_PLUGIN_PROFILER_ERROR(expr, c_api)                  \
  do {                                                                       \
    PLUGIN_Profiler_Error* error = (expr);                                   \
    std::unique_ptr<PLUGIN_Profiler_Error,                                   \
                    std::function<void(PLUGIN_Profiler_Error*)>>             \
        _error(error, MakeErrorDeleter(c_api));                              \
    absl::Status _status = PluginProfilerErrorToStatus(_error.get(), c_api); \
    if (!_status.ok()) {                                                     \
      return _status;                                                        \
    }                                                                        \
  } while (false)

std::function<void(PLUGIN_Profiler_Error*)> MakeErrorDeleter(
    const PLUGIN_Profiler_Api* api) {
  return [api](PLUGIN_Profiler_Error* error) -> void {
    PLUGIN_Profiler_Error_Destroy_Args destroy_args;
    destroy_args.struct_size = PLUGIN_Profiler_Error_Destroy_Args_STRUCT_SIZE;
    destroy_args.priv = nullptr;
    destroy_args.error = error;

    api->error_destroy(&destroy_args);
  };
}

absl::string_view GetPluginProfilerErrorMessage(
    const PLUGIN_Profiler_Error* error, const PLUGIN_Profiler_Api* api) {
  PLUGIN_Profiler_Error_Message_Args message_args;
  message_args.struct_size = PLUGIN_Profiler_Error_Message_Args_STRUCT_SIZE;
  message_args.priv = nullptr;
  message_args.error = error;
  api->error_message(&message_args);
  return absl::string_view(message_args.message, message_args.message_size);
}

absl::StatusCode PluginProfilerErrorToStatusCode(
    const PLUGIN_Profiler_Error* error, const PLUGIN_Profiler_Api* api) {
  PLUGIN_Profiler_Error_GetCode_Args args;
  args.struct_size = PLUGIN_Profiler_Error_GetCode_Args_STRUCT_SIZE;
  args.priv = nullptr;
  args.error = error;
  PLUGIN_Profiler_Error* get_code_error = api->error_get_code(&args);
  if (get_code_error != nullptr) {
    std::unique_ptr<PLUGIN_Profiler_Error,
                    std::function<void(PLUGIN_Profiler_Error*)>>
        error_ptr(get_code_error, MakeErrorDeleter(api));
    LOG(FATAL) << GetPluginProfilerErrorMessage(error_ptr.get(), api);
  }
  return static_cast<absl::StatusCode>(args.code);
}

absl::Status PluginProfilerErrorToStatus(const PLUGIN_Profiler_Error* error,
                                         const PLUGIN_Profiler_Api* api) {
  if (error == nullptr) {
    return absl::OkStatus();
  }
  return absl::Status(PluginProfilerErrorToStatusCode(error, api),
                      GetPluginProfilerErrorMessage(error, api));
}

}  // namespace

PluginTracer::PluginTracer(const PLUGIN_Profiler_Api* profiler_api,
                           const tensorflow::ProfileOptions& options) {
  if (profiler_api == nullptr) {
    LOG(ERROR) << "The plugin does not implement a profiler interface. This "
                  "could restrict the profiling capabilities.";
    return;
  }
  if (profiler_api->struct_size != PLUGIN_Profiler_Api_STRUCT_SIZE) {
    LOG(ERROR) << "Unexpected PLUGIN_Profiler_Api size: expected "
               << PLUGIN_Profiler_Api_STRUCT_SIZE << ", got "
               << profiler_api->struct_size
               << ". Check installed software versions.";
    return;
  }
  profiler_api_ = profiler_api;

  PLUGIN_Profiler_Create_Args args;
  std::string options_str = options.SerializeAsString();
  args.options = options_str.c_str();
  args.options_size = options_str.size();
  PLUGIN_Profiler_Error* error = profiler_api_->create(&args);
  if (error != nullptr) {
    std::unique_ptr<PLUGIN_Profiler_Error,
                    std::function<void(PLUGIN_Profiler_Error*)>>
        error_ptr(error, MakeErrorDeleter(profiler_api_));
    LOG(ERROR) << GetPluginProfilerErrorMessage(error_ptr.get(), profiler_api_);
    return;
  }

  profiler_ = args.profiler;
}

PluginTracer::~PluginTracer() {
  PLUGIN_Profiler_Destroy_Args args;
  args.profiler = profiler_;
  PLUGIN_Profiler_Error* error = profiler_api_->destroy(&args);
  if (error != nullptr) {
    std::unique_ptr<PLUGIN_Profiler_Error,
                    std::function<void(PLUGIN_Profiler_Error*)>>
        error_ptr(error, MakeErrorDeleter(profiler_api_));
    LOG(ERROR) << GetPluginProfilerErrorMessage(error_ptr.get(), profiler_api_);
    return;
  }
}

absl::Status PluginTracer::Start() {
  PLUGIN_Profiler_Start_Args args;
  args.profiler = profiler_;
  RETURN_STATUS_IF_PLUGIN_PROFILER_ERROR(profiler_api_->start(&args),
                                         profiler_api_);
  return absl::OkStatus();
}

absl::Status PluginTracer::Stop() {
  PLUGIN_Profiler_Stop_Args args;
  args.profiler = profiler_;
  RETURN_STATUS_IF_PLUGIN_PROFILER_ERROR(profiler_api_->stop(&args),
                                         profiler_api_);
  return absl::OkStatus();
}

absl::Status PluginTracer::CollectData(XSpace* space) {
  PLUGIN_Profiler_CollectData_Args args;
  args.profiler = profiler_;
  args.buffer = nullptr;
  RETURN_STATUS_IF_PLUGIN_PROFILER_ERROR(profiler_api_->collect_data(&args),
                                         profiler_api_);
  if (args.buffer_size_in_bytes > 0) {
    std::vector<uint8_t> buffer(args.buffer_size_in_bytes);
    XSpace xspace;
    xspace.ParseFromArray(args.buffer, args.buffer_size_in_bytes);
    for (XPlane& tpu_plane : *xspace.mutable_planes()) {
      XPlane* plane = space->add_planes();
      plane->Swap(&tpu_plane);
    }
  }
  return absl::OkStatus();
}

}  // namespace profiler
}  // namespace xla
