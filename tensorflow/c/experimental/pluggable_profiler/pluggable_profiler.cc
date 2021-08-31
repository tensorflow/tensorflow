/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include <vector>

#include "tensorflow/c/experimental/pluggable_device_utils/pluggable_device_utils.h"
#include "tensorflow/c/experimental/pluggable_profiler/pluggable_profiler_internal.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/profiler/lib/profiler_factory.h"
#include "tensorflow/core/profiler/lib/profiler_interface.h"
#include "tensorflow/core/profiler/profiler_options.pb.h"

namespace tensorflow {
namespace profiler {

namespace {

Status ValidateTPProfilerRegistrationParams(
    const TF_ProfilerRegistrationParams& params) {
  VALIDATE_STRUCT_SIZE(TF_ProfilerRegistrationParams, params,
                       TF_PROFILER_REGISTRATION_PARAMS_STRUCT_SIZE);
  VALIDATE_MEMBER(TF_ProfilerRegistrationParams, params, destroy_profiler);
  VALIDATE_MEMBER(TF_ProfilerRegistrationParams, params, destroy_profiler_fns);
  return Status::OK();
}

Status ValidateTPProfiler(const TP_Profiler& profiler) {
  VALIDATE_STRUCT_SIZE(TP_Profiler, profiler, TP_PROFILER_STRUCT_SIZE);
  VALIDATE_MEMBER(TP_Profiler, profiler, type);
  TF_RETURN_IF_ERROR(pluggable_device::ValidateDeviceType(profiler.type));
  return Status::OK();
}

Status ValidateTPProfilerFns(const TP_ProfilerFns& profiler_fns) {
  VALIDATE_STRUCT_SIZE(TP_ProfilerFns, profiler_fns,
                       TF_PROFILER_FNS_STRUCT_SIZE);
  VALIDATE_MEMBER(TP_ProfilerFns, profiler_fns, start);
  VALIDATE_MEMBER(TP_ProfilerFns, profiler_fns, stop);
  VALIDATE_MEMBER(TP_ProfilerFns, profiler_fns, collect_data_xspace);
  return Status::OK();
}

class PluggableProfiler : public tensorflow::profiler::ProfilerInterface {
 public:
  // The caller must have validated profiler_fns and profiler.
  static std::unique_ptr<tensorflow::profiler::ProfilerInterface>
  CreatePluggableProfiler(const ProfileOptions& options, TP_Profiler profiler,
                          TP_ProfilerFns profiler_fns) {
    if (options.device_tracer_level() == 0) {
      return nullptr;
    }
    if (options.device_type() != ProfileOptions::UNSPECIFIED &&
        options.device_type() != ProfileOptions::PLUGGABLE_DEVICE) {
      return nullptr;
    }
    return absl::WrapUnique(new PluggableProfiler(profiler_fns, profiler));
  }

  Status Start() override {
    std::unique_ptr<TF_Status, pluggable_device::TFStatusDeleter> status(
        TF_NewStatus());
    profiler_fns_.start(&profiler_, status.get());
    return tensorflow::StatusFromTF_Status(status.get());
  }

  Status Stop() override {
    std::unique_ptr<TF_Status, pluggable_device::TFStatusDeleter> status(
        TF_NewStatus());
    profiler_fns_.stop(&profiler_, status.get());
    return tensorflow::StatusFromTF_Status(status.get());
  }

  Status CollectData(XSpace* space) override {
    std::unique_ptr<TF_Status, pluggable_device::TFStatusDeleter> status(
        TF_NewStatus());
    // Get size of buffer required for Plugin to serialize XSpace into it.
    size_t size_in_bytes;
    profiler_fns_.collect_data_xspace(&profiler_, /*buffer=*/nullptr,
                                      &size_in_bytes, status.get());

    // Prepare an appropriately sized buffer.
    if (size_in_bytes > 0) {
      std::vector<uint8_t> buffer(size_in_bytes);
      profiler_fns_.collect_data_xspace(&profiler_, buffer.data(),
                                        &size_in_bytes, status.get());
      // Deserialize XSpace from the buffer and return it.
      XSpace plugin_space;
      plugin_space.ParseFromArray(buffer.data(), buffer.size());
      for (XPlane& plugin_plane : *plugin_space.mutable_planes()) {
        XPlane* plane = space->add_planes();
        plane->Swap(&plugin_plane);
      }
    }
    return tensorflow::StatusFromTF_Status(status.get());
  }

 private:
  PluggableProfiler(TP_ProfilerFns profiler_fns, TP_Profiler profiler)
      : profiler_fns_(profiler_fns), profiler_(profiler) {}
  TP_ProfilerFns profiler_fns_;
  TP_Profiler profiler_;
};

class PluggableProfilerFactory {
 public:
  PluggableProfilerFactory(TP_Profiler profiler,
                           void (*destroy_profiler)(TP_Profiler*),
                           TP_ProfilerFns profiler_fns,
                           void (*destroy_profiler_fns)(TP_ProfilerFns*))
      : profiler_(std::move(profiler)),
        destroy_profiler_(destroy_profiler),
        profiler_fns_(std::move(profiler_fns)),
        destroy_profiler_fns_(destroy_profiler_fns) {}

  ~PluggableProfilerFactory() {
    destroy_profiler_(&profiler_);
    destroy_profiler_fns_(&profiler_fns_);
  }

  std::unique_ptr<tensorflow::profiler::ProfilerInterface>
  CreatePluggableProfiler(const ProfileOptions& options) {
    return PluggableProfiler::CreatePluggableProfiler(options, profiler_,
                                                      profiler_fns_);
  }

 private:
  TP_Profiler profiler_{TP_PROFILER_STRUCT_SIZE};
  void (*destroy_profiler_)(TP_Profiler*);
  TP_ProfilerFns profiler_fns_{TP_PROFILER_FNS_STRUCT_SIZE};
  void (*destroy_profiler_fns_)(TP_ProfilerFns*);
};

}  // namespace

Status InitPluginProfiler(TFInitProfilerFn init_fn) {
  TF_ProfilerRegistrationParams params{
      TF_PROFILER_REGISTRATION_PARAMS_STRUCT_SIZE};
  TP_Profiler profiler{TP_PROFILER_STRUCT_SIZE};
  TP_ProfilerFns profiler_fns{TP_PROFILER_FNS_STRUCT_SIZE};
  params.major_version = TP_MAJOR;
  params.minor_version = TP_MINOR;
  params.patch_version = TP_PATCH;
  params.profiler = &profiler;
  params.profiler_fns = &profiler_fns;
  std::unique_ptr<TF_Status, pluggable_device::TFStatusDeleter> status(
      TF_NewStatus());
  init_fn(&params, status.get());
  TF_RETURN_IF_ERROR(tensorflow::StatusFromTF_Status(status.get()));
  TF_RETURN_IF_ERROR(ValidateTPProfilerRegistrationParams(params));
  TF_RETURN_IF_ERROR(ValidateTPProfiler(profiler));
  TF_RETURN_IF_ERROR(ValidateTPProfilerFns(profiler_fns));

  PluggableProfilerFactory factory(std::move(profiler), params.destroy_profiler,
                                   std::move(profiler_fns),
                                   params.destroy_profiler_fns);
  std::function<std::unique_ptr<ProfilerInterface>(const ProfileOptions&)>
      create_func = [factory = std::move(factory)](
                        const ProfileOptions& options) mutable {
        return factory.CreatePluggableProfiler(options);
      };

  tensorflow::profiler::RegisterProfilerFactory(std::move(create_func));

  return Status::OK();
}
}  // namespace profiler
}  // namespace tensorflow
