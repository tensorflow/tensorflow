/* Copyright 2019 The OpenXLA Authors.

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

#include <stdlib.h>

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "third_party/gpus/cuda/include/cuda.h"
#include "xla/backends/profiler/gpu/cupti_collector.h"
#include "xla/backends/profiler/gpu/cupti_tracer.h"
#include "xla/backends/profiler/gpu/cupti_tracer_options_utils.h"
#include "xla/backends/profiler/gpu/gpu_metadata.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/profiler/utils/time_utils.h"
#include "xla/tsl/util/env_var.h"
#include "tsl/profiler/lib/profiler_factory.h"
#include "tsl/profiler/lib/profiler_interface.h"
#include "tsl/profiler/protobuf/profiler_options.pb.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace xla {
namespace profiler {

using tensorflow::ProfileOptions;
using tensorflow::profiler::XSpace;
using tsl::ReadBoolFromEnvVar;

// GpuTracer for GPU.
class GpuTracer : public tsl::profiler::ProfilerInterface {
 public:
  explicit GpuTracer(CuptiTracer* cupti_tracer, ProfileOptions profile_options)
      : cupti_tracer_(cupti_tracer), profile_options_(profile_options) {
    VLOG(1) << "GpuTracer created.";
  }
  ~GpuTracer() override {}

  // GpuTracer interface:
  absl::Status Start() override;
  absl::Status Stop() override;
  absl::Status CollectData(XSpace* space) override;

 private:
  absl::Status DoStart();
  absl::Status DoStop();

  enum State {
    kNotStarted,
    kStartedOk,
    kStartedError,
    kStoppedOk,
    kStoppedError
  };
  State profiling_state_ = State::kNotStarted;

  CuptiTracer* cupti_tracer_;
  CuptiTracerOptions options_;
  std::unique_ptr<CuptiTraceCollector> cupti_collector_;
  ProfileOptions profile_options_;
  std::vector<std::unique_ptr<tensorflow::profiler::XPlane>> xplanes_;
};

absl::Status GpuTracer::DoStart() {
  if (!cupti_tracer_->IsAvailable()) {
    return absl::UnavailableError("Another profile session running.");
  }

  options_.cbids_selected = CuptiTracer::CreateDefaultCallbackIds();

  bool trace_concurrent_kernels = false;
  ReadBoolFromEnvVar("TF_GPU_CUPTI_FORCE_CONCURRENT_KERNEL", true,
                     &trace_concurrent_kernels)
      .IgnoreError();
  options_.activities_selected.push_back(
      trace_concurrent_kernels ? CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL
                               : CUPTI_ACTIVITY_KIND_KERNEL);
  options_.activities_selected.push_back(CUPTI_ACTIVITY_KIND_MEMCPY);
  options_.activities_selected.push_back(CUPTI_ACTIVITY_KIND_MEMCPY2);
  options_.activities_selected.push_back(CUPTI_ACTIVITY_KIND_OVERHEAD);
  options_.activities_selected.push_back(CUPTI_ACTIVITY_KIND_MEMSET);

  // TODO: Change default to true once we have more confidence in HES.
  ReadBoolFromEnvVar("TF_GPU_CUPTI_ENABLE_ACTIVITY_HW_TRACING", false,
                     &options_.enable_activity_hardware_tracing)
      .IgnoreError();

// CUDA/CUPTI 10 have issues (leaks and crashes) with CuptiFinalize.
#if CUDA_VERSION >= 11000
  options_.cupti_finalize = true;
#endif

  CuptiTracerCollectorOptions collector_options;
  int num_gpus = cupti_tracer_->NumGpus();
  collector_options.num_gpus = num_gpus;

  // TODO: Add a test to verify that the options are set correctly and
  // collectors are generating correct data once ProfileData is
  // available(b/399675726).
  TF_RETURN_IF_ERROR(UpdateCuptiTracerOptionsFromProfilerOptions(
      profile_options_, options_, collector_options));

  if (collector_options.num_gpus <= 0 ||
      collector_options.num_gpus > num_gpus) {
    if (collector_options.num_gpus != 0) {
      LOG(WARNING)
          << "The provided number of GPUs (" << collector_options.num_gpus
          << ") is invalid. Profiling will be done on all available GPUs ("
          << num_gpus << ").";
    }
    collector_options.num_gpus = num_gpus;
  }

  uint64_t start_gputime_ns = CuptiTracer::GetTimestamp();
  uint64_t start_walltime_ns = tsl::profiler::GetCurrentTimeNanos();
  cupti_collector_ = CreateCuptiCollector(collector_options, start_walltime_ns,
                                          start_gputime_ns);

  xplanes_.reserve(collector_options.num_gpus);
  for (int i = 0; i < collector_options.num_gpus; ++i) {
    xplanes_.push_back(std::make_unique<tensorflow::profiler::XPlane>());
  }
  TF_RETURN_IF_ERROR(
      cupti_tracer_->Enable(options_, cupti_collector_.get(), xplanes_));
  AddGpuMetadata();
  return absl::OkStatus();
}

absl::Status GpuTracer::Start() {
  absl::Status status = DoStart();
  if (status.ok()) {
    profiling_state_ = State::kStartedOk;
    return absl::OkStatus();
  }
  profiling_state_ = State::kStartedError;
  return status;
}

absl::Status GpuTracer::DoStop() {
  cupti_tracer_->Disable();
  return absl::OkStatus();
}

absl::Status GpuTracer::Stop() {
  if (profiling_state_ == State::kStartedOk) {
    absl::Status status = DoStop();
    profiling_state_ = status.ok() ? State::kStoppedOk : State::kStoppedError;
  }
  return absl::OkStatus();
}

absl::Status GpuTracer::CollectData(XSpace* space) {
  VLOG(2) << "Collecting data to XSpace from GpuTracer.";
  switch (profiling_state_) {
    case State::kNotStarted:
      VLOG(1) << "No trace data collected, session wasn't started";
      return absl::OkStatus();
    case State::kStartedOk:
      return absl::FailedPreconditionError(
          "Cannot collect trace before stopping");
    case State::kStartedError:
      LOG(ERROR) << "Cannot collect, profiler failed to start";
      return absl::OkStatus();
    case State::kStoppedError:
      VLOG(1) << "No trace data collected";
      return absl::OkStatus();
    case State::kStoppedOk: {
      std::string cupti_error = CuptiTracer::ErrorIfAny();
      if (!cupti_error.empty()) {
        space->add_errors(std::move(cupti_error));
      }
      std::string events_dropped = cupti_collector_->ReportNumEventsIfDropped();
      if (!events_dropped.empty()) {
        space->add_warnings(std::move(events_dropped));
      }
      if (options_.pm_sampler_options.enable) {
        // Adds PM sampling xplanes to the response before CuptiCollector to
        // merge and export all the events.
        for (auto& xplane : xplanes_) {
          if (xplane) {
            *space->add_planes() = *xplane;
          }
        }
      }
      if (cupti_collector_) {
        uint64_t end_gpu_ns = cupti_collector_->GetTracingEndTimeNs();
        cupti_collector_->Export(space, end_gpu_ns);
      }
      return absl::OkStatus();
    }
  }
  return absl::InternalError(
      absl::StrCat("Invalid profiling state: ", profiling_state_));
}

// Not in anonymous namespace for testing purposes.
std::unique_ptr<tsl::profiler::ProfilerInterface> CreateGpuTracer(
    const ProfileOptions& options) {
  if (options.device_tracer_level() == 0) {
    return nullptr;
  }
  if (options.device_type() != ProfileOptions::GPU &&
      options.device_type() != ProfileOptions::UNSPECIFIED) {
    return nullptr;
  }
  profiler::CuptiTracer* cupti_tracer =
      profiler::CuptiTracer::GetCuptiTracerSingleton();
  if (!cupti_tracer->IsAvailable()) {
    return nullptr;
  }
  return std::make_unique<profiler::GpuTracer>(cupti_tracer, options);
}

auto register_gpu_tracer_factory = [] {
  RegisterProfilerFactory(&CreateGpuTracer);
  return 0;
}();

}  // namespace profiler
}  // namespace xla
