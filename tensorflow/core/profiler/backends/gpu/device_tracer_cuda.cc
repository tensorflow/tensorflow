/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#if GOOGLE_CUDA

#include <stdlib.h>

#include <memory>
#include <utility>

#include "absl/container/fixed_array.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "tensorflow/core/framework/step_stats.pb.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/profiler/backends/gpu/cupti_collector.h"
#include "tensorflow/core/profiler/backends/gpu/cupti_tracer.h"
#include "tensorflow/core/profiler/backends/gpu/cupti_wrapper.h"
#include "tensorflow/core/profiler/lib/profiler_factory.h"
#include "tensorflow/core/profiler/lib/profiler_interface.h"
#include "tensorflow/core/profiler/protobuf/xplane.pb.h"
#include "tensorflow/core/profiler/utils/time_utils.h"
#include "tensorflow/core/util/env_var.h"

namespace tensorflow {
namespace profiler {

// GpuTracer for GPU.
class GpuTracer : public profiler::ProfilerInterface {
 public:
  GpuTracer(CuptiTracer* cupti_tracer, CuptiInterface* cupti_interface)
      : cupti_tracer_(cupti_tracer) {
    VLOG(1) << "GpuTracer created.";
  }
  ~GpuTracer() override {}

  // GpuTracer interface:
  Status Start() override;
  Status Stop() override;
  Status CollectData(XSpace* space) override;

 private:
  Status DoStart();
  Status DoStop();

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
};

Status GpuTracer::DoStart() {
  if (!cupti_tracer_->IsAvailable()) {
    return errors::Unavailable("Another profile session running.");
  }

  options_.cbids_selected = {
      // KERNEL
      CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel,
      // MEMCPY
      CUPTI_DRIVER_TRACE_CBID_cuMemcpy,
      CUPTI_DRIVER_TRACE_CBID_cuMemcpyAsync,
      CUPTI_DRIVER_TRACE_CBID_cuMemcpyHtoD_v2,
      CUPTI_DRIVER_TRACE_CBID_cuMemcpyHtoDAsync_v2,
      CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoH_v2,
      CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoHAsync_v2,
      CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoD_v2,
      CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoDAsync_v2,
      CUPTI_DRIVER_TRACE_CBID_cuMemcpyAtoH_v2,
      CUPTI_DRIVER_TRACE_CBID_cuMemcpyAtoHAsync_v2,
      CUPTI_DRIVER_TRACE_CBID_cuMemcpyAtoD_v2,
      CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoA_v2,
      CUPTI_DRIVER_TRACE_CBID_cuMemcpyAtoA_v2,
      CUPTI_DRIVER_TRACE_CBID_cuMemcpy2D_v2,
      CUPTI_DRIVER_TRACE_CBID_cuMemcpy2DUnaligned_v2,
      CUPTI_DRIVER_TRACE_CBID_cuMemcpy2DAsync_v2,
      CUPTI_DRIVER_TRACE_CBID_cuMemcpy3D_v2,
      CUPTI_DRIVER_TRACE_CBID_cuMemcpy3DAsync_v2,
      CUPTI_DRIVER_TRACE_CBID_cuMemcpyHtoA_v2,
      CUPTI_DRIVER_TRACE_CBID_cuMemcpyHtoAAsync_v2,
      // MemAlloc
      CUPTI_DRIVER_TRACE_CBID_cuMemAlloc_v2,
      CUPTI_DRIVER_TRACE_CBID_cuMemAllocPitch_v2,
      // MemFree
      CUPTI_DRIVER_TRACE_CBID_cuMemFree_v2,
      // Memset
      CUPTI_DRIVER_TRACE_CBID_cuMemsetD8_v2,
      CUPTI_DRIVER_TRACE_CBID_cuMemsetD16_v2,
      CUPTI_DRIVER_TRACE_CBID_cuMemsetD32_v2,
      CUPTI_DRIVER_TRACE_CBID_cuMemsetD2D8_v2,
      CUPTI_DRIVER_TRACE_CBID_cuMemsetD2D16_v2,
      CUPTI_DRIVER_TRACE_CBID_cuMemsetD2D32_v2,
      CUPTI_DRIVER_TRACE_CBID_cuMemsetD8Async,
      CUPTI_DRIVER_TRACE_CBID_cuMemsetD16Async,
      CUPTI_DRIVER_TRACE_CBID_cuMemsetD32Async,
      CUPTI_DRIVER_TRACE_CBID_cuMemsetD2D8Async,
      CUPTI_DRIVER_TRACE_CBID_cuMemsetD2D16Async,
      CUPTI_DRIVER_TRACE_CBID_cuMemsetD2D32Async,
      // GENERIC
      CUPTI_DRIVER_TRACE_CBID_cuStreamSynchronize,
  };

  bool use_cupti_activity_api = true;
  ReadBoolFromEnvVar("TF_GPU_CUPTI_USE_ACTIVITY_API", true,
                     &use_cupti_activity_api)
      .IgnoreError();
  options_.enable_event_based_activity = !use_cupti_activity_api;

  bool trace_concurrent_kernels = false;
  ReadBoolFromEnvVar("TF_GPU_CUPTI_FORCE_CONCURRENT_KERNEL", false,
                     &trace_concurrent_kernels)
      .IgnoreError();
  options_.activities_selected.push_back(
      trace_concurrent_kernels ? CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL
                               : CUPTI_ACTIVITY_KIND_KERNEL);
  options_.activities_selected.push_back(CUPTI_ACTIVITY_KIND_MEMCPY);
  options_.activities_selected.push_back(CUPTI_ACTIVITY_KIND_MEMCPY2);
  options_.activities_selected.push_back(CUPTI_ACTIVITY_KIND_OVERHEAD);
  options_.activities_selected.push_back(CUPTI_ACTIVITY_KIND_MEMSET);

// CUDA/CUPTI 10 have issues (leaks and crashes) with CuptiFinalize.
#if CUDA_VERSION < 10000
  if (!trace_concurrent_kernels) options_.cupti_finalize = true;
#elif CUDA_VERSION >= 11000
  options_.cupti_finalize = true;
#endif

  CuptiTracerCollectorOptions collector_options;
  collector_options.num_gpus = cupti_tracer_->NumGpus();
  uint64 start_gputime_ns = CuptiTracer::GetTimestamp();
  uint64 start_walltime_ns = GetCurrentTimeNanos();
  cupti_collector_ = CreateCuptiCollector(collector_options, start_walltime_ns,
                                          start_gputime_ns);

  cupti_tracer_->Enable(options_, cupti_collector_.get());
  return OkStatus();
}

Status GpuTracer::Start() {
  Status status = DoStart();
  if (status.ok()) {
    profiling_state_ = State::kStartedOk;
    return OkStatus();
  } else {
    profiling_state_ = State::kStartedError;
    return status;
  }
}

Status GpuTracer::DoStop() {
  cupti_tracer_->Disable();
  return OkStatus();
}

Status GpuTracer::Stop() {
  if (profiling_state_ == State::kStartedOk) {
    Status status = DoStop();
    profiling_state_ = status.ok() ? State::kStoppedOk : State::kStoppedError;
  }
  return OkStatus();
}

Status GpuTracer::CollectData(XSpace* space) {
  VLOG(2) << "Collecting data to XSpace from GpuTracer.";
  switch (profiling_state_) {
    case State::kNotStarted:
      VLOG(1) << "No trace data collected, session wasn't started";
      return OkStatus();
    case State::kStartedOk:
      return errors::FailedPrecondition("Cannot collect trace before stopping");
    case State::kStartedError:
      LOG(ERROR) << "Cannot collect, profiler failed to start";
      return OkStatus();
    case State::kStoppedError:
      VLOG(1) << "No trace data collected";
      return OkStatus();
    case State::kStoppedOk: {
      std::string cupti_error = CuptiTracer::ErrorIfAny();
      if (!cupti_error.empty()) {
        space->add_errors(std::move(cupti_error));
      }
      std::string events_dropped = cupti_collector_->ReportNumEventsIfDropped();
      if (!events_dropped.empty()) {
        space->add_warnings(std::move(events_dropped));
      }
      if (cupti_collector_) {
        uint64 end_gpu_ns = CuptiTracer::GetTimestamp();
        cupti_collector_->Export(space, end_gpu_ns);
      }
      return OkStatus();
    }
  }
  return errors::Internal("Invalid profiling state: ", profiling_state_);
}

// Not in anonymous namespace for testing purposes.
std::unique_ptr<profiler::ProfilerInterface> CreateGpuTracer(
    const ProfileOptions& options) {
  if (options.device_tracer_level() == 0) return nullptr;
  if (options.device_type() != ProfileOptions::GPU &&
      options.device_type() != ProfileOptions::UNSPECIFIED)
    return nullptr;
  profiler::CuptiTracer* cupti_tracer =
      profiler::CuptiTracer::GetCuptiTracerSingleton();
  if (!cupti_tracer->IsAvailable()) {
    return nullptr;
  }
  profiler::CuptiInterface* cupti_interface = profiler::GetCuptiInterface();
  return absl::make_unique<profiler::GpuTracer>(cupti_tracer, cupti_interface);
}

auto register_gpu_tracer_factory = [] {
  RegisterProfilerFactory(&CreateGpuTracer);
  return 0;
}();

}  // namespace profiler
}  // namespace tensorflow

#endif  // GOOGLE_CUDA
