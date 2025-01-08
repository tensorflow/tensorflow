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

#if GOOGLE_CUDA

#include <stdlib.h>

#include <memory>
#include <utility>

#include "absl/container/fixed_array.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "third_party/gpus/cuda/include/cuda.h"
#include "xla/backends/profiler/gpu/cupti_collector.h"
#include "xla/backends/profiler/gpu/cupti_tracer.h"
#include "xla/backends/profiler/gpu/cupti_wrapper.h"
#include "xla/tsl/profiler/utils/time_utils.h"
#include "xla/tsl/util/env_var.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/macros.h"
#include "tsl/platform/thread_annotations.h"
#include "tsl/profiler/lib/profiler_factory.h"
#include "tsl/profiler/lib/profiler_interface.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace xla {
namespace profiler {

using tensorflow::ProfileOptions;
using tensorflow::profiler::XSpace;
using tsl::ReadBoolFromEnvVar;

// GpuTracer for GPU.
class GpuTracer : public tsl::profiler::ProfilerInterface {
 public:
  explicit GpuTracer(CuptiTracer* cupti_tracer) : cupti_tracer_(cupti_tracer) {
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
};

absl::Status GpuTracer::DoStart() {
  if (!cupti_tracer_->IsAvailable()) {
    return tsl::errors::Unavailable("Another profile session running.");
  }

  options_.cbids_selected = {
    // KERNEL
    CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel,
#if CUDA_VERSION >= 11080  // CUDA 11.8
    CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx,
#endif  // CUDA_VERSION >= 11080
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

// CUDA/CUPTI 10 have issues (leaks and crashes) with CuptiFinalize.
#if CUDA_VERSION >= 11000
  options_.cupti_finalize = true;
#endif

  CuptiTracerCollectorOptions collector_options;
  collector_options.num_gpus = cupti_tracer_->NumGpus();
  uint64_t start_gputime_ns = CuptiTracer::GetTimestamp();
  uint64_t start_walltime_ns = tsl::profiler::GetCurrentTimeNanos();
  cupti_collector_ = CreateCuptiCollector(collector_options, start_walltime_ns,
                                          start_gputime_ns);

  cupti_tracer_->Enable(options_, cupti_collector_.get());
  return absl::OkStatus();
}

absl::Status GpuTracer::Start() {
  absl::Status status = DoStart();
  if (status.ok()) {
    profiling_state_ = State::kStartedOk;
    return absl::OkStatus();
  } else {
    profiling_state_ = State::kStartedError;
    return status;
  }
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
      return tsl::errors::FailedPrecondition(
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
      if (cupti_collector_) {
        uint64_t end_gpu_ns = cupti_collector_->GetTracingEndTimeNs();
        cupti_collector_->Export(space, end_gpu_ns);
      }
      return absl::OkStatus();
    }
  }
  return tsl::errors::Internal("Invalid profiling state: ", profiling_state_);
}

// Not in anonymous namespace for testing purposes.
std::unique_ptr<tsl::profiler::ProfilerInterface> CreateGpuTracer(
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
  return std::make_unique<profiler::GpuTracer>(cupti_tracer);
}

auto register_gpu_tracer_factory = [] {
  RegisterProfilerFactory(&CreateGpuTracer);
  return 0;
}();

}  // namespace profiler
}  // namespace xla

#endif  // GOOGLE_CUDA
