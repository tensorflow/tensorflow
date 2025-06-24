/* Copyright 2024 The OpenXLA Authors. All Rights Reserved.

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

#if TENSORFLOW_USE_ROCM

#include <memory>
#include <utility>

#include "absl/container/fixed_array.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "xla/backends/profiler/gpu/rocm_collector.h"
#include "xla/backends/profiler/gpu/rocm_tracer.h"
#include "xla/tsl/profiler/backends/cpu/annotation_stack.h"
#include "xla/tsl/profiler/utils/parse_annotation.h"
#include "xla/tsl/profiler/utils/xplane_builder.h"
#include "xla/tsl/profiler/utils/xplane_schema.h"
#include "xla/tsl/profiler/utils/xplane_utils.h"
#include "xla/tsl/util/env_var.h"
#include "tsl/platform/abi.h"
#include "tsl/platform/env_time.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/macros.h"
#include "tsl/platform/thread_annotations.h"
#include "tsl/profiler/lib/profiler_factory.h"
#include "tsl/profiler/lib/profiler_interface.h"

namespace xla {
namespace profiler {

using tensorflow::ProfileOptions;
using tsl::profiler::Annotation;
using tsl::profiler::AnnotationStack;
using tsl::profiler::FindOrAddMutablePlaneWithName;
using tsl::profiler::GetStatTypeStr;
using tsl::profiler::GpuPlaneName;
using tsl::profiler::kDeviceVendorAMD;
using tsl::profiler::kThreadIdOverhead;
using tsl::profiler::ParseAnnotationStack;
using tsl::profiler::ProfilerInterface;
using tsl::profiler::RegisterProfilerFactory;
using tsl::profiler::StatType;
using tsl::profiler::XEventBuilder;
using tsl::profiler::XEventMetadata;
using tsl::profiler::XLineBuilder;
using tsl::profiler::XPlaneBuilder;
using tsl::profiler::XSpace;

// GpuTracer for ROCm GPU.
class GpuTracer : public profiler::ProfilerInterface {
 public:
  GpuTracer(RocmTracer* rocm_tracer) : rocm_tracer_(rocm_tracer) {
    LOG(INFO) << "GpuTracer created.";
  }
  ~GpuTracer() override {}

  // GpuTracer interface:
  absl::Status Start() override;
  absl::Status Stop() override;
  absl::Status CollectData(XSpace* space) override;

 private:
  absl::Status DoStart();
  absl::Status DoStop();

  RocmTracerOptions GetRocmTracerOptions();

  RocmTraceCollectorOptions GetRocmTraceCollectorOptions(uint32_t num_gpus);

  enum State {
    kNotStarted,
    kStartedOk,
    kStartedError,
    kStoppedOk,
    kStoppedError
  };
  State profiling_state_ = State::kNotStarted;

  RocmTracer* rocm_tracer_;
  std::unique_ptr<RocmTraceCollector> rocm_trace_collector_;
};

RocmTracerOptions GpuTracer::GetRocmTracerOptions() {
  // TODO(rocm-profiler): We need support for context similar to CUDA
  RocmTracerOptions options;
  std::vector<uint32_t> empty_vec;

  // clang formatting does not preserve one entry per line
  // clang-format off
  std::vector<uint32_t> hip_api_domain_ops{
      // KERNEL
      HIP_API_ID_hipExtModuleLaunchKernel,
      HIP_API_ID_hipModuleLaunchKernel,
      HIP_API_ID_hipHccModuleLaunchKernel,
      HIP_API_ID_hipLaunchKernel,
      HIP_API_ID_hipExtLaunchKernel,
      // MEMCPY
      HIP_API_ID_hipMemcpy,
      HIP_API_ID_hipMemcpyAsync,
      HIP_API_ID_hipMemcpyDtoD,
      HIP_API_ID_hipMemcpyDtoDAsync,
      HIP_API_ID_hipMemcpyDtoH,
      HIP_API_ID_hipMemcpyDtoHAsync,
      HIP_API_ID_hipMemcpyHtoD,
      HIP_API_ID_hipMemcpyHtoDAsync,
      HIP_API_ID_hipMemcpyPeer,
      HIP_API_ID_hipMemcpyPeerAsync,

      // MEMSet
      HIP_API_ID_hipMemsetD32,
      HIP_API_ID_hipMemsetD32Async,
      HIP_API_ID_hipMemsetD16,
      HIP_API_ID_hipMemsetD16Async,
      HIP_API_ID_hipMemsetD8,
      HIP_API_ID_hipMemsetD8Async,
      HIP_API_ID_hipMemset,
      HIP_API_ID_hipMemsetAsync,

      // MEMAlloc
      HIP_API_ID_hipMalloc,
      HIP_API_ID_hipMallocPitch,
      // MEMFree
      HIP_API_ID_hipFree,
      // GENERIC
      HIP_API_ID_hipStreamSynchronize,
  };
  // clang-format on

  options.api_tracking_set =
      std::set<uint32_t>(hip_api_domain_ops.begin(), hip_api_domain_ops.end());

  // These are the list of APIs we track since roctracer activity
  // does not provide all the information necessary to fully populate the
  // TF events. We need to track the APIs for those activities in API domain but
  // we only use them for filling the missing items in their corresponding
  // activity (using correlation id).
  // clang-format off
  std::vector<uint32_t> hip_api_aux_ops{
    HIP_API_ID_hipStreamWaitEvent,
    // TODO(rocm-profiler): finding device ID from hipEventSynchronize need some
    // extra work, we ignore it for now.
    // HIP_API_ID_hipEventSynchronize,
    HIP_API_ID_hipHostFree,
    HIP_API_ID_hipHostMalloc,
    HIP_API_ID_hipSetDevice  //  added to track default device
  };

  // clang-format on

  hip_api_domain_ops.insert(hip_api_domain_ops.end(), hip_api_aux_ops.begin(),
                            hip_api_aux_ops.end());

  // options.api_callbacks.emplace(ACTIVITY_DOMAIN_HIP_API, hip_api_domain_ops);
  options.api_callbacks.emplace(ACTIVITY_DOMAIN_HIP_API, empty_vec);

  options.activity_tracing.emplace(ACTIVITY_DOMAIN_HIP_OPS, empty_vec);

  return options;
}

RocmTraceCollectorOptions GpuTracer::GetRocmTraceCollectorOptions(
    uint32_t num_gpus) {
  RocmTraceCollectorOptions options;
  options.max_callback_api_events = 2 * 1024 * 1024;
  options.max_activity_api_events = 2 * 1024 * 1024;
  options.max_annotation_strings = 1024 * 1024;
  options.num_gpus = num_gpus;
  return options;
}

absl::Status GpuTracer::DoStart() {
  if (!rocm_tracer_->IsAvailable()) {
    return tsl::errors::Unavailable("Another profile session running.");
  }

  AnnotationStack::Enable(true);

  RocmTraceCollectorOptions trace_collector_options =
      GetRocmTraceCollectorOptions(rocm_tracer_->NumGpus());
  uint64_t start_gputime_ns = RocmTracer::GetTimestamp();
  uint64_t start_walltime_ns = tsl::EnvTime::NowNanos();
  rocm_trace_collector_ = CreateRocmCollector(
      trace_collector_options, start_walltime_ns, start_gputime_ns);

  RocmTracerOptions tracer_options = GetRocmTracerOptions();
  rocm_tracer_->Enable(tracer_options, rocm_trace_collector_.get());

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
  rocm_tracer_->Disable();
  AnnotationStack::Enable(false);
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
  switch (profiling_state_) {
    case State::kNotStarted:
      VLOG(3) << "No trace data collected, session wasn't started";
      return absl::OkStatus();
    case State::kStartedOk:
      return tsl::errors::FailedPrecondition(
          "Cannot collect trace before stopping");
    case State::kStartedError:
      LOG(ERROR) << "Cannot collect, roctracer failed to start";
      return absl::OkStatus();
    case State::kStoppedError:
      VLOG(3) << "No trace data collected";
      return absl::OkStatus();
    case State::kStoppedOk: {
      if (rocm_trace_collector_) rocm_trace_collector_->Export(space);
      return absl::OkStatus();
    }
  }
  return absl::InternalError(
      absl::StrCat("Invalid profiling state: ", profiling_state_));
}

// Not in anonymous namespace for testing purposes.
std::unique_ptr<profiler::ProfilerInterface> CreateGpuTracer(
    const ProfileOptions& options) {
  if (options.device_type() != ProfileOptions::GPU &&
      options.device_type() != ProfileOptions::UNSPECIFIED)
    return nullptr;

  profiler::RocmTracer* rocm_tracer =
      profiler::RocmTracer::GetRocmTracerSingleton();
  if (!rocm_tracer->IsAvailable()) return nullptr;

  return std::make_unique<profiler::GpuTracer>(rocm_tracer);
}

auto register_rocm_gpu_tracer_factory = [] {
  RegisterProfilerFactory(&CreateGpuTracer);
  return 0;
}();

}  // namespace profiler
}  // namespace xla

#endif  // TENSORFLOW_USE_ROCM
