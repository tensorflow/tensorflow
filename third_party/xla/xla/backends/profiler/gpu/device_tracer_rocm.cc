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

#include <cstdint>
#include <memory>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "xla/backends/profiler/gpu/rocm_collector.h"
#include "xla/backends/profiler/gpu/rocm_tracer.h"
#include "xla/backends/profiler/gpu/rocm_tracer_utils.h"
#include "xla/debug_options_flags.h"
#include "xla/tsl/platform/env_time.h"
#include "xla/tsl/profiler/backends/cpu/annotation_stack.h"
#include "tsl/profiler/lib/profiler_factory.h"
#include "tsl/profiler/lib/profiler_interface.h"
#include "tsl/profiler/protobuf/profiler_options.pb.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace xla {
namespace profiler {

using tensorflow::ProfileOptions;
using tsl::profiler::AnnotationStack;
using tsl::profiler::ProfilerInterface;
using tsl::profiler::XSpace;

// GpuTracer for ROCm GPU.
class GpuTracer : public profiler::ProfilerInterface {
 public:
  explicit GpuTracer(RocmTracer* rocm_tracer) : rocm_tracer_(rocm_tracer) {
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
  RocmTracerOptions options;
  options.max_annotation_strings = 4 * 1024 * 1024;
  return options;
}

RocmTraceCollectorOptions GpuTracer::GetRocmTraceCollectorOptions(
    uint32_t num_gpus) {
  RocmTraceCollectorOptions options;
  options.num_gpus = num_gpus;

  const auto& dbg = xla::GetDebugOptionsFromFlags();
  int64_t max_events = dbg.xla_gpu_rocm_max_trace_events();
  VLOG(2) << "max number of events to be trace from flag = " << max_events;
  if (max_events <= 0) {
    max_events = 4 * 1024 * 1024;
  }
  if (max_events > 1'000'000'000LL) {
    max_events = 1'000'000'000LL;
  }
  VLOG(3) << "maximum number of events to be traced = " << max_events;

  options.max_callback_api_events = max_events;
  options.max_activity_api_events = max_events;
  options.max_annotation_strings = max_events;
  return options;
}

absl::Status GpuTracer::DoStart() {
  AnnotationStack::Enable(true);
  uint64_t start_gputime_ns = RocmTracer::GetTimestamp();
  uint64_t start_walltime_ns = tsl::EnvTime::NowNanos();

  RocmTracerOptions tracer_options = GetRocmTracerOptions();
  RocmTraceCollectorOptions trace_collector_options =
      GetRocmTraceCollectorOptions(rocm_tracer_->NumGpus());
  rocm_trace_collector_ = CreateRocmCollector(
      trace_collector_options, start_walltime_ns, start_gputime_ns);

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
      return absl::FailedPreconditionError(
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
  auto& rocm_tracer = profiler::RocmTracer::GetRocmTracerSingleton();
  if (!rocm_tracer.IsAvailable()) return nullptr;
  return std::make_unique<profiler::GpuTracer>(&rocm_tracer);
}

auto register_rocm_gpu_tracer_factory = [] {
  RegisterProfilerFactory(&CreateGpuTracer);
  return 0;
}();

}  // namespace profiler
}  // namespace xla
