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

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/match.h"
#include "xla/backends/profiler/gpu/rocm_collector.h"
#include "xla/backends/profiler/gpu/rocm_tracer.h"
#include "xla/service/gpu/model/hlo_op_profiler.h"

namespace xla {
namespace gpu {

class RocmKernelTracer : public HloOpProfiler::KernelTracer,
                         public profiler::RocmTraceCollector {
 public:
  RocmKernelTracer()
      : profiler::RocmTraceCollector(MakeCollectorOptions(
            profiler::RocmTracer::GetRocmTracerSingleton().NumGpus())),
        rocm_tracer_(&profiler::RocmTracer::GetRocmTracerSingleton()),
        start_timestamp_ns_(profiler::RocmTracer::GetTimestamp()) {
    CHECK(rocm_tracer_->IsAvailable());
    profiler::RocmTracerOptions options;
    options.max_annotation_strings = 1024 * 1024;
    rocm_tracer_->Enable(options, this);
  }

  uint64_t getMedianKernelTimeNs() && override {
    rocm_tracer_->Disable();  // Also flushes buffer.
    if (kernel_times_ns_.empty()) {
      LOG(ERROR) << "No kernel events";
      return 0;
    }
    std::sort(kernel_times_ns_.begin(), kernel_times_ns_.end());
    auto i = kernel_times_ns_.size() / 2;
    // Return median value if number of values is odd.
    if (kernel_times_ns_.size() % 2 != 0) {
      return kernel_times_ns_[i];
    }
    // Return average of the two middle values if the number of values is even.
    return (kernel_times_ns_[i - 1] + kernel_times_ns_[i] + 1) / 2;
  }

 private:
  static profiler::RocmTraceCollectorOptions MakeCollectorOptions(
      uint32_t num_gpus) {
    profiler::RocmTraceCollectorOptions options;
    options.max_callback_api_events = 2 * 1024 * 1024;
    options.max_activity_api_events = 2 * 1024 * 1024;
    options.max_annotation_strings = 1024 * 1024;
    options.num_gpus = num_gpus;
    return options;
  }

  static bool IsComputeKernel(const std::string& name) {
    // Exclude ROCm runtime (memory fill, copy, etc.)
    return !absl::StartsWith(name, "__amd_rocclr_");
  }

  // RocmTraceCollector interface
  void AddEvent(profiler::RocmTracerEvent&& event, bool is_auxiliary) override {
    if (event.type == profiler::RocmTracerEventType::Kernel &&
        event.source == profiler::RocmTracerEventSource::Activity &&
        IsComputeKernel(event.name)) {
      // Filter out events that started before the tracer was created.
      // This discards stray events from warmup runs.
      if (event.start_time_ns < start_timestamp_ns_) {
        return;
      }
      kernel_times_ns_.push_back(event.end_time_ns - event.start_time_ns);
      VLOG(3) << "Kernel dispatch: " << event.name << ", "
              << event.end_time_ns - event.start_time_ns << "ns";
    }
  }

  void OnEventsDropped(const std::string& reason,
                       uint32_t num_events) override {
    LOG(WARNING) << "Dropped " << num_events << " events: " << reason;
  }
  void Flush() override {}
  void Export(tsl::profiler::XSpace* space) override {}

  profiler::RocmTracer* rocm_tracer_;
  std::vector<uint64_t> kernel_times_ns_;
  uint64_t start_timestamp_ns_;
};

/*static*/ std::unique_ptr<HloOpProfiler::KernelTracer>
HloOpProfiler::GetKernelTracer() {
  return std::make_unique<RocmKernelTracer>();
}

}  // namespace gpu
}  // namespace xla
