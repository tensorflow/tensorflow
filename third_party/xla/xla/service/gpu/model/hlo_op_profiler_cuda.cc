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

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "xla/backends/profiler/gpu/cupti_buffer_events.h"
#include "xla/backends/profiler/gpu/cupti_collector.h"
#include "xla/backends/profiler/gpu/cupti_tracer.h"
#include "xla/service/gpu/model/hlo_op_profiler.h"

namespace xla {
namespace gpu {

class CuptiKernelTracer : public HloOpProfiler::KernelTracer,
                          public profiler::CuptiTraceCollector {
 public:
  CuptiKernelTracer()
      : profiler::CuptiTraceCollector({}),
        cupti_tracer_(profiler::CuptiTracer::GetCuptiTracerSingleton()) {
    CHECK(cupti_tracer_->IsAvailable());
    profiler::CuptiTracerOptions options;
    options.cbids_selected.push_back(
        // Not interested in API callbacks, but empty list enables them all.
        CUPTI_DRIVER_TRACE_CBID_cu64GLMapBufferObject);
    options.activities_selected.push_back(CUPTI_ACTIVITY_KIND_KERNEL);
    cupti_tracer_->Enable(options, this).IgnoreError();
  }

  uint64_t getMedianKernelTimeNs() && override {
    cupti_tracer_->Disable();  // Also flushes buffer.
    if (kernel_times_ns_.empty()) {
      LOG(ERROR) << "No kernel events";
      return 0;
    }
    absl::c_sort(kernel_times_ns_);
    auto i = kernel_times_ns_.size() / 2;
    // Return median value if number of values is odd.
    if (kernel_times_ns_.size() % 2 != 0) {
      return kernel_times_ns_[i];
    }
    // Return average of the two middle values if the number of values is even.
    return (kernel_times_ns_[i - 1] + kernel_times_ns_[i] + 1) / 2;
  }

 private:
  // CuptiTraceCollector
  void AddEvent(profiler::CuptiTracerEvent&& event) override {
    if (event.type == profiler::CuptiTracerEventType::Kernel) {
      kernel_times_ns_.push_back(event.end_time_ns - event.start_time_ns);
    }
    VLOG(5) << "CuptiTracerEvent: " << event.name << ", "
            << event.end_time_ns - event.start_time_ns << "ns";
  }
  void OnEventsDropped(const std::string& reason,
                       uint32_t num_events) override {
    LOG(WARNING) << "Dropped " << num_events << " events: " << reason;
  }
  void Flush() override {}

  profiler::CuptiTracer* cupti_tracer_;
  std::vector<uint64_t> kernel_times_ns_;
};

/*static*/ std::unique_ptr<HloOpProfiler::KernelTracer>
HloOpProfiler::GetKernelTracer() {
  return std::make_unique<CuptiKernelTracer>();
}

}  // namespace gpu
}  // namespace xla
