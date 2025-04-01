/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/backends/gpu/codegen/triton/kernel_name_tracer.h"
#include "xla/backends/profiler/gpu/cupti_collector.h"
#include "xla/backends/profiler/gpu/cupti_tracer.h"
#include "xla/tsl/profiler/utils/time_utils.h"

namespace xla::gpu {

// This class allows to get the name of the kernel that was used.
// It works only on CUDA. It uses CuptiTracer to get the kernel name.
class KernelNameTracerCuda : public KernelNameTracer {
 public:
  KernelNameTracerCuda()
      : cupti_tracer_(profiler::CuptiTracer::GetCuptiTracerSingleton()) {}

  void start() override;

  std::vector<std::string> stop() override;

 private:
  profiler::CuptiTracer* cupti_tracer_;  // Not owned.
  std::unique_ptr<profiler::CuptiTraceCollector> cupti_collector_;
};

std::unique_ptr<KernelNameTracer> KernelNameTracer::Create() {
  return std::make_unique<KernelNameTracerCuda>();
}

void KernelNameTracerCuda::start() {
  profiler::CuptiTracerCollectorOptions collector_options;
  collector_options.num_gpus = profiler::CuptiTracer::NumGpus();
  auto start_gputime_ns = profiler::CuptiTracer::GetTimestamp();
  auto start_walltime_ns = tsl::profiler::GetCurrentTimeNanos();
  cupti_collector_ = profiler::CreateCuptiCollector(
      collector_options, start_walltime_ns, start_gputime_ns);
  profiler::CuptiTracerOptions options;
  options.activities_selected = {CUPTI_ACTIVITY_KIND_KERNEL};
  cupti_tracer_->Enable(options, cupti_collector_.get()).IgnoreError();
}

std::vector<std::string> KernelNameTracerCuda::stop() {
  cupti_tracer_->Disable();
  uint64_t end_gpu_ns = cupti_collector_->GetTracingEndTimeNs();
  auto space = std::make_unique<tensorflow::profiler::XSpace>();
  cupti_collector_->Export(space.get(), end_gpu_ns);
  for (const auto& plane : space->planes()) {
    std::vector<std::string> names;
    if (plane.name() == "/device:GPU:0") {
      for (const auto& metadata : plane.event_metadata()) {
        names.push_back(metadata.second.name());
      }
      return names;
    }
  }
  return {};
}

}  // namespace xla::gpu
