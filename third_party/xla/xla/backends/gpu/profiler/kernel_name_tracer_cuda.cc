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

#include "xla/backends/gpu/profiler/kernel_name_tracer.h"
#include "xla/backends/gpu/profiler/kernel_name_tracer_factory.h"
#include "xla/backends/profiler/gpu/cupti_collector.h"
#include "xla/backends/profiler/gpu/cupti_tracer.h"
#include "xla/stream_executor/cuda/cuda_platform_id.h"
#include "xla/stream_executor/platform/platform_object_registry.h"
#include "xla/tsl/profiler/utils/time_utils.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

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

void KernelNameTracerCuda::start() {
  profiler::CuptiTracerCollectorOptions collector_options{};
  collector_options.num_gpus = profiler::CuptiTracer::NumGpus();
  auto start_gputime_ns = profiler::CuptiTracer::GetTimestamp();
  auto start_walltime_ns = tsl::profiler::GetCurrentTimeNanos();
  cupti_collector_ = profiler::CreateCuptiCollector(
      collector_options, start_walltime_ns, start_gputime_ns);
  profiler::CuptiTracerOptions options{};
  options.activities_selected = {CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL};
  cupti_tracer_->Enable(options, cupti_collector_.get()).IgnoreError();
}

std::vector<std::string> KernelNameTracerCuda::stop() {
  cupti_tracer_->Disable();
  uint64_t end_gpu_ns = cupti_collector_->GetTracingEndTimeNs();
  auto space = std::make_unique<tensorflow::profiler::XSpace>();
  cupti_collector_->Export(space.get(), end_gpu_ns);
  for (const auto& plane : space->planes()) {
    if (plane.name() == "/device:GPU:0") {
      std::vector<std::string> names;
      for (const auto& line : plane.lines()) {
        for (const auto& event : line.events()) {
          if (auto it = plane.event_metadata().find(event.metadata_id());
              it != plane.event_metadata().end()) {
            names.push_back(it->second.name());
          }
        }
      }
      return names;
    }
  }

  return {};
}

STREAM_EXECUTOR_REGISTER_OBJECT_STATICALLY(
    CudaKernelNameTracerFactory, KernelNameTracerFactory,
    stream_executor::cuda::kCudaPlatformId,
    []() -> std::unique_ptr<KernelNameTracer> {
      return std::make_unique<KernelNameTracerCuda>();
    });

}  // namespace xla::gpu
