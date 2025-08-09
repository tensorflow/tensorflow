/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/backends/profiler/gpu/cupti_pm_sampler.h"

#include "absl/status/status.h"
#include "xla/backends/profiler/gpu/cupti_interface.h"

namespace xla {
namespace profiler {

// Stub implementation of CuptiPmSampler
// Full implementation is in cupti_pm_sampler_impl.h/.cc

absl::Status CuptiPmSampler::Initialize(size_t num_gpus,
                                        CuptiPmSamplerOptions* options) {
  return absl::OkStatus();
}

absl::Status CuptiPmSampler::StartSampler() { return absl::OkStatus(); }

absl::Status CuptiPmSampler::StopSampler() { return absl::OkStatus(); }

absl::Status CuptiPmSampler::Deinitialize() { return absl::OkStatus(); }

}  // namespace profiler
}  // namespace xla
