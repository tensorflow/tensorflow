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

#include <cstddef>
#include <memory>

#include "absl/status/statusor.h"
#include "third_party/gpus/cuda/extras/CUPTI/include/cupti_version.h"
#include "xla/backends/profiler/gpu/cupti_pm_sampler.h"

#if CUPTI_API_VERSION >= 24
#include "xla/backends/profiler/gpu/cupti_pm_sampler_impl.h"
#else
#include "xla/backends/profiler/gpu/cupti_pm_sampler_stub.h"
#endif

#include "xla/backends/profiler/gpu/cupti_pm_sampler_factory.h"

namespace xla {
namespace profiler {

absl::StatusOr<std::unique_ptr<CuptiPmSampler>> CreatePmSampler(
    size_t num_gpus, const CuptiPmSamplerOptions& options) {
#if CUPTI_API_VERSION >= 24
  return CuptiPmSamplerImpl::Create(num_gpus, options);
#else
  return std::make_unique<CuptiPmSamplerStub>();
#endif
}

}  // namespace profiler
}  // namespace xla
