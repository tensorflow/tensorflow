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

#ifndef XLA_BACKENDS_PROFILER_GPU_CUPTI_PM_SAMPLER_H_
#define XLA_BACKENDS_PROFILER_GPU_CUPTI_PM_SAMPLER_H_

#include <cstddef>
#include <functional>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/time/time.h"
#include "xla/backends/profiler/gpu/cupti_collector.h"

namespace xla {
namespace profiler {

// By default provide just a minimal interface which can be stubbed out
//
// Specialize a full implementation for the actual PM sampler in
// cupti_pm_sampler_impl.h

// Configuration for PM sampler
// Do not use types that are defined in new CUPTI PM sampler header
struct CuptiPmSamplerOptions {
  // Whether to enable PM sampler
  bool enable = false;
  // List of metrics to enable
  std::vector<std::string> metrics{};
  // Pinned system memory buffer size in bytes to process PM counters
  // Default: 64MB, Max: 4GB
  size_t hw_buf_size = 64 * 1024 * 1024;
  // Sample interval of 500,000ns = 2khz
  size_t sample_interval_ns = 500'000;
  // Decode thread triggers every 100ms (should have 200 samples @ 2khz)
  absl::Duration decode_period = absl::Milliseconds(100);
  // Maximum samples to allocate host space for, 2.5x expected
  size_t max_samples = 500;
  // Devices per decode thread
  size_t devs_per_decode_thd = 8;
  // What to do with samples once gathered
  // Note, must be thread-safe - may be called by multiple decode threads
  // simultaneously (with different samples data per thread)
  std::function<void(PmSamples* samples)> process_samples;
};

class CuptiPmSampler {
 public:
  CuptiPmSampler() = default;

  // Not copyable or movable
  CuptiPmSampler(const CuptiPmSampler&) = delete;
  CuptiPmSampler(CuptiPmSampler&&) = delete;
  CuptiPmSampler& operator=(const CuptiPmSampler&) = delete;
  CuptiPmSampler& operator=(CuptiPmSampler&&) = delete;

  // Destructor
  virtual ~CuptiPmSampler() = default;

  // Start sampler
  virtual absl::Status StartSampler() = 0;

  // Stop sampler
  virtual absl::Status StopSampler() = 0;

  // Deinitialize the PM sampler
  virtual absl::Status Deinitialize() = 0;
};

}  // namespace profiler
}  // namespace xla

#endif  // XLA_BACKENDS_PROFILER_GPU_CUPTI_PM_SAMPLER_H_
