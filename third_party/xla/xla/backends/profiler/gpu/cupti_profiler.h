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
#ifndef XLA_BACKENDS_PROFILER_GPU_CUPTI_PROFILER_H_
#define XLA_BACKENDS_PROFILER_GPU_CUPTI_PROFILER_H_

#include <optional>
#include <string>

#include "third_party/gpus/cuda/extras/CUPTI/include/cupti.h"
#include "xla/backends/profiler/gpu/cupti_interface.h"
#include "tsl/platform/types.h"

namespace xla {
namespace profiler {

struct CuptiProfilerOptions {};

// The class enables CUPTI Profiling/Perfworks API.
class CuptiProfiler {
 public:
  // Not copyable or movable
  CuptiProfiler(const CuptiProfiler&) = delete;
  CuptiProfiler& operator=(const CuptiProfiler&) = delete;

  // Returns a pointer to singleton CuptiProfiler.
  static CuptiProfiler* GetCuptiProfilerSingleton();

  // Only one profile session can be live in the same time.
  bool IsAvailable() const;
  bool NeedRootAccess() const { return need_root_access_; }

  void Enable(const CuptiProfilerOptions& option);
  void Disable();

  static uint64_t GetTimestamp();
  static int NumGpus();
  // Returns the error (if any) when using libcupti.
  static std::string ErrorIfAny();

 protected:
  // protected constructor for injecting mock cupti interface for testing.
  explicit CuptiProfiler(CuptiInterface* cupti_interface);

 private:
  int num_gpus_;
  std::optional<CuptiProfilerOptions> option_;
  CuptiInterface* cupti_interface_ = nullptr;

  // CUPTI 10.1 and higher need root access to profile.
  bool need_root_access_ = false;

  // Cupti handle for driver or runtime API callbacks. Cupti permits a single
  // subscriber to be active at any time and can be used to trace Cuda runtime
  // as and driver calls for all contexts and devices.
  CUpti_SubscriberHandle subscriber_;  // valid when api_tracing_enabled_.
};

}  // namespace profiler
}  // namespace xla

#endif  // XLA_BACKENDS_PROFILER_GPU_CUPTI_PROFILER_H_
