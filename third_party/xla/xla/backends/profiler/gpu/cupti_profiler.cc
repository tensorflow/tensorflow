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

#include "xla/backends/profiler/gpu/cupti_profiler.h"

#include "absl/cleanup/cleanup.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "tsl/platform/env.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/host_info.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/macros.h"

namespace xla {
namespace profiler {

namespace {

/*static*/ std::string ErrorWithHostname(absl::string_view error_message) {
  return absl::StrCat(tsl::port::Hostname(), ": ", error_message);
}

}  // namespace

CuptiProfiler::CuptiProfiler(CuptiInterface *cupti_interface)
    : num_gpus_(NumGpus()) {}

/* static */ CuptiProfiler *CuptiProfiler::GetCuptiProfilerSingleton() {
  static auto *singleton = new CuptiProfiler(GetCuptiInterface());
  return singleton;
}

bool CuptiProfiler::IsAvailable() const { return NumGpus(); }

int CuptiProfiler::NumGpus() {
  static int num_gpus = []() -> int {
    if (cuInit(0) != CUDA_SUCCESS) {
      return 0;
    }
    int gpu_count;
    if (cuDeviceGetCount(&gpu_count) != CUDA_SUCCESS) {
      return 0;
    }
    LOG(INFO) << "Profiler found " << gpu_count << " GPUs";
    return gpu_count;
  }();
  return num_gpus;
}

void CuptiProfiler::Enable(const CuptiProfilerOptions &option) {}

void CuptiProfiler::Disable() {}

/*static*/ uint64_t CuptiProfiler::GetTimestamp() {
  uint64_t tsc;
  CuptiInterface *cupti_interface = GetCuptiInterface();
  if (cupti_interface && cupti_interface->GetTimestamp(&tsc) == CUPTI_SUCCESS) {
    return tsc;
  }
  // Return 0 on error. If an activity timestamp is 0, the activity will be
  // dropped during time normalization.
  return 0;
}

/*static*/ std::string CuptiProfiler::ErrorIfAny() {
  if (CuptiProfiler::NumGpus() == 0) {
    return ErrorWithHostname("No GPU detected.");
  } else if (CuptiProfiler::GetCuptiProfilerSingleton()->NeedRootAccess()) {
    return ErrorWithHostname(
        "Insufficient privilege to run libcupti (you need root permission).");
  } else if (CuptiProfiler::GetTimestamp() == 0) {
    return ErrorWithHostname(
        "Failed to load libcupti (is it installed and accessible?)");
  }
  return "";
}

}  // namespace profiler
}  // namespace xla
