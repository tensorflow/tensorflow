/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/stream_executor/rocm/rocm_diagnostics.h"

#include <cstdlib>

#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "rocm/include/hip/hip_runtime.h"
#include "xla/stream_executor/rocm/rocm_status.h"
#include "xla/stream_executor/rocm/rocm_version_parser.h"
#include "xla/stream_executor/semantic_version.h"
#include "tsl/platform/host_info.h"

namespace stream_executor {
namespace rocm {
namespace {

void LogEnv(const char* name) {
  const char* value = std::getenv(name);
  if (value == nullptr) {
    LOG(INFO) << "env: " << name << "=<unset>";
    return;
  }
  LOG(INFO) << "env: " << name << "=\"" << value << "\"";
}

void LogPackedHipVersion(const char* api_name, hipError_t err, int packed) {
  if (err != hipSuccess) {
    LOG(INFO) << api_name << " failed: " << gpu::ToString(err);
    return;
  }
  absl::StatusOr<SemanticVersion> parsed = ParseRocmVersion(packed);
  if (!parsed.ok()) {
    LOG(INFO) << api_name << " packed=" << packed
              << " parse=" << parsed.status();
    return;
  }
  LOG(INFO) << api_name << " packed=" << packed << " parsed=" << *parsed;
}

}  // namespace

void LogDiagnosticInformation() {
  LogEnv("HIP_VISIBLE_DEVICES");
  LogEnv("ROCR_VISIBLE_DEVICES");

  LOG(INFO) << "retrieving ROCm diagnostic information for host: "
            << tsl::port::Hostname();

  int runtime_version = 0;
  hipError_t runtime_err = hipRuntimeGetVersion(&runtime_version);
  LogPackedHipVersion("hipRuntimeGetVersion", runtime_err, runtime_version);

  int driver_version = 0;
  hipError_t driver_err = hipDriverGetVersion(&driver_version);
  LogPackedHipVersion("hipDriverGetVersion", driver_err, driver_version);

  int device_count = 0;
  hipError_t err = hipGetDeviceCount(&device_count);
  if (err != hipSuccess) {
    LOG(INFO) << "hipGetDeviceCount failed: " << gpu::ToString(err);
    return;
  }
  if (device_count <= 0) {
    LOG(INFO) << "no ROCm GPU device is present: hipGetDeviceCount="
              << device_count;
    return;
  }
  LOG(INFO) << "hipGetDeviceCount=" << device_count;

  hipDeviceProp_t props{};
  err = hipGetDeviceProperties(&props, 0);
  if (err != hipSuccess) {
    LOG(INFO) << "hipGetDeviceProperties(0) failed: " << gpu::ToString(err);
    return;
  }
  LOG(INFO) << "device 0 name=\"" << props.name
            << "\" gcnArchName=" << props.gcnArchName;
}

}  // namespace rocm
}  // namespace stream_executor
