/* Copyright 2017 The OpenXLA Authors.

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

#include "xla/service/platform_util.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/match.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "xla/debug_options_flags.h"
#include "xla/service/compiler.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/cuda/cuda_platform_id.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/host/host_platform_id.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/rocm/rocm_platform_id.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/sycl/sycl_platform_id.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/threadpool.h"
#include "xla/util.h"

namespace xla {

namespace {

// Minimum supported CUDA compute capability is 3.5.
constexpr se::CudaComputeCapability kMinCudaComputeCapability(
    3, 5, se::CudaComputeCapability::FeatureExtension::kNone);

bool IsInterpreter(const se::Platform* p) {
  return absl::AsciiStrToLower(p->Name()) == "interpreter";
}

std::string CanonicalPlatformName(absl::string_view platform_name) {
  std::string lowercase_platform_name = absl::AsciiStrToLower(platform_name);
  // "cpu" and "host" mean the same thing.
  if (lowercase_platform_name == "cpu") {
    return "host";
  }
  // When configured on CUDA, "gpu" and "cuda" mean the same thing.
  // When configured on ROCm, "gpu" and "rocm" mean the same thing.
  // When configured on SYCL, "gpu" and "sycl" mean the same thing.
  if (lowercase_platform_name == "gpu") {
#if TENSORFLOW_USE_ROCM
    return "rocm";
#elif TENSORFLOW_USE_SYCL
    return "sycl";
#else
    return "cuda";
#endif
  }
  return lowercase_platform_name;
}

absl::StatusOr<std::vector<se::Platform*>> GetSupportedPlatforms() {
  return se::PlatformManager::PlatformsWithFilter(
      [](const se::Platform* platform) {
        return Compiler::ExistsForPlatform(platform);
      });
}

// Returns whether the device in StreamExecutor is supported.
absl::Status IsDeviceSupported(se::StreamExecutor* executor) {
  const auto& description = executor->GetDeviceDescription();
  if (executor->GetPlatform()->id() == se::cuda::kCudaPlatformId) {
    // CUDA devices must have a minimum compute capability.
    se::CudaComputeCapability cc = description.cuda_compute_capability();
    if (!cc.SupportsAllFeaturesOf(kMinCudaComputeCapability)) {
      return Internal(
          "StreamExecutor cuda device (%d) requires compute capability '%s' "
          "but is of '%s'",
          executor->device_ordinal(), kMinCudaComputeCapability.ToString(),
          cc.ToString());
    }
  } else if (executor->GetPlatform()->id() == se::rocm::kROCmPlatformId) {
    auto rocm_compute_capability = description.rocm_compute_capability();
    if (!rocm_compute_capability.is_supported_gfx_version()) {
      return Internal(
          "StreamExecutor ROCM device (%d) is of unsupported AMDGPU version "
          "'%s' . The supported AMDGPU versions are '%s'.",
          executor->device_ordinal(), rocm_compute_capability.gfx_version(),
          rocm_compute_capability.supported_gfx_versions_str());
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<se::StreamExecutor*> ExecutorForDevice(se::Platform* platform,
                                                      int device_ordinal) {
  TF_ASSIGN_OR_RETURN(se::StreamExecutor * exec,
                      platform->ExecutorForDevice(device_ordinal));
  TF_RETURN_IF_ERROR(IsDeviceSupported(exec));
  return exec;
}

absl::StatusOr<std::vector<int>> GetDeviceOrdinals(
    se::Platform* platform,
    const std::optional<std::set<int>>& allowed_devices) {
  int device_count = platform->VisibleDeviceCount();
  TF_RET_CHECK(device_count > 0) << "No devices found on " << platform->Name();

  if (platform->id() == se::host::kHostPlatformId) {
    // On host "devices", StreamExecutor exports a device for each hardware
    // thread. Because we parallelize a single computation across threads, it
    // doesn't make sense to expose these as separate devices, so by default we
    // fix the number of devices to one.  However we do let the user override
    // this behavior to help run tests on the host that run models in parallel
    // across multiple devices.
    device_count =
        GetDebugOptionsFromFlags().xla_force_host_platform_device_count();
  }

  std::vector<int> device_ordinals;
  if (allowed_devices.has_value()) {
    device_ordinals.assign(allowed_devices->begin(), allowed_devices->end());
    if (device_ordinals.size() > device_count) {
      LOG(WARNING) << "Allowed device set contains " << device_ordinals.size()
                   << " devices, but platform only sees " << device_count;
      std::sort(device_ordinals.begin(), device_ordinals.end());
      device_ordinals.resize(device_count);
    }
  } else {
    for (int i = 0; i < device_count; ++i) {
      device_ordinals.push_back(i);
    }
  }
  return device_ordinals;
}

}  // namespace

absl::StatusOr<std::string> PlatformUtil::CanonicalPlatformName(
    absl::string_view platform_name) {
  return xla::CanonicalPlatformName(platform_name);
}

absl::StatusOr<se::Platform::Id> PlatformUtil::GetPlatformIdFromCanonicalName(
    absl::string_view platform_name) {
  static const se::Platform::Id kKnownPlatforms[] = {
      se::host::kHostPlatformId,
      se::cuda::kCudaPlatformId,
      se::rocm::kROCmPlatformId,
      se::sycl::kSyclPlatformId,
  };

  for (se::Platform::Id id : kKnownPlatforms) {
    if (absl::EqualsIgnoreCase(id->ToName(), platform_name)) {
      return id;
    }
  }

  return InvalidArgument("Unknown platform name: %s", platform_name);
}

absl::StatusOr<std::vector<se::Platform*>>
PlatformUtil::GetSupportedPlatforms() {
  // Gather all platforms which have an XLA compiler.
  return xla::GetSupportedPlatforms();
}

absl::StatusOr<se::Platform*> PlatformUtil::GetDefaultPlatform() {
  if (const char* allow_default = getenv("XLA_ALLOW_GET_DEFAULT_PLATFORM");
      allow_default != nullptr && strcmp(allow_default, "true") != 0) {
    return FailedPrecondition(
        "GetDefaultPlatform is not allowed "
        "(XLA_ALLOW_GET_DEFAULT_PLATFORM=\"%s\") and the platform must be "
        "specified. If this is a test that has been migrated to PJRT, "
        "double-check that you are using a PJRT-compatible test class.",
        allow_default);
  }
  TF_ASSIGN_OR_RETURN(auto platforms, GetSupportedPlatforms());
  TF_RET_CHECK(!platforms.empty()) << "No platforms found";

  if (platforms.size() == 1) {
    return platforms[0];
  }
  se::Platform* platform = nullptr;
  if (platforms.size() == 2) {
    for (int i = 0; i < 2; i++) {
      if (IsInterpreter(platforms[i]) && !IsInterpreter(platforms[1 - i])) {
        platform = platforms[1 - i];
        break;
      }
    }
  }
  if (platform != nullptr) {
    return platform;
  }

  // Multiple platforms present and we can't pick a reasonable default.
  std::string platforms_string = absl::StrJoin(
      platforms, ", ",
      [](std::string* out, const se::Platform* p) { out->append(p->Name()); });
  return InvalidArgument(
      "Platform must be specified - Multiple platforms found: %s.",
      platforms_string);
}

/*static*/ absl::StatusOr<se::Platform*> PlatformUtil::GetPlatform(
    absl::string_view platform_name) {
  TF_ASSIGN_OR_RETURN(se::Platform * platform,
                      se::PlatformManager::PlatformWithName(
                          xla::CanonicalPlatformName(platform_name)));
  TF_RETURN_IF_ERROR(Compiler::GetForPlatform(platform->id()).status());
  return platform;
}

absl::StatusOr<std::vector<se::StreamExecutor*>>
PlatformUtil::GetStreamExecutors(
    se::Platform* platform,
    const std::optional<std::set<int>>& allowed_devices) {
  TF_ASSIGN_OR_RETURN(std::vector<int> device_ordinals,
                      GetDeviceOrdinals(platform, allowed_devices));

  std::vector<absl::StatusOr<se::StreamExecutor*>> executors(
      device_ordinals.size(),
      absl::UnknownError("StreamExecutor not initialized"));
  {
    tsl::thread::ThreadPool thread_pool(
        tsl::Env::Default(), "device_initialization", device_ordinals.size());
    // Once a stream executor is instantiated it will cause allocations on
    // the device, for example for GPUs cuda context, cudnn handles etc. will
    // be constructed. By constructing stream executors only on the
    // allowed_devices, we don't make any allocations on other devices.
    // This helps in multi-process executions on the same host like horovod or
    // shared hosts.
    for (int i = 0; i < device_ordinals.size(); ++i) {
      thread_pool.Schedule([platform, &executors, &device_ordinals, i]() {
        executors[i] = ExecutorForDevice(platform, device_ordinals[i]);
      });
    }
    // Block here in thread_pool destructor until all devices are initialized.
  }

  std::vector<se::StreamExecutor*> out;
  for (int i = 0; i < executors.size(); ++i) {
    absl::StatusOr<se::StreamExecutor*>& se = executors[i];
    if (se.ok()) {
      out.push_back(*se);
    } else {
      LOG(ERROR) << "Failed to create stream executor for device "
                 << platform->Name() << ":" << device_ordinals[i] << ": "
                 << se.status().message();
    }
  }
  if (out.empty()) {
    return Internal("no supported devices found for platform %s",
                    platform->Name());
  }
  return out;
}

}  // namespace xla
