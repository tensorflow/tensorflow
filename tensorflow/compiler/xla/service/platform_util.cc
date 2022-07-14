/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/platform_util.h"

#include <algorithm>
#include <string>
#include <utility>

#include "absl/strings/ascii.h"
#include "absl/strings/str_join.h"
#include "tensorflow/compiler/xla/debug_options_flags.h"
#include "tensorflow/compiler/xla/service/compiler.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"

namespace xla {

// Minimum supported CUDA compute capability is 3.5.
constexpr int kMinCudaComputeCapabilityMajor = 3;
constexpr int kMinCudaComputeCapabilityMinor = 5;

// The name of the interpreter platform.
constexpr char kInterpreter[] = "interpreter";

namespace {

std::string CanonicalPlatformName(const std::string& platform_name) {
  std::string lowercase_platform_name = absl::AsciiStrToLower(platform_name);
  // "cpu" and "host" mean the same thing.
  if (lowercase_platform_name == "cpu") {
    return "host";
  }
  // When configured on CUDA, "gpu" and "cuda" mean the same thing.
  // When configured on ROCm, "gpu" and "rocm" mean the same thing.
  if (lowercase_platform_name == "gpu") {
#if TENSORFLOW_USE_ROCM
    return "rocm";
#else
    return "cuda";
#endif
  }
  return lowercase_platform_name;
}

StatusOr<std::vector<se::Platform*>> GetSupportedPlatforms() {
  return se::MultiPlatformManager::PlatformsWithFilter(
      [](const se::Platform* platform) {
        auto compiler_status = Compiler::GetForPlatform(platform);
        bool supported = compiler_status.ok();
        if (!supported) {
          LOG(INFO) << "platform " << platform->Name() << " present but no "
                    << "XLA compiler available: "
                    << compiler_status.status().error_message();
        }
        return supported;
      });
}

}  // namespace

/* static */ StatusOr<std::vector<se::Platform*>>
PlatformUtil::GetSupportedPlatforms() {
  // Gather all platforms which have an XLA compiler.
  return xla::GetSupportedPlatforms();
}

/* static */ StatusOr<se::Platform*> PlatformUtil::GetDefaultPlatform() {
  TF_ASSIGN_OR_RETURN(auto platforms, GetSupportedPlatforms());

  se::Platform* platform = nullptr;
  if (platforms.empty()) {
    return NotFound("no platforms found");
  } else if (platforms.size() == 1) {
    platform = platforms[0];
  } else if (platforms.size() == 2) {
    for (int i = 0; i < 2; i++) {
      if (absl::AsciiStrToLower(platforms[i]->Name()) == kInterpreter &&
          absl::AsciiStrToLower(platforms[1 - i]->Name()) != kInterpreter) {
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
      "must specify platform because more than one platform (except for the "
      "interpreter platform) found: %s.",
      platforms_string);
}

/*static*/ StatusOr<se::Platform*> PlatformUtil::GetPlatform(
    const std::string& platform_name) {
  TF_ASSIGN_OR_RETURN(se::Platform * platform,
                      se::MultiPlatformManager::PlatformWithName(
                          CanonicalPlatformName(platform_name)));
  TF_RETURN_IF_ERROR(Compiler::GetForPlatform(platform).status());
  return platform;
}

// Returns whether the device underlying the given StreamExecutor is supported
// by XLA.
static bool IsDeviceSupported(se::StreamExecutor* executor) {
  const auto& description = executor->GetDeviceDescription();
  if (executor->platform()->id() == se::cuda::kCudaPlatformId) {
    // CUDA devices must have a minimum compute capability.
    se::CudaComputeCapability cc = description.cuda_compute_capability();
    if (!cc.IsAtLeast(kMinCudaComputeCapabilityMajor,
                      kMinCudaComputeCapabilityMinor)) {
      LOG(INFO) << "StreamExecutor cuda device (" << executor->device_ordinal()
                << ") is of "
                << "insufficient compute capability: "
                << kMinCudaComputeCapabilityMajor << "."
                << kMinCudaComputeCapabilityMinor << " required, "
                << "device is " << cc.ToString();
      return false;
    }
  } else if (executor->platform()->id() == se::rocm::kROCmPlatformId) {
    auto rocm_compute_capability = description.rocm_compute_capability();
    if (!rocm_compute_capability.is_supported_gfx_version()) {
      LOG(INFO) << "StreamExecutor ROCM device (" << executor->device_ordinal()
                << ") is of unsupported "
                << "AMDGPU version : " << rocm_compute_capability.gfx_version()
                << ". The supported AMDGPU versions are "
                << rocm_compute_capability.supported_gfx_versions_str() << ".";
      return false;
    }
  }
  return true;
}

/* static */ StatusOr<std::vector<se::StreamExecutor*>>
PlatformUtil::GetStreamExecutors(
    se::Platform* platform,
    const std::optional<std::set<int>>& allowed_devices) {
  int device_count = platform->VisibleDeviceCount();
  if (device_count <= 0) {
    return NotFound("no %s devices found", platform->Name());
  }
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
  std::vector<se::StreamExecutor*> stream_executors(device_count, nullptr);
  VLOG(1) << "Initializing devices";
  {
    tensorflow::thread::ThreadPool thread_pool(
        tensorflow::Env::Default(), "device_initialization", device_count);
    auto create_fn = [](se::Platform* platform,
                        std::vector<se::StreamExecutor*>& stream_executors,
                        int device_ordinal, int count) {
      VLOG(1) << "Started device init " << device_ordinal;
      auto executor_status = platform->ExecutorForDevice(device_ordinal);
      if (executor_status.ok()) {
        se::StreamExecutor* executor = executor_status.ValueOrDie();
        if (IsDeviceSupported(executor)) {
          stream_executors[count] = executor;
        }
      } else {
        LOG(WARNING) << "unable to create StreamExecutor for "
                     << platform->Name() << ":" << device_ordinal << ": "
                     << executor_status.status().error_message();
      }
      VLOG(1) << "Finished device init " << device_ordinal;
    };
    // Once a stream executor is instantiated it will cause allocations on
    // the device, for example for GPUs cuda context, cudnn handles etc. will
    // be constructed. By constructing stream executors only on the
    // allowed_devices, we don't make any allocations on other devices.
    // This helps in multi-process executions on the same host like horovod or
    // shared hosts.
    if (allowed_devices) {
      int count = 0;
      for (const auto& i : *allowed_devices) {
        if (count >= device_count) {
          break;
        }
        thread_pool.Schedule(
            [platform, &stream_executors, i, count, &create_fn]() {
              create_fn(platform, stream_executors, i, count);
            });
        count++;
      }
    } else {
      for (int i = 0; i < device_count; ++i) {
        thread_pool.Schedule([platform, &stream_executors, i, &create_fn]() {
          create_fn(platform, stream_executors, i, i);
        });
      }
    }
    // Block here in thread_pool destructor until all devices are initialized.
  }
  VLOG(1) << "Device initialization complete";

  std::vector<se::StreamExecutor*> out;
  for (se::StreamExecutor* executor : stream_executors) {
    if (executor != nullptr) {
      out.push_back(executor);
    }
  }
  if (out.empty()) {
    return InternalError("no supported devices found for platform %s",
                         platform->Name());
  }
  return out;
}

}  // namespace xla
