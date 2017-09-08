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

#include "tensorflow/compiler/xla/service/compiler.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"

namespace se = ::perftools::gputools;

namespace xla {

// Minimum supported CUDA compute capability is 3.5.
constexpr int kMinCudaComputeCapabilityMajor = 3;
constexpr int kMinCudaComputeCapabilityMinor = 5;

/* static */ StatusOr<std::vector<se::Platform*>>
PlatformUtil::GetSupportedPlatforms() {
  se::MultiPlatformManager::PlatformMap platform_map;
  se::port::Status platforms_status = se::MultiPlatformManager::WithPlatforms(
      [&platform_map](se::MultiPlatformManager::PlatformMap* map) {
        platform_map = *map;
        return se::port::Status::OK();
      });
  if (platform_map.empty()) {
    LOG(WARNING) << "no executor platforms available: platform map is empty";
  }

  // Gather all platforms which have an XLA compiler.
  std::vector<se::Platform*> platforms;
  for (auto& platform_pair : platform_map) {
    auto* platform = platform_pair.second;
    auto compiler_status = Compiler::GetForPlatform(platform);
    if (compiler_status.ok()) {
      if (platform->VisibleDeviceCount() > 0) {
        LOG(INFO) << "platform " << platform->Name() << " present with "
                  << platform->VisibleDeviceCount() << " visible devices";
      } else {
        LOG(WARNING) << "platform " << platform->Name() << " present but no "
                     << "visible devices found";
      }
      // Note: currently we call zero device platforms "supported" on the basis
      // that, if the platform support was linked in, it was probably intended
      // to be used for execution, and this way we can flag an error.
      //
      // TODO(b/33730287) If we want an alternative version of this behavior we
      // could add an --xla_fallback_to_host flag.
      platforms.push_back(platform);
    } else {
      LOG(INFO) << "platform " << platform->Name() << " present but no "
                << "XLA compiler available: "
                << compiler_status.status().error_message();
    }
  }
  return platforms;
}

/* static */ StatusOr<se::Platform*> PlatformUtil::GetDefaultPlatform() {
  TF_ASSIGN_OR_RETURN(auto platforms, GetSupportedPlatforms());
  if (platforms.empty()) {
    return NotFound("no platforms found");
  } else if (platforms.size() == 1) {
    return platforms[0];
  } else if (platforms.size() == 2) {
    // In the service we always link the cpu backend for ComputeConstant. So if
    // one of the two platforms is CPU then pick the other (non-cpu) platform as
    // the default.
    if (platforms[0]->id() == se::host::kHostPlatformId) {
      return platforms[1];
    } else if (platforms[1]->id() == se::host::kHostPlatformId) {
      return platforms[0];
    }
  }

  // Multiple platforms present and we can't pick a reasonable default.
  auto l = [](string* out, const se::Platform* p) { out->append(p->Name()); };
  string platforms_string = tensorflow::str_util::Join(platforms, ", ", l);
  return InvalidArgument(
      "must specify platform because more than one platform found: %s",
      platforms_string.c_str());
}

// Returns whether the device underlying the given StreamExecutor is supported
// by XLA.
static bool IsDeviceSupported(se::StreamExecutor* executor) {
  const auto& description = executor->GetDeviceDescription();
  if (executor->platform()->id() == se::cuda::kCudaPlatformId) {
    // CUDA devices must have a minimum compute capability.
    int major_version, minor_version;
    if (description.cuda_compute_capability(&major_version, &minor_version)) {
      if (major_version < kMinCudaComputeCapabilityMajor ||
          (major_version == kMinCudaComputeCapabilityMajor &&
           minor_version < kMinCudaComputeCapabilityMinor)) {
        LOG(INFO) << "StreamExecutor cuda device ("
                  << executor->device_ordinal() << ") is of "
                  << "insufficient compute capability: "
                  << kMinCudaComputeCapabilityMajor << "."
                  << kMinCudaComputeCapabilityMinor << " required, "
                  << "device is " << major_version << "." << minor_version;
        return false;
      }
    }
  }
  return true;
}

/* static */ StatusOr<std::vector<se::StreamExecutor*>>
PlatformUtil::GetStreamExecutors(se::Platform* platform) {
  int device_count = platform->VisibleDeviceCount();
  if (device_count <= 0) {
    return NotFound("no %s devices found", platform->Name().c_str());
  }
  if (platform->id() == se::host::kHostPlatformId) {
    // On host "devices", StreamExecutor exports a device for each hardware
    // thread. Because we parallelize a single computation across threads, it
    // doesn't make sense to expose these as separate devices, so fix the number
    // of devices to one.
    device_count = 1;
  }
  std::vector<se::StreamExecutor*> stream_executors(device_count, nullptr);
  VLOG(1) << "Initializing devices";
  {
    tensorflow::thread::ThreadPool thread_pool(
        tensorflow::Env::Default(), "device_initialization", device_count);
    for (int i = 0; i < device_count; ++i) {
      thread_pool.Schedule([platform, i, &stream_executors]() {
        VLOG(1) << "Started device init " << i;
        se::StreamExecutorConfig config;
        config.ordinal = i;
        auto executor_status = platform->GetExecutor(config);
        if (executor_status.ok()) {
          se::StreamExecutor* executor = executor_status.ValueOrDie();
          if (IsDeviceSupported(executor)) {
            stream_executors[i] = executor;
          }
        } else {
          LOG(WARNING) << "unable to create StreamExecutor for "
                       << platform->Name() << ":" << i << ": "
                       << executor_status.status().error_message();
        }
        VLOG(1) << "Finished device init " << i;
      });
    }
    // Block here in thread_pool destructor until all devices are initialized.
  }
  VLOG(1) << "Device initialization complete";
  if (std::all_of(stream_executors.begin(), stream_executors.end(),
                  [](se::StreamExecutor* s) { return s == nullptr; })) {
    return InternalError("no supported devices found for platform %s",
                         platform->Name().c_str());
  }
  return stream_executors;
}

}  // namespace xla
