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

string CanonicalPlatformName(const string& name) {
  string platform_str = absl::AsciiStrToLower(name);
  // "cpu" and "host" mean the same thing.
  if (platform_str == "cpu") {
    platform_str = "host";
  }
  // "gpu" and "cuda" mean the same thing.
  if (platform_str == "gpu") {
    platform_str = "cuda";
  }
  return platform_str;
}

}  // namespace

/* static */ StatusOr<std::vector<se::Platform*>>
PlatformUtil::GetSupportedPlatforms() {
  std::vector<se::Platform*> all_platforms =
      se::MultiPlatformManager::AllPlatforms();
  if (all_platforms.empty()) {
    LOG(WARNING) << "no executor platforms available: platform map is empty";
  }

  // Gather all platforms which have an XLA compiler.
  std::vector<se::Platform*> platforms;
  for (se::Platform* platform : all_platforms) {
    auto compiler_status = Compiler::GetForPlatform(platform);
    if (compiler_status.ok()) {
      platforms.push_back(platform);
    } else {
      LOG(INFO) << "platform " << platform->Name() << " present but no "
                << "XLA compiler available: "
                << compiler_status.status().error_message();
    }
  }
  return platforms;
}

/* static */ StatusOr<se::Platform*> PlatformUtil::GetSolePlatform() {
  TF_ASSIGN_OR_RETURN(auto platforms, GetSupportedPlatforms());
  if (platforms.empty()) {
    return NotFound("no platforms found");
  } else if (platforms.size() == 1) {
    se::Platform* platform = platforms[0];
    if (!platform->Initialized()) {
      TF_RETURN_IF_ERROR(platform->Initialize({}));
    }
    return platform;
  }

  // Multiple platforms present and we can't pick a reasonable default.
  string platforms_string = absl::StrJoin(
      platforms, ", ",
      [](string* out, const se::Platform* p) { out->append(p->Name()); });
  return InvalidArgument(
      "must specify platform because more than one platform found: %s",
      platforms_string);
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
    if (!platform->Initialized()) {
      TF_RETURN_IF_ERROR(platform->Initialize({}));
    }
    return platform;
  }

  // Multiple platforms present and we can't pick a reasonable default.
  string platforms_string = absl::StrJoin(
      platforms, ", ",
      [](string* out, const se::Platform* p) { out->append(p->Name()); });
  return InvalidArgument(
      "must specify platform because more than one platform (except for the "
      "interpreter platform) found: %s",
      platforms_string);
}

/*static*/ StatusOr<se::Platform*> PlatformUtil::GetPlatform(
    const string& platform_name) {
  string platform_str = CanonicalPlatformName(platform_name);
  TF_ASSIGN_OR_RETURN(auto platforms, PlatformUtil::GetSupportedPlatforms());
  for (se::Platform* platform : platforms) {
    if (absl::AsciiStrToLower(platform->Name()) == platform_str) {
      if (!platform->Initialized()) {
        TF_RETURN_IF_ERROR(platform->Initialize({}));
      }
      return platform;
    }
  }
  return InvalidArgument("platform %s not found", platform_name);
}

/*static*/ StatusOr<se::Platform*> PlatformUtil::GetPlatformExceptFor(
    const string& platform_name) {
  string platform_str = CanonicalPlatformName(platform_name);

  TF_ASSIGN_OR_RETURN(auto platforms, PlatformUtil::GetSupportedPlatforms());
  std::vector<se::Platform*> matched;
  for (se::Platform* platform : platforms) {
    if (absl::AsciiStrToLower(platform->Name()) != platform_name) {
      matched.push_back(platform);
    }
  }
  if (matched.empty()) {
    return InvalidArgument("unable to find platform that is not %s",
                           platform_name);
  }
  if (matched.size() == 1) {
    auto platform = matched[0];
    if (!platform->Initialized()) {
      TF_RETURN_IF_ERROR(platform->Initialize({}));
    }
    return platform;
  }
  string matched_string = absl::StrJoin(
      matched, ", ",
      [](string* out, const se::Platform* p) { out->append(p->Name()); });
  return InvalidArgument(
      "found multiple platforms %s, but expected one platform except for %s",
      matched_string, platform_name);
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
PlatformUtil::GetStreamExecutors(
    se::Platform* platform,
    const absl::optional<std::set<int>>& allowed_devices) {
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
    for (int i = 0; i < device_count; ++i) {
      if (allowed_devices && (*allowed_devices).count(i) == 0) {
        VLOG(1) << "Not initializing StreamExecutor for device " << i
                << " since it is not in the visible device list";
        continue;
      }
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
                         platform->Name());
  }
  return stream_executors;
}

}  // namespace xla
