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

#include "xla/ffi/ffi_registry.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/no_destructor.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/invoke.h"
#include "xla/service/platform_util.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"

namespace xla::ffi {

internal::HandlerRegistrationMap& internal::StaticHandlerRegistrationMap() {
  static absl::NoDestructor<internal::HandlerRegistrationMap> registry;
  return *registry;
}

// The minimum XLA:FFI API version that XLA runtime supports.
static constexpr std::pair<int32_t, int32_t> kMinSupportedApiVersion = {
    /*major=*/0,
    /*minor=*/1,
};

// The maximum XLA:FFI API version that XLA runtime supports.
static constexpr std::pair<int32_t, int32_t> kMaxSupportedApiVersion = {
    XLA_FFI_API_MAJOR,
    XLA_FFI_API_MINOR,
};

static bool IsSupportedApiVersion(const XLA_FFI_Api_Version& api_version) {
  std::pair<int32_t, int32_t> version = {api_version.major_version,
                                         api_version.minor_version};
  return version >= kMinSupportedApiVersion &&
         version <= kMaxSupportedApiVersion;
}

bool IsCommandBufferCompatible(const XLA_FFI_Metadata& metadata) {
  return metadata.traits & XLA_FFI_HANDLER_TRAITS_COMMAND_BUFFER_COMPATIBLE;
}

static std::vector<std::string> GetHandlerStages(
    const XLA_FFI_Handler_Bundle& bundle) {
  std::vector<std::string> stages;
  if (bundle.instantiate != nullptr) {
    stages.push_back("instantiate");
  }
  if (bundle.prepare != nullptr) {
    stages.push_back("prepare");
  }
  if (bundle.initialize != nullptr) {
    stages.push_back("initialize");
  }
  if (bundle.execute != nullptr) {
    stages.push_back("execute");
  }
  return stages;
}

absl::Status RegisterHandler(const XLA_FFI_Api* api, absl::string_view name,
                             absl::string_view platform,
                             XLA_FFI_Handler_Bundle bundle,
                             XLA_FFI_Handler_Traits traits) {
  TF_ASSIGN_OR_RETURN(std::string canonical_platform,
                      PlatformUtil::CanonicalPlatformName(platform));

  if (bundle.execute == nullptr) {
    return InvalidArgument(
        "FFI handler for %s on a platform %s must provide an execute "
        "implementation",
        name, platform);
  }

  // Check the API version that FFI handler was compiled with is supported.
  TF_ASSIGN_OR_RETURN(XLA_FFI_Metadata metadata,
                      GetMetadata(api, bundle.execute));
  if (!IsSupportedApiVersion(metadata.api_version)) {
    return InvalidArgument(
        "XLA FFI handler registration for %s on platform %s (canonical %s) "
        "failed because the handler's API version (%d.%d) is incompatible "
        "with the framework's API version (%d.%d). Minimum supported API "
        "version is (%d.%d).",
        name, platform, canonical_platform, metadata.api_version.major_version,
        metadata.api_version.minor_version, kMaxSupportedApiVersion.first,
        kMaxSupportedApiVersion.second, kMinSupportedApiVersion.first,
        kMinSupportedApiVersion.second);
  }

  // Incorporate handler traits passed explicitly via handler registration API.
  metadata.traits |= traits;

  // Incorporate state type id from the instantiate implementation if present.
  if (bundle.instantiate) {
    TF_ASSIGN_OR_RETURN(XLA_FFI_Metadata instantiate_metadata,
                        GetMetadata(api, bundle.instantiate));
    metadata.state_type_id = instantiate_metadata.state_type_id;
  }

  auto& registry = internal::StaticHandlerRegistrationMap();
  absl::MutexLock lock(registry.mu);

  VLOG(3) << absl::StreamFormat(
      "Register XLA FFI handler for '%s'; platform=%s (canonical=%s), "
      "stages=[%s], metadata=%v, registry=%p",
      name, platform, canonical_platform,
      absl::StrJoin(GetHandlerStages(bundle), ", "), metadata, &registry);

  HandlerRegistration registration{metadata, bundle};
  auto [it, emplaced] = registry.map.try_emplace(
      registry.MakeKey(name, canonical_platform), registration);

  // We might accidentally link the same FFI library multiple times (because
  // linking shared libraries is hard), and we choose to ignore this problem as
  // long as we register exactly the same handler.
  if (!emplaced) {
    const HandlerRegistration& existing = it->second;
    if (existing.metadata != metadata) {
      return InvalidArgument(
          "Duplicate FFI handler registration for %s on platform %s "
          "(canonical %s) with different metadata: %v vs %v",
          name, platform, canonical_platform, existing.metadata, metadata);
    }
    if (existing.bundle != bundle) {
      return InvalidArgument(
          "Duplicate FFI handler registration for %s on platform %s "
          "(canonical %s) with different bundle addresses",
          name, platform, canonical_platform);
    }
  }

  return absl::OkStatus();
}

absl::StatusOr<HandlerRegistration> FindHandler(absl::string_view name,
                                                absl::string_view platform) {
  TF_ASSIGN_OR_RETURN(std::string canonical_platform,
                      PlatformUtil::CanonicalPlatformName(platform));

  auto& registry = internal::StaticHandlerRegistrationMap();
  absl::MutexLock lock(registry.mu);

  auto it = registry.map.find(registry.MakeKey(name, canonical_platform));
  if (it == registry.map.end()) {
    return NotFound(
        "No FFI handler registered for %s on a platform %s (canonical %s)",
        name, platform, canonical_platform);
  }
  return it->second;
}

absl::StatusOr<absl::flat_hash_map<std::string, HandlerRegistration>>
StaticRegisteredHandlers(absl::string_view platform) {
  TF_ASSIGN_OR_RETURN(std::string canonical_platform,
                      PlatformUtil::CanonicalPlatformName(platform));

  auto& registry = internal::StaticHandlerRegistrationMap();
  absl::MutexLock lock(registry.mu);

  absl::flat_hash_map<std::string, HandlerRegistration> calls;
  for (const auto& [metadata, handler] : registry.map) {
    if (canonical_platform == metadata.second) {
      calls[metadata.first] = handler;
    }
  }

  return calls;
}

}  // namespace xla::ffi
