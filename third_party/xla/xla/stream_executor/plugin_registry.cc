/* Copyright 2015 The OpenXLA Authors.

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

#include "xla/stream_executor/plugin_registry.h"

#include <string>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "xla/stream_executor/platform.h"

namespace stream_executor {

/* static */ PluginRegistry* PluginRegistry::Instance() {
  static PluginRegistry* instance = new PluginRegistry();
  return instance;
}

template <typename FactoryT>
PluginKind GetPluginKind() {
  if constexpr (std::is_same_v<FactoryT, PluginRegistry::BlasFactory>) {
    return PluginKind::kBlas;
  } else if constexpr (std::is_same_v<FactoryT, PluginRegistry::DnnFactory>) {
    return PluginKind::kDnn;
  } else if constexpr (std::is_same_v<FactoryT, PluginRegistry::FftFactory>) {
    return PluginKind::kFft;
  } else {
    static_assert(false, "Unsupported factory type");
  }
}
template <typename FactoryT>
absl::string_view GetPluginName() {
  if constexpr (std::is_same_v<FactoryT, PluginRegistry::BlasFactory>) {
    return "BLAS";
  } else if constexpr (std::is_same_v<FactoryT, PluginRegistry::DnnFactory>) {
    return "DNN";
  } else if constexpr (std::is_same_v<FactoryT, PluginRegistry::FftFactory>) {
    return "FFT";
  } else {
    static_assert(false, "Unsupported factory type");
  }
}

template <typename FactoryT>
absl::Status PluginRegistry::RegisterFactory(Platform::Id platform_id,
                                             const std::string& name,
                                             FactoryT factory) {
  PluginKind plugin_kind = GetPluginKind<FactoryT>();
  absl::MutexLock lock(&registry_mutex_);
  auto [_, inserted] = factories_.insert({{platform_id, plugin_kind}, factory});
  if (!inserted) {
    return absl::AlreadyExistsError(
        absl::StrFormat("Attempting to register factory for plugin %s when "
                        "one has already been registered",
                        name));
  }
  return absl::OkStatus();
}

template <typename FactoryT>
absl::StatusOr<FactoryT> PluginRegistry::GetFactory(
    Platform::Id platform_id) const {
  PluginKind plugin_kind = GetPluginKind<FactoryT>();
  absl::MutexLock lock(&registry_mutex_);
  auto it = factories_.find({platform_id, plugin_kind});
  if (it == factories_.end()) {
    absl::string_view name = GetPluginName<FactoryT>();
    return absl::FailedPreconditionError(
        absl::StrFormat("No suitable %s plugin registered. Have you linked in "
                        "a %s-providing plugin?",
                        name, name));
  }
  return std::get<FactoryT>(it->second);
}

bool PluginRegistry::HasFactory(Platform::Id platform_id,
                                PluginKind plugin_kind) const {
  absl::MutexLock lock(&registry_mutex_);
  return factories_.contains({platform_id, plugin_kind});
}

// Explicit instantiations to support types exposed in user/public API.
#define EMIT_PLUGIN_SPECIALIZATIONS(FACTORY_TYPE)                \
                                                                 \
  template absl::Status                                          \
  PluginRegistry::RegisterFactory<PluginRegistry::FACTORY_TYPE>( \
      Platform::Id platform_id, const std::string& name,         \
      PluginRegistry::FACTORY_TYPE factory);                     \
                                                                 \
  template absl::StatusOr<PluginRegistry::FACTORY_TYPE>          \
  PluginRegistry::GetFactory(Platform::Id platform_id) const;

EMIT_PLUGIN_SPECIALIZATIONS(BlasFactory);
EMIT_PLUGIN_SPECIALIZATIONS(DnnFactory);
EMIT_PLUGIN_SPECIALIZATIONS(FftFactory);

}  // namespace stream_executor
