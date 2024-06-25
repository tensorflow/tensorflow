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

#include <optional>
#include <string>

#include "absl/base/const_init.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "xla/stream_executor/platform.h"

namespace stream_executor {

// Returns the string representation of the specified PluginKind.
std::string PluginKindString(PluginKind plugin_kind) {
  switch (plugin_kind) {
    case PluginKind::kBlas:
      return "BLAS";
    case PluginKind::kDnn:
      return "DNN";
    case PluginKind::kFft:
      return "FFT";
    case PluginKind::kInvalid:
    default:
      return "kInvalid";
  }
}

static absl::Mutex& GetPluginRegistryMutex() {
  static absl::Mutex mu(absl::kConstInit);
  return mu;
}

/* static */ PluginRegistry* PluginRegistry::instance_ = nullptr;

PluginRegistry::PluginRegistry() {}

/* static */ PluginRegistry* PluginRegistry::Instance() {
  absl::MutexLock lock{&GetPluginRegistryMutex()};
  if (instance_ == nullptr) {
    instance_ = new PluginRegistry();
  }
  return instance_;
}

template <typename FACTORY_TYPE>
absl::Status PluginRegistry::RegisterFactoryInternal(
    const std::string& plugin_name, FACTORY_TYPE factory,
    std::optional<FACTORY_TYPE>* factories) {
  absl::MutexLock lock{&GetPluginRegistryMutex()};

  if (factories->has_value()) {
    return absl::AlreadyExistsError(
        absl::StrFormat("Attempting to register factory for plugin %s when "
                        "one has already been registered",
                        plugin_name));
  }

  (*factories) = factory;
  return absl::OkStatus();
}

bool PluginRegistry::HasFactory(Platform::Id platform_id,
                                PluginKind plugin_kind) const {
  auto iter = factories_.find(platform_id);
  if (iter == factories_.end()) {
    return false;
  }

  switch (plugin_kind) {
    case PluginKind::kBlas:
      return iter->second.blas.has_value();
    case PluginKind::kDnn:
      return iter->second.dnn.has_value();
    case PluginKind::kFft:
      return iter->second.fft.has_value();
    default:
      break;
  }

  LOG(ERROR) << "Invalid plugin kind specified: "
             << PluginKindString(plugin_kind);
  return false;
}

// Explicit instantiations to support types exposed in user/public API.
#define EMIT_PLUGIN_SPECIALIZATIONS(FACTORY_TYPE, FACTORY_VAR, PLUGIN_STRING) \
                                                                              \
  template absl::Status                                                       \
  PluginRegistry::RegisterFactoryInternal<PluginRegistry::FACTORY_TYPE>(      \
      const std::string& plugin_name, PluginRegistry::FACTORY_TYPE factory,   \
      std::optional<PluginRegistry::FACTORY_TYPE>* factories);                \
                                                                              \
  template <>                                                                 \
  absl::Status PluginRegistry::RegisterFactory<PluginRegistry::FACTORY_TYPE>( \
      Platform::Id platform_id, const std::string& name,                      \
      PluginRegistry::FACTORY_TYPE factory) {                                 \
    return RegisterFactoryInternal(name, factory,                             \
                                   &factories_[platform_id].FACTORY_VAR);     \
  }                                                                           \
                                                                              \
  template <>                                                                 \
  absl::StatusOr<PluginRegistry::FACTORY_TYPE> PluginRegistry::GetFactory(    \
      Platform::Id platform_id) {                                             \
    auto plugin_id = factories_[platform_id].FACTORY_VAR;                     \
                                                                              \
    if (!plugin_id.has_value()) {                                             \
      return absl::FailedPreconditionError(                                   \
          "No suitable " PLUGIN_STRING                                        \
          " plugin registered. Have you linked in a " PLUGIN_STRING           \
          "-providing plugin?");                                              \
    } else {                                                                  \
      VLOG(2) << "Selecting default " PLUGIN_STRING " plugin";                \
    }                                                                         \
    return factories_[platform_id].FACTORY_VAR.value();                       \
  }

EMIT_PLUGIN_SPECIALIZATIONS(BlasFactory, blas, "BLAS");
EMIT_PLUGIN_SPECIALIZATIONS(DnnFactory, dnn, "DNN");
EMIT_PLUGIN_SPECIALIZATIONS(FftFactory, fft, "FFT");

}  // namespace stream_executor
