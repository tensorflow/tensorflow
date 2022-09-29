/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/stream_executor/plugin_registry.h"

#include "absl/base/const_init.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "tensorflow/compiler/xla/stream_executor/lib/error.h"
#include "tensorflow/compiler/xla/stream_executor/multi_platform_manager.h"

namespace stream_executor {

const PluginId kNullPlugin = nullptr;

// Returns the string representation of the specified PluginKind.
std::string PluginKindString(PluginKind plugin_kind) {
  switch (plugin_kind) {
    case PluginKind::kBlas:
      return "BLAS";
    case PluginKind::kDnn:
      return "DNN";
    case PluginKind::kFft:
      return "FFT";
    case PluginKind::kRng:
      return "RNG";
    case PluginKind::kInvalid:
    default:
      return "kInvalid";
  }
}

PluginRegistry::DefaultFactories::DefaultFactories() :
    blas(kNullPlugin), dnn(kNullPlugin), fft(kNullPlugin), rng(kNullPlugin) { }

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

void PluginRegistry::MapPlatformKindToId(PlatformKind platform_kind,
                                         Platform::Id platform_id) {
  platform_id_by_kind_[platform_kind] = platform_id;
}

template <typename FACTORY_TYPE>
port::Status PluginRegistry::RegisterFactoryInternal(
    PluginId plugin_id, const std::string& plugin_name, FACTORY_TYPE factory,
    std::map<PluginId, FACTORY_TYPE>* factories) {
  absl::MutexLock lock{&GetPluginRegistryMutex()};

  if (factories->find(plugin_id) != factories->end()) {
    return port::Status(
        port::error::ALREADY_EXISTS,
        absl::StrFormat("Attempting to register factory for plugin %s when "
                        "one has already been registered",
                        plugin_name));
  }

  (*factories)[plugin_id] = factory;
  plugin_names_[plugin_id] = plugin_name;
  return ::tsl::OkStatus();
}

template <typename FACTORY_TYPE>
port::StatusOr<FACTORY_TYPE> PluginRegistry::GetFactoryInternal(
    PluginId plugin_id, const std::map<PluginId, FACTORY_TYPE>& factories,
    const std::map<PluginId, FACTORY_TYPE>& generic_factories) const {
  auto iter = factories.find(plugin_id);
  if (iter == factories.end()) {
    iter = generic_factories.find(plugin_id);
    if (iter == generic_factories.end()) {
      return port::Status(
          port::error::NOT_FOUND,
          absl::StrFormat("Plugin ID %p not registered.", plugin_id));
    }
  }

  return iter->second;
}

bool PluginRegistry::SetDefaultFactory(Platform::Id platform_id,
                                       PluginKind plugin_kind,
                                       PluginId plugin_id) {
  if (!HasFactory(platform_id, plugin_kind, plugin_id)) {
    port::StatusOr<Platform*> status =
        MultiPlatformManager::PlatformWithId(platform_id);
    std::string platform_name = "<unregistered platform>";
    if (status.ok()) {
      platform_name = status.value()->Name();
    }

    LOG(ERROR) << "A factory must be registered for a platform before being "
               << "set as default! "
               << "Platform name: " << platform_name
               << ", PluginKind: " << PluginKindString(plugin_kind)
               << ", PluginId: " << plugin_id;
    return false;
  }

  switch (plugin_kind) {
    case PluginKind::kBlas:
      default_factories_[platform_id].blas = plugin_id;
      break;
    case PluginKind::kDnn:
      default_factories_[platform_id].dnn = plugin_id;
      break;
    case PluginKind::kFft:
      default_factories_[platform_id].fft = plugin_id;
      break;
    case PluginKind::kRng:
      default_factories_[platform_id].rng = plugin_id;
      break;
    default:
      LOG(ERROR) << "Invalid plugin kind specified: "
                 << static_cast<int>(plugin_kind);
      return false;
  }

  return true;
}

bool PluginRegistry::HasFactory(const PluginFactories& factories,
                                PluginKind plugin_kind,
                                PluginId plugin_id) const {
  switch (plugin_kind) {
    case PluginKind::kBlas:
      return factories.blas.find(plugin_id) != factories.blas.end();
    case PluginKind::kDnn:
      return factories.dnn.find(plugin_id) != factories.dnn.end();
    case PluginKind::kFft:
      return factories.fft.find(plugin_id) != factories.fft.end();
    case PluginKind::kRng:
      return factories.rng.find(plugin_id) != factories.rng.end();
    default:
      LOG(ERROR) << "Invalid plugin kind specified: "
                 << PluginKindString(plugin_kind);
      return false;
  }
}

bool PluginRegistry::HasFactory(Platform::Id platform_id,
                                PluginKind plugin_kind,
                                PluginId plugin_id) const {
  auto iter = factories_.find(platform_id);
  if (iter != factories_.end()) {
    if (HasFactory(iter->second, plugin_kind, plugin_id)) {
      return true;
    }
  }

  return HasFactory(generic_factories_, plugin_kind, plugin_id);
}

// Explicit instantiations to support types exposed in user/public API.
#define EMIT_PLUGIN_SPECIALIZATIONS(FACTORY_TYPE, FACTORY_VAR, PLUGIN_STRING) \
  template port::StatusOr<PluginRegistry::FACTORY_TYPE>                       \
  PluginRegistry::GetFactoryInternal<PluginRegistry::FACTORY_TYPE>(           \
      PluginId plugin_id,                                                     \
      const std::map<PluginId, PluginRegistry::FACTORY_TYPE>& factories,      \
      const std::map<PluginId, PluginRegistry::FACTORY_TYPE>&                 \
          generic_factories) const;                                           \
                                                                              \
  template port::Status                                                       \
  PluginRegistry::RegisterFactoryInternal<PluginRegistry::FACTORY_TYPE>(      \
      PluginId plugin_id, const std::string& plugin_name,                     \
      PluginRegistry::FACTORY_TYPE factory,                                   \
      std::map<PluginId, PluginRegistry::FACTORY_TYPE>* factories);           \
                                                                              \
  template <>                                                                 \
  port::Status PluginRegistry::RegisterFactory<PluginRegistry::FACTORY_TYPE>( \
      Platform::Id platform_id, PluginId plugin_id, const std::string& name,  \
      PluginRegistry::FACTORY_TYPE factory) {                                 \
    return RegisterFactoryInternal(plugin_id, name, factory,                  \
                                   &factories_[platform_id].FACTORY_VAR);     \
  }                                                                           \
                                                                              \
  template <>                                                                 \
  port::Status PluginRegistry::RegisterFactoryForAllPlatforms<                \
      PluginRegistry::FACTORY_TYPE>(PluginId plugin_id,                       \
                                    const std::string& name,                  \
                                    PluginRegistry::FACTORY_TYPE factory) {   \
    return RegisterFactoryInternal(plugin_id, name, factory,                  \
                                   &generic_factories_.FACTORY_VAR);          \
  }                                                                           \
                                                                              \
  template <>                                                                 \
  port::StatusOr<PluginRegistry::FACTORY_TYPE> PluginRegistry::GetFactory(    \
      Platform::Id platform_id, PluginId plugin_id) {                         \
    if (plugin_id == PluginConfig::kDefault) {                                \
      plugin_id = default_factories_[platform_id].FACTORY_VAR;                \
                                                                              \
      if (plugin_id == kNullPlugin) {                                         \
        return port::Status(                                                  \
            port::error::FAILED_PRECONDITION,                                 \
            "No suitable " PLUGIN_STRING                                      \
            " plugin registered. Have you linked in a " PLUGIN_STRING         \
            "-providing plugin?");                                            \
      } else {                                                                \
        VLOG(2) << "Selecting default " PLUGIN_STRING " plugin, "             \
                << plugin_names_[plugin_id];                                  \
      }                                                                       \
    }                                                                         \
    return GetFactoryInternal(plugin_id, factories_[platform_id].FACTORY_VAR, \
                              generic_factories_.FACTORY_VAR);                \
  }                                                                           \
                                                                              \
  /* TODO(b/22689637): Also temporary WRT MultiPlatformManager */             \
  template <>                                                                 \
  port::StatusOr<PluginRegistry::FACTORY_TYPE> PluginRegistry::GetFactory(    \
      PlatformKind platform_kind, PluginId plugin_id) {                       \
    auto iter = platform_id_by_kind_.find(platform_kind);                     \
    if (iter == platform_id_by_kind_.end()) {                                 \
      return port::Status(port::error::FAILED_PRECONDITION,                   \
                          absl::StrFormat("Platform kind %d not registered.", \
                                          static_cast<int>(platform_kind)));  \
    }                                                                         \
    return GetFactory<PluginRegistry::FACTORY_TYPE>(iter->second, plugin_id); \
  }

EMIT_PLUGIN_SPECIALIZATIONS(BlasFactory, blas, "BLAS");
EMIT_PLUGIN_SPECIALIZATIONS(DnnFactory, dnn, "DNN");
EMIT_PLUGIN_SPECIALIZATIONS(FftFactory, fft, "FFT");
EMIT_PLUGIN_SPECIALIZATIONS(RngFactory, rng, "RNG");

}  // namespace stream_executor
