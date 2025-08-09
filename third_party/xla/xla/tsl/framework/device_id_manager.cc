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

#include "xla/tsl/framework/device_id_manager.h"

#include <unordered_map>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "xla/tsl/framework/device_id.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/macros.h"
#include "xla/tsl/platform/status.h"
#include "xla/tsl/platform/statusor.h"
#include "tsl/platform/thread_annotations.h"

namespace tsl {
namespace {
// Manages the map between TfDeviceId and platform device id.
class TfToPlatformDeviceIdMap {
 public:
  static TfToPlatformDeviceIdMap* singleton() {
    static auto* const id_map = new TfToPlatformDeviceIdMap;
    return id_map;
  }

  absl::Status Insert(const DeviceType& type, TfDeviceId tf_device_id,
                      PlatformDeviceId platform_device_id)
      TF_LOCKS_EXCLUDED(mu_) {
    std::pair<IdMapType::iterator, bool> result;
    {
      absl::MutexLock lock(&mu_);
      TypeIdMapType::iterator device_id_map_iter =
          id_map_.insert({type.type_string(), IdMapType()}).first;
      result = device_id_map_iter->second.insert(
          {tf_device_id.value(), platform_device_id.value()});
    }
    if (!result.second && platform_device_id.value() != result.first->second) {
      return errors::AlreadyExists(
          "TensorFlow device (", type, ":", tf_device_id.value(),
          ") is being mapped to multiple devices (", platform_device_id.value(),
          " now, and ", result.first->second,
          " previously), which is not supported. "
          "This may be the result of providing different ",
          type, " configurations (ConfigProto.gpu_options, for example ",
          "different visible_device_list) when creating multiple Sessions in ",
          "the same process. This is not currently supported, see ",
          "https://github.com/tensorflow/tensorflow/issues/19083");
    }
    return absl::OkStatus();
  }

  bool Find(const DeviceType& type, TfDeviceId tf_device_id,
            PlatformDeviceId* platform_device_id) const TF_LOCKS_EXCLUDED(mu_) {
    // TODO(mrry): Consider replacing this with an atomic `is_initialized` bit,
    // to avoid writing to a shared cache line in the tf_shared_lock.
    absl::ReaderMutexLock lock(&mu_);
    auto type_id_map_iter = id_map_.find(type.type_string());
    if (type_id_map_iter == id_map_.end()) return false;
    auto id_map_iter = type_id_map_iter->second.find(tf_device_id.value());
    if (id_map_iter == type_id_map_iter->second.end()) return false;
    *platform_device_id = id_map_iter->second;
    return true;
  }

  absl::StatusOr<std::vector<TfDeviceId>> GetTfDevicesOnPlatform(
      const DeviceType& type, PlatformDeviceId platform_device_id) const
      TF_LOCKS_EXCLUDED(mu_) {
    absl::ReaderMutexLock lock(&mu_);
    auto type_id_map_iter = id_map_.find(type.type_string());
    if (type_id_map_iter == id_map_.end()) {
      return absl::NotFoundError(
          absl::StrCat("TensorFlow device type: ", type.type_string(),
                       " was not registered"));
    }
    std::vector<TfDeviceId> tf_device_ids;
    for (const auto& [tf_device, platform_device] : type_id_map_iter->second) {
      if (platform_device == platform_device_id.value()) {
        tf_device_ids.push_back(TfDeviceId(tf_device));
      }
    }
    return tf_device_ids;
  }

 private:
  TfToPlatformDeviceIdMap() = default;

  void TestOnlyReset() TF_LOCKS_EXCLUDED(mu_) {
    absl::MutexLock lock(&mu_);
    id_map_.clear();
  }

  // Map from physical device id to platform device id.
  using IdMapType = std::unordered_map<int32, int32>;
  // Map from DeviceType to IdMapType.
  // We use std::string instead of DeviceType because the key should
  // be default-initializable.
  using TypeIdMapType = std::unordered_map<std::string, IdMapType>;
  mutable absl::Mutex mu_;
  TypeIdMapType id_map_ TF_GUARDED_BY(mu_);

  friend class ::tsl::DeviceIdManager;
  TfToPlatformDeviceIdMap(const TfToPlatformDeviceIdMap&) = delete;
  void operator=(const TfToPlatformDeviceIdMap&) = delete;
};
}  // namespace

absl::Status DeviceIdManager::InsertTfPlatformDeviceIdPair(
    const DeviceType& type, TfDeviceId tf_device_id,
    PlatformDeviceId platform_device_id) {
  return TfToPlatformDeviceIdMap::singleton()->Insert(type, tf_device_id,
                                                      platform_device_id);
}

absl::Status DeviceIdManager::TfToPlatformDeviceId(
    const DeviceType& type, TfDeviceId tf_device_id,
    PlatformDeviceId* platform_device_id) {
  if (TfToPlatformDeviceIdMap::singleton()->Find(type, tf_device_id,
                                                 platform_device_id)) {
    return absl::OkStatus();
  }
  return errors::NotFound("TensorFlow device ", type, ":", tf_device_id.value(),
                          " was not registered");
}

absl::StatusOr<std::vector<TfDeviceId>> DeviceIdManager::GetTfDevicesOnPlatform(
    const DeviceType& type, PlatformDeviceId platform_device_id) {
  return TfToPlatformDeviceIdMap::singleton()->GetTfDevicesOnPlatform(
      type, platform_device_id);
}

void DeviceIdManager::TestOnlyReset() {
  TfToPlatformDeviceIdMap::singleton()->TestOnlyReset();
}

}  // namespace tsl
