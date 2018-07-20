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

#include "tensorflow/core/common_runtime/gpu/gpu_id_manager.h"

#include <unordered_map>

#include "tensorflow/core/common_runtime/gpu/gpu_id.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"

namespace tensorflow {
namespace {
// Manages the map between TfGpuId and platform GPU id.
class TfToPlatformGpuIdMap {
 public:
  static TfToPlatformGpuIdMap* singleton() {
    static auto* id_map = new TfToPlatformGpuIdMap;
    return id_map;
  }

  Status Insert(TfGpuId tf_gpu_id, PlatformGpuId platform_gpu_id)
      LOCKS_EXCLUDED(mu_) {
    std::pair<IdMapType::iterator, bool> result;
    {
      mutex_lock lock(mu_);
      result = id_map_.insert({tf_gpu_id.value(), platform_gpu_id.value()});
    }
    if (!result.second && platform_gpu_id.value() != result.first->second) {
      return errors::AlreadyExists(
          "TensorFlow device (GPU:", tf_gpu_id.value(),
          ") is being mapped to "
          "multiple CUDA devices (",
          platform_gpu_id.value(), " now, and ", result.first->second,
          " previously), which is not supported. "
          "This may be the result of providing different GPU configurations "
          "(ConfigProto.gpu_options, for example different visible_device_list)"
          " when creating multiple Sessions in the same process. This is not "
          " currently supported, see "
          "https://github.com/tensorflow/tensorflow/issues/19083");
    }
    return Status::OK();
  }

  bool Find(TfGpuId tf_gpu_id, PlatformGpuId* platform_gpu_id) const
      LOCKS_EXCLUDED(mu_) {
    mutex_lock lock(mu_);
    auto result = id_map_.find(tf_gpu_id.value());
    if (result == id_map_.end()) return false;
    *platform_gpu_id = result->second;
    return true;
  }

 private:
  TfToPlatformGpuIdMap() = default;

  void TestOnlyReset() LOCKS_EXCLUDED(mu_) {
    mutex_lock lock(mu_);
    id_map_.clear();
  }

  using IdMapType = std::unordered_map<int32, int32>;
  mutable mutex mu_;
  IdMapType id_map_ GUARDED_BY(mu_);

  friend class ::tensorflow::GpuIdManager;
  TF_DISALLOW_COPY_AND_ASSIGN(TfToPlatformGpuIdMap);
};
}  // namespace

Status GpuIdManager::InsertTfPlatformGpuIdPair(TfGpuId tf_gpu_id,
                                               PlatformGpuId platform_gpu_id) {
  return TfToPlatformGpuIdMap::singleton()->Insert(tf_gpu_id, platform_gpu_id);
}

Status GpuIdManager::TfToPlatformGpuId(TfGpuId tf_gpu_id,
                                       PlatformGpuId* platform_gpu_id) {
  if (TfToPlatformGpuIdMap::singleton()->Find(tf_gpu_id, platform_gpu_id)) {
    return Status::OK();
  }
  return errors::NotFound("TensorFlow device GPU:", tf_gpu_id.value(),
                          " was not registered");
}

void GpuIdManager::TestOnlyReset() {
  TfToPlatformGpuIdMap::singleton()->TestOnlyReset();
}

}  // namespace tensorflow
