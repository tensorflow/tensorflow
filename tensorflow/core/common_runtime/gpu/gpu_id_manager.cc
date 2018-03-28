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
// Manages the map between TfGpuId and CUDA GPU id.
class TfToCudaGpuIdMap {
 public:
  static TfToCudaGpuIdMap* singleton() {
    static auto* id_map = new TfToCudaGpuIdMap;
    return id_map;
  }

  void InsertOrDie(TfGpuId tf_gpu_id, CudaGpuId cuda_gpu_id)
      LOCKS_EXCLUDED(mu_) {
    std::pair<IdMapType::iterator, bool> result;
    {
      mutex_lock lock(mu_);
      result = id_map_.insert({tf_gpu_id.value(), cuda_gpu_id.value()});
    }
    if (!result.second) {
      CHECK_EQ(cuda_gpu_id.value(), result.first->second)
          << "Mapping the same TfGpuId to a different CUDA GPU id."
          << " TfGpuId: " << tf_gpu_id
          << " Existing mapped CUDA GPU id: " << result.first->second
          << " CUDA GPU id being tried to map to: " << cuda_gpu_id;
    }
  }

  CudaGpuId FindOrDie(TfGpuId tf_gpu_id) const LOCKS_EXCLUDED(mu_) {
    mutex_lock lock(mu_);
    return FindOrDieLocked(tf_gpu_id);
  }

  bool Find(TfGpuId tf_gpu_id, CudaGpuId* cuda_gpu_id) const
      LOCKS_EXCLUDED(mu_) {
    mutex_lock lock(mu_);
    if (id_map_.count(tf_gpu_id.value()) == 0) return false;
    *cuda_gpu_id = FindOrDieLocked(tf_gpu_id);
    return true;
  }

 private:
  TfToCudaGpuIdMap() = default;

  CudaGpuId FindOrDieLocked(TfGpuId tf_gpu_id) const
      EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    auto result = id_map_.find(tf_gpu_id.value());
    CHECK(result != id_map_.end())
        << "Could not find the mapping for TfGpuId: " << tf_gpu_id;
    return CudaGpuId(result->second);
  }

  void TestOnlyReset() LOCKS_EXCLUDED(mu_) {
    mutex_lock lock(mu_);
    id_map_.clear();
  }

  using IdMapType = std::unordered_map<int32, int32>;
  mutable mutex mu_;
  IdMapType id_map_ GUARDED_BY(mu_);

  friend class ::tensorflow::GpuIdManager;
  TF_DISALLOW_COPY_AND_ASSIGN(TfToCudaGpuIdMap);
};
}  // namespace

void GpuIdManager::InsertTfCudaGpuIdPair(TfGpuId tf_gpu_id,
                                         CudaGpuId cuda_gpu_id) {
  TfToCudaGpuIdMap::singleton()->InsertOrDie(tf_gpu_id, cuda_gpu_id);
}

Status GpuIdManager::TfToCudaGpuId(TfGpuId tf_gpu_id, CudaGpuId* cuda_gpu_id) {
  if (TfToCudaGpuIdMap::singleton()->Find(tf_gpu_id, cuda_gpu_id)) {
    return Status::OK();
  }
  return errors::NotFound("TF GPU device with id ", tf_gpu_id.value(),
                          " was not registered");
}

CudaGpuId GpuIdManager::TfToCudaGpuId(TfGpuId tf_gpu_id) {
  return TfToCudaGpuIdMap::singleton()->FindOrDie(tf_gpu_id);
}

void GpuIdManager::TestOnlyReset() {
  TfToCudaGpuIdMap::singleton()->TestOnlyReset();
}

}  // namespace tensorflow
