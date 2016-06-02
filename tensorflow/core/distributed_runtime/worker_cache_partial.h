/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_WORKER_CACHE_PARTIAL_H_
#define TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_WORKER_CACHE_PARTIAL_H_

#include <string>
#include <unordered_map>

#include "tensorflow/core/distributed_runtime/worker_cache.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/protobuf/worker.pb.h"

namespace tensorflow {

// Implements the part of the interface that caches and returns remote
// device status attributes.
class WorkerCachePartial : public WorkerCacheInterface {
 public:
  bool GetDeviceBusNonBlocking(const string& device, BusAdjacency* ba) override;

  void GetDeviceBusAsync(const string& device, BusAdjacency* ba,
                         StatusCallback) override;

  ~WorkerCachePartial() override {}

  // Clear all entries from the DeviceStatus cache.
  void FlushStatusCache();

 private:
  mutex mu_;

  // Initiate a GetStatusAsync to the remote task named by "task", and
  // update the cache with all the DeviceAttributes reported.
  Status RefreshDeviceStatus(const string& device_name);

  typedef std::unordered_map<string, DeviceAttributes> StatusMap;
  StatusMap device_status_cache_ GUARDED_BY(mu_);
};

}  // namespace tensorflow
#endif  // TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_WORKER_CACHE_PARTIAL_H_
