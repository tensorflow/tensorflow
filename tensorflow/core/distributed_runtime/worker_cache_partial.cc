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

#include "tensorflow/core/distributed_runtime/worker_cache_partial.h"

#include "tensorflow/core/common_runtime/process_util.h"
#include "tensorflow/core/distributed_runtime/worker_interface.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/device_name_utils.h"

namespace tensorflow {

bool WorkerCachePartial::GetDeviceLocalityNonBlocking(
    const string& device_name, DeviceLocality* locality) {
  mutex_lock lock(mu_);  // could use reader lock
  auto iter = device_status_cache_.find(device_name);
  if (iter != device_status_cache_.end()) {
    *locality = iter->second.locality();
    return true;
  }
  return false;
}

void WorkerCachePartial::GetDeviceLocalityAsync(const string& device_name,
                                                DeviceLocality* locality,
                                                StatusCallback done) {
  if (!GetDeviceLocalityNonBlocking(device_name, locality)) {
    // If cache entry was empty, make one try to fill it by RPC.
    SchedClosure([this, &device_name, locality, done]() {
      absl::Status s = RefreshDeviceStatus(device_name);
      if (s.ok() && !GetDeviceLocalityNonBlocking(device_name, locality)) {
        s = errors::Unavailable("No known remote device: ", device_name);
      }
      done(s);
    });
    return;
  }
  done(absl::OkStatus());
}

absl::Status WorkerCachePartial::RefreshDeviceStatus(
    const string& device_name) {
  string task;
  string device;
  absl::Status s;
  if (!DeviceNameUtils::SplitDeviceName(device_name, &task, &device)) {
    s = errors::InvalidArgument("Bad device name to RefreshDeviceStatus: ",
                                device_name);
  }
  auto deleter = [this, &task](WorkerInterface* wi) {
    ReleaseWorker(task, wi);
  };
  std::unique_ptr<WorkerInterface, decltype(deleter)> rwi(
      GetOrCreateWorker(task), deleter);
  if (s.ok() && !rwi) {
    s = errors::Internal("RefreshDeviceStatus, unknown worker task: ", task);
  }

  if (s.ok()) {
    GetStatusRequest req;
    GetStatusResponse resp;
    s = rwi->GetStatus(&req, &resp);
    if (s.ok()) {
      mutex_lock lock(mu_);
      for (auto& dev_attr : resp.device_attributes()) {
        device_status_cache_[dev_attr.name()] = dev_attr;
      }
    }
  }
  return s;
}

void WorkerCachePartial::FlushStatusCache() {
  mutex_lock lock(mu_);
  device_status_cache_.clear();
}

}  // namespace tensorflow
