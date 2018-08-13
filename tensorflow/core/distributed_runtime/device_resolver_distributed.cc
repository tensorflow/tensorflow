/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/distributed_runtime/device_resolver_distributed.h"

#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/distributed_runtime/worker_cache.h"

namespace tensorflow {
DeviceResolverDistributed::DeviceResolverDistributed(
    const DeviceMgr* dev_mgr, WorkerCacheInterface* worker_cache,
    const string& task_name)
    : dev_mgr_(dev_mgr), worker_cache_(worker_cache), task_name_(task_name) {}

void DeviceResolverDistributed::GetLocalityAsync(const string& device,
                                                 const string& task,
                                                 DeviceLocality* locality,
                                                 const StatusCallback& done) {
  if (task.empty() || task == task_name_) {
    // Device is local to this task.
    Device* dev;
    Status s = dev_mgr_->LookupDevice(device, &dev);
    if (s.ok()) {
      *locality = dev->attributes().locality();
    }
    done(s);
    return;
  } else {
    // Lookup of a remote device: first try the local cache.
    bool found = false;
    {
      mutex_lock l(mu_);
      auto it = attr_table_.find(device);
      if (it != attr_table_.end()) {
        *locality = it->second.locality();
        found = true;
      }
    }
    if (found) {
      done(Status::OK());
      return;
    }
  }
  // Device is remote and no cache entry was found.  Refresh the cache
  // then retry the lookup.
  RefreshRemoteAttributes(
      device, task, [this, device, task, locality, done](const Status& s) {
        if (!s.ok()) {
          done(s);
        } else {
          GetLocalityAsync(device, task, locality, done);
        }
      });
}

void DeviceResolverDistributed::GetDeviceLocalitiesAsync(
    const CollInstanceParams& inst_params,
    std::vector<DeviceLocality>* localities, const StatusCallback& done) {
  localities->clear();
  GetDeviceLocalitiesRecursive(inst_params, localities, done);
}

void DeviceResolverDistributed::GetDeviceLocalitiesRecursive(
    const CollInstanceParams& inst_params,
    std::vector<DeviceLocality>* localities, const StatusCallback& done) {
  size_t i = localities->size();
  if (i < inst_params.device_names.size()) {
    localities->push_back(DeviceLocality());
    GetLocalityAsync(inst_params.device_names[i], inst_params.task_names[i],
                     &localities->back(),
                     [this, &inst_params, localities, done](const Status& s) {
                       if (!s.ok()) {
                         done(s);
                         return;
                       } else {
                         GetDeviceLocalitiesRecursive(inst_params, localities,
                                                      done);
                       }
                     });
  } else {
    done(Status::OK());
  }
}

void DeviceResolverDistributed::RefreshRemoteAttributes(
    const string& device, const string& task, const StatusCallback& done) {
  GetStatusRequest* req = new GetStatusRequest;
  GetStatusResponse* resp = new GetStatusResponse;
  WorkerInterface* worker = worker_cache_->CreateWorker(task);
  CHECK(worker) << "Failed to get worker for " << task;
  worker->GetStatusAsync(
      req, resp, [this, device, task, req, resp, worker, done](Status s) {
        if (s.ok()) {
          mutex_lock l(mu_);
          for (const DeviceAttributes& da : resp->device_attributes()) {
            attr_table_[da.name()] = da;
          }
        }
        done(s);
        delete req;
        delete resp;
        worker_cache_->ReleaseWorker(task, worker);
      });
}

void DeviceResolverDistributed::ClearTask(const string& task) {
  mutex_lock l(mu_);
  // First find all the keys belonging to the task.
  std::unordered_set<string> task_keys;
  for (const auto& it : attr_table_) {
    const string& device_name = it.first;
    if (DeviceNameUtils::IsSameAddressSpace(task, device_name)) {
      task_keys.insert(device_name);
    }
  }
  // Then delete them.
  for (const string& key : task_keys) {
    attr_table_.erase(key);
  }
}

}  // namespace tensorflow
