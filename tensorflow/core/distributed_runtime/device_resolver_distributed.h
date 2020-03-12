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
#ifndef TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_DEVICE_RESOLVER_DISTRIBUTED_H_
#define TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_DEVICE_RESOLVER_DISTRIBUTED_H_

#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "tensorflow/core/framework/collective.h"
#include "tensorflow/core/framework/device_attributes.pb.h"

namespace tensorflow {
class DeviceMgr;
class WorkerCacheInterface;

class DeviceResolverDistributed : public DeviceResolverInterface {
 public:
  DeviceResolverDistributed(const DeviceMgr* dev_mgr,
                            WorkerCacheInterface* worker_cache,
                            const string& task_name);

  virtual ~DeviceResolverDistributed() {}

  void GetAllDeviceAttributesAsync(const std::vector<string>& devices,
                                   const std::vector<string>& tasks,
                                   std::vector<DeviceAttributes>* attributes,
                                   const StatusCallback& done) override;

  void GetDeviceAttributesAsync(const string& device, const string& task,
                                DeviceAttributes* attributes,
                                const StatusCallback& done) override;

  void ClearTask(const string& task) override;

  void ClearCache() override;

 protected:
  // Loads attr_table_ with device attributes retrieved from remote task.
  void RefreshRemoteAttributes(const string& device, const string& task,
                               const StatusCallback& done)
      TF_LOCKS_EXCLUDED(mu_);

  // Subroutine used by GetAllDeviceAttributesAsync.  Recursively extends
  // *attributes with DeviceAttributes of the corresponding device named
  // by inst_params.instance.device_names.
  void GetAllDeviceAttributesRecursive(
      const std::vector<string>& devices, const std::vector<string>& tasks,
      std::vector<DeviceAttributes>* attributes, const StatusCallback& done);

  const DeviceMgr* dev_mgr_;            // Not owned
  WorkerCacheInterface* worker_cache_;  // Not owned
  const string task_name_;
  mutex mu_;
  absl::flat_hash_map<string, DeviceAttributes> attr_table_ TF_GUARDED_BY(mu_);
};

}  // namespace tensorflow
#endif  // TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_DEVICE_RESOLVER_DISTRIBUTED_H_
