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

#include "tensorflow/core/framework/collective.h"
#include "tensorflow/core/framework/device_attributes.pb.h"
#include "tensorflow/core/lib/gtl/flatmap.h"

namespace tensorflow {
class DeviceMgr;
class WorkerCacheInterface;

class DeviceResolverDistributed : public DeviceResolverInterface {
 public:
  DeviceResolverDistributed(const DeviceMgr* dev_mgr,
                            WorkerCacheInterface* worker_cache,
                            const string& task_name);

  virtual ~DeviceResolverDistributed() {}

  void GetDeviceLocalitiesAsync(const CollInstanceParams& inst_params,
                                std::vector<DeviceLocality>* localities,
                                const StatusCallback& done) override;

  void GetLocalityAsync(const string& device, const string& task,
                        DeviceLocality* locality,
                        const StatusCallback& done) override;

  void ClearTask(const string& task) override;

 protected:
  // Loads attr_table_ with device attributes retrieved from remote task.
  void RefreshRemoteAttributes(const string& device, const string& task,
                               const StatusCallback& done) LOCKS_EXCLUDED(mu_);

  // Subroutine used by GetDeviceLocalitiesAsync.  Recursively extends
  // *localities with DeviceLocality of the corresponding device named
  // by inst_params.instance.device_names.
  void GetDeviceLocalitiesRecursive(const CollInstanceParams& inst_params,
                                    std::vector<DeviceLocality>* localities,
                                    const StatusCallback& done);

  const DeviceMgr* dev_mgr_;            // Not owned
  WorkerCacheInterface* worker_cache_;  // Not owned
  const string task_name_;
  mutex mu_;
  gtl::FlatMap<string, DeviceAttributes> attr_table_ GUARDED_BY(mu_);
};

}  // namespace tensorflow
#endif  // TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_DEVICE_RESOLVER_DISTRIBUTED_H_
