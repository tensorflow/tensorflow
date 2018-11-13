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
#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_DEVICE_RESOLVER_LOCAL_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_DEVICE_RESOLVER_LOCAL_H_

#include <string>

#include "tensorflow/core/framework/collective.h"
#include "tensorflow/core/framework/device_attributes.pb.h"

namespace tensorflow {
class DeviceMgr;

// Implements DeviceResolverInterface in a single-task context.
class DeviceResolverLocal : public DeviceResolverInterface {
 public:
  DeviceResolverLocal(const DeviceMgr* dev_mgr) : dev_mgr_(dev_mgr) {}

  virtual ~DeviceResolverLocal() {}

  void GetDeviceLocalitiesAsync(const CollInstanceParams& ci_params,
                                std::vector<DeviceLocality>* localities,
                                const StatusCallback& done) override;

  void GetLocalityAsync(const string& device, const string& task,
                        DeviceLocality* locality,
                        const StatusCallback& done) override;

  void ClearTask(const string& task) override {}

 protected:
  const DeviceMgr* dev_mgr_;
};

}  // namespace tensorflow
#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_DEVICE_RESOLVER_LOCAL_H_
