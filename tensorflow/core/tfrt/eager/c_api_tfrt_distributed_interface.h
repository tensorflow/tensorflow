/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_TFRT_EAGER_C_API_TFRT_DISTRIBUTED_INTERFACE_H_
#define TENSORFLOW_CORE_TFRT_EAGER_C_API_TFRT_DISTRIBUTED_INTERFACE_H_

#include "tensorflow/c/eager/immediate_execution_distributed_manager.h"

namespace tensorflow {
class DeviceSet;
}
namespace tfrt {
class RequestContextBuilder;

namespace tf {
class DistributedManagerContextInterface
    : public tensorflow::ImmediateExecutionDistributedManager {
 public:
  virtual void UpdateRequestContextBuilder(RequestContextBuilder* builder) = 0;
  virtual void PopulateRemoteDevices(tensorflow::DeviceSet* dev_set) = 0;
};
}  // namespace tf
}  // namespace tfrt

#endif  // TENSORFLOW_CORE_TFRT_EAGER_C_API_TFRT_DISTRIBUTED_INTERFACE_H_
