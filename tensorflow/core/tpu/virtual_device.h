/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_TPU_VIRTUAL_DEVICE_H_
#define TENSORFLOW_CORE_TPU_VIRTUAL_DEVICE_H_

#include "tensorflow/core/common_runtime/device.h"

namespace tensorflow {

// A dummy device that exists primarily for operator placement, without
// corresponding directly to a piece of hardware.
class VirtualDevice : public Device {
 public:
  VirtualDevice(Env* env, const DeviceAttributes& device_attributes);

  Status Sync() override;
  Allocator* GetAllocator(AllocatorAttributes attr) override;
  Status MakeTensorFromProto(const TensorProto& tensor_proto,
                             const AllocatorAttributes alloc_attrs,
                             Tensor* tensor) override;
  Status TryGetDeviceContext(DeviceContext** out_context) override;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_TPU_VIRTUAL_DEVICE_H_
