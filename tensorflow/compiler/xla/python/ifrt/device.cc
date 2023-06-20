/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/python/ifrt/device.h"

#include <vector>

namespace xla {
namespace ifrt {

std::vector<int> GetDeviceIds(DeviceList device_list) {
  std::vector<int> ids;
  ids.reserve(device_list.devices().size());
  for (const Device* device : device_list.devices()) {
    ids.push_back(device->id());
  }
  return ids;
}

}  // namespace ifrt
}  // namespace xla
