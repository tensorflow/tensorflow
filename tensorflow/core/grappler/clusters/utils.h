/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_GRAPPLER_CLUSTERS_UTILS_H_
#define TENSORFLOW_GRAPPLER_CLUSTERS_UTILS_H_

#include "tensorflow/core/protobuf/device_properties.pb.h"
#include "tensorflow/core/util/device_name_utils.h"

namespace tensorflow {
namespace grappler {

// Returns the DeviceProperties of the CPU on which grappler is running.
DeviceProperties GetLocalCPUInfo();

// Returns the DeviceProperties for the specified GPU attached to the server on
// which grappler is running.
DeviceProperties GetLocalGPUInfo(int gpu_id);

// Returns the DeviceProperties of the specified device
DeviceProperties GetDeviceInfo(const DeviceNameUtils::ParsedName& device);

}  // end namespace grappler
}  // end namespace tensorflow

#endif  // TENSORFLOW_GRAPPLER_CLUSTERS_UTILS_H_
