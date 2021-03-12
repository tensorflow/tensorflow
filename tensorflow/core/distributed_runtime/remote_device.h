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

#ifndef TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_REMOTE_DEVICE_H_
#define TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_REMOTE_DEVICE_H_

#include <functional>
#include <string>
#include <vector>

#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/protobuf.h"

namespace tensorflow {
class DeviceAttributes;
class Device;
class Env;
class WorkerCacheInterface;

// This callback should have the same definition as DeviceMgr::LookupDevice
// It assigns *device with pointer to Device of the given 'name', where 'name'
// is either a full device name, or just the replica-local suffix.
typedef std::function<Status(StringPiece name, Device** device)>
    LookupLocalDevice;

// Creates Remote Devices for the provided device attributes. Helpful when the
// list of attributes is known, and doesn't need to be discovered via RPC.
void AsRemoteDevices(
    Env* env,
    const protobuf::RepeatedPtrField<DeviceAttributes>& device_attributes,
    LookupLocalDevice lookup_local_device,
    std::vector<std::unique_ptr<Device>>* remote_devices);

// NewRemoteDevices discovers available devices on the
// 'worker_name'.  The implementation uses 'channel_cache' to
// discover how to communicate with the 'worker_name' (via gRPC, for
// example).
//
// NewRemoteDevices does not block.
//
// On success, the 'done' callback is given the OK status and a vector
// of Device*. The caller should take ownership of these devices.
//
// Otherwise, the 'done' callback is given an error status and the
// vector is empty.
typedef std::function<void(const Status&, std::vector<Device*>*)>
    NewRemoteDevicesDone;
void NewRemoteDevices(Env* env, WorkerCacheInterface* worker_cache,
                      const string& worker_name, NewRemoteDevicesDone done);

// Create Remote Device based on the given attributes.
std::unique_ptr<Device> NewRemoteDevice(Env* env,
                                        DeviceAttributes device_attribute);
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_REMOTE_DEVICE_H_
