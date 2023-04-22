/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_DEVICE_SET_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_DEVICE_SET_H_

#include <memory>
#include <unordered_map>
#include <vector>

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/device_name_utils.h"

namespace tensorflow {

typedef std::vector<std::pair<Device*, int32>> PrioritizedDeviceVector;

// DeviceSet is a container class for managing the various types of
// devices used by a model.
class DeviceSet {
 public:
  DeviceSet();
  ~DeviceSet();

  // Does not take ownership of 'device'.
  void AddDevice(Device* device) TF_LOCKS_EXCLUDED(devices_mu_);

  // Set the device designated as the "client".  This device
  // must also be registered via AddDevice().
  void set_client_device(Device* device) {
    DCHECK(client_device_ == nullptr);
    client_device_ = device;
  }

  // Returns a pointer to the device designated as the "client".
  Device* client_device() const { return client_device_; }

  // Return the list of devices in this set.
  const std::vector<Device*>& devices() const { return devices_; }

  // Given a DeviceNameUtils::ParsedName (which may have some
  // wildcards for different components), fills "*devices" with all
  // devices in "*this" that match "spec".
  void FindMatchingDevices(const DeviceNameUtils::ParsedName& spec,
                           std::vector<Device*>* devices) const;

  // Finds the device with the given "fullname". Returns nullptr if
  // not found.
  Device* FindDeviceByName(const string& fullname) const;

  // Return the list of unique device types in this set, ordered
  // with more preferable devices earlier.
  std::vector<DeviceType> PrioritizedDeviceTypeList() const;

  // Return the prioritized list of devices in this set.
  // Devices are prioritized first by `DeviceTypeOrder`, then by name.
  const PrioritizedDeviceVector& prioritized_devices() const
      TF_LOCKS_EXCLUDED(devices_mu_);

  // Return the prioritized list of unique device types in this set.
  //
  // The list will be ordered by decreasing priority. The priorities (the second
  // element in the list's `std::pair<DeviceType, int32>`) will be initialized
  // to the value of `DeviceTypeOrder` for the device types.
  const PrioritizedDeviceTypeVector& prioritized_device_types() const
      TF_LOCKS_EXCLUDED(devices_mu_);

  // An order to sort by device types according to system-determined
  // priority.
  //
  // Higher result implies higher priority.
  static int DeviceTypeOrder(const DeviceType& d);

  // Sorts a PrioritizedDeviceVector according to devices and explicit
  // priorities.
  //
  // After a call to this function, the argument vector will be sorted by
  // explicit priority (the second element in the `std::pair<DeviceType,
  // int32>`), then by `DeviceTypeOrder` of the device type, then by device
  // locality, and lastly by device name.
  static void SortPrioritizedDeviceVector(PrioritizedDeviceVector* vector);

  // Sorts a PrioritizedDeviceTypeVector according to types and explicit
  // priorities.
  //
  // After a call to this function, the argument vector will be sorted by
  // explicit priority (the second element in the `std::pair<DeviceType,
  // int32>`), then by `DeviceTypeOrder` of the device type.
  static void SortPrioritizedDeviceTypeVector(
      PrioritizedDeviceTypeVector* vector);

 private:
  mutable mutex devices_mu_;

  // Not owned.
  std::vector<Device*> devices_;

  // Cached prioritized vector, created on-the-fly when
  // prioritized_devices() is called.
  mutable PrioritizedDeviceVector prioritized_devices_
      TF_GUARDED_BY(devices_mu_);

  // Cached prioritized vector, created on-the-fly when
  // prioritized_device_types() is called.
  mutable PrioritizedDeviceTypeVector prioritized_device_types_
      TF_GUARDED_BY(devices_mu_);

  // Fullname -> device* for device in devices_.
  std::unordered_map<string, Device*> device_by_name_;

  // client_device_ points to an element of devices_ that we consider
  // to be the client device (in this local process).
  Device* client_device_ = nullptr;

  TF_DISALLOW_COPY_AND_ASSIGN(DeviceSet);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_DEVICE_SET_H_
