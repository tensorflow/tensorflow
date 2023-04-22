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

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_DEVICE_MGR_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_DEVICE_MGR_H_

#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/lib/core/arena.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/platform/macros.h"

namespace tensorflow {

class DeviceAttributes;

// Represents a set of devices.
class DeviceMgr {
 public:
  DeviceMgr() = default;
  virtual ~DeviceMgr();

  // Returns attributes of all devices.
  virtual void ListDeviceAttributes(
      std::vector<DeviceAttributes>* devices) const = 0;

  // Returns raw pointers to the underlying devices.
  virtual std::vector<Device*> ListDevices() const = 0;

  // Returns a string listing all devices.
  virtual string DebugString() const = 0;

  // Returns a string of all the device mapping.
  virtual string DeviceMappingString() const = 0;

  // Assigns *device with pointer to Device of the given name.
  // Accepts either a full device name, or just the replica-local suffix.
  virtual Status LookupDevice(StringPiece name, Device** device) const = 0;

  // Check if the current device manager contains device with the given
  // incarnation ID. Looking up by incarnation IDs because they are randomly
  // generated and not intentionally reused (unlike device pointers).
  virtual bool ContainsDevice(int64_t device_incarnation) const = 0;

  // Clears given containers of all devices if 'container' is
  // non-empty. Otherwise, clears default containers of all devices.
  virtual void ClearContainers(gtl::ArraySlice<string> containers) const = 0;

  virtual int NumDeviceType(const string& type) const = 0;

  // Returns an arbitrary CPU device if one is present, otherwise return
  // nullptr.
  virtual Device* HostCPU() const = 0;

  TF_DISALLOW_COPY_AND_ASSIGN(DeviceMgr);
};

// Represents a static set of devices.
class StaticDeviceMgr : public DeviceMgr {
 public:
  // Constructs a StaticDeviceMgr from a list of devices.
  explicit StaticDeviceMgr(std::vector<std::unique_ptr<Device>> devices);

  // Constructs a StaticDeviceMgr managing a single device.
  explicit StaticDeviceMgr(std::unique_ptr<Device> device);

  ~StaticDeviceMgr() override;

  void ListDeviceAttributes(
      std::vector<DeviceAttributes>* devices) const override;
  std::vector<Device*> ListDevices() const override;
  string DebugString() const override;
  string DeviceMappingString() const override;
  Status LookupDevice(StringPiece name, Device** device) const override;
  bool ContainsDevice(int64_t device_incarnation) const override;
  void ClearContainers(gtl::ArraySlice<string> containers) const override;
  int NumDeviceType(const string& type) const override;
  Device* HostCPU() const override;

 private:
  const std::vector<std::unique_ptr<Device>> devices_;

  StringPiece CopyToBackingStore(StringPiece s);

  absl::flat_hash_set<int64> device_incarnation_set_;
  std::unordered_map<StringPiece, Device*, StringPieceHasher> device_map_;
  core::Arena name_backing_store_;  // Storage for keys in device_map_
  std::unordered_map<string, int> device_type_counts_;
  Device* cpu_device_;

  TF_DISALLOW_COPY_AND_ASSIGN(StaticDeviceMgr);
};

// Size of stale device buffer for temporary storage of removed devices.
static const size_t kStaleDeviceBufferSize = 8192;

// Represents a dynamic set of devices
class DynamicDeviceMgr : public DeviceMgr {
 public:
  // Constructs an empty DynamicDeviceMgr.
  DynamicDeviceMgr();

  // Constructs a DynamicDeviceMgr from a list of devices.
  // TODO(b/183966398): Remove StaticDeviceMgr since there's no usage.
  explicit DynamicDeviceMgr(std::vector<std::unique_ptr<Device>> devices);

  ~DynamicDeviceMgr() override;

  void ListDeviceAttributes(
      std::vector<DeviceAttributes>* devices) const override;
  std::vector<Device*> ListDevices() const override;
  string DebugString() const override;
  string DeviceMappingString() const override;
  Status LookupDevice(StringPiece name, Device** device) const override;
  bool ContainsDevice(int64_t device_incarnation) const override;
  void ClearContainers(gtl::ArraySlice<string> containers) const override;
  int NumDeviceType(const string& type) const override;
  Device* HostCPU() const override;

  // Add devices to device manager. Returns error for repeated device names.
  Status AddDevices(std::vector<std::unique_ptr<Device>> devices);

  // Remove devices from device manager.
  // Returns error for non-existing devices or if the HostCPU() device is in the
  // input list. If an error is returned, the device list is not modified.
  Status RemoveDevices(const std::vector<Device*>& devices);

  // Remove devices from device manager by their names. Returns error for
  // non-existing devices or if the HostCPU() device is given in the input list.
  // If an error is returned, the device list is not modified.
  Status RemoveDevicesByName(const std::vector<string>& device_names);

 private:
  mutable mutex devices_mu_;

  std::vector<std::unique_ptr<Device>> dynamic_devices_
      TF_GUARDED_BY(devices_mu_);

  absl::flat_hash_set<int64> device_incarnation_set_ TF_GUARDED_BY(devices_mu_);
  std::unordered_map<string, Device*> device_map_ TF_GUARDED_BY(devices_mu_);

  std::unordered_map<string, int> device_type_counts_
      TF_GUARDED_BY(devices_mu_);

  mutable std::atomic<Device*> cpu_device_;  // memoize `HostCPU` result

  class DeviceCircularBuffer {
   public:
    DeviceCircularBuffer() : index_(0) {
      devices_.resize(kStaleDeviceBufferSize);
    }
    void add(std::unique_ptr<Device> device) {
      devices_[index_] = std::move(device);
      index_ = (index_ + 1) % kStaleDeviceBufferSize;
    }

   private:
    int index_;
    std::vector<std::unique_ptr<Device>> devices_;
  };

  // Buffer to temporarily store the removed devices. Raw device pointers are
  // accessible to DeviceSet, and if the function instantiation process directly
  // access fields through the device set, the underlying device object must
  // still be available to avoid segmentation fault. We keep the devices in this
  // buffer only for that purpose.
  DeviceCircularBuffer stale_devices_ TF_GUARDED_BY(devices_mu_);

  TF_DISALLOW_COPY_AND_ASSIGN(DynamicDeviceMgr);
};
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_DEVICE_MGR_H_
