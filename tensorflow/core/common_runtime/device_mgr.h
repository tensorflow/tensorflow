#ifndef TENSORFLOW_COMMON_RUNTIME_DEVICE_MGR_H_
#define TENSORFLOW_COMMON_RUNTIME_DEVICE_MGR_H_

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/public/status.h"

namespace tensorflow {

class DeviceAttributes;

class DeviceMgr {
 public:
  // TODO(zhifengc): Other initialization information.
  explicit DeviceMgr(const std::vector<Device*>& devices);
  ~DeviceMgr();

  // Returns attributes of all devices.
  void ListDeviceAttributes(std::vector<DeviceAttributes>* devices) const;

  std::vector<Device*> ListDevices() const;

  // Returns a string listing all devices.
  string DebugString() const;

  // Returns a string of all the device mapping.
  string DeviceMappingString() const;

  // Assigns *device with pointer to Device of the given name.
  // Accepts either a full device name, or just the replica-local suffix.
  Status LookupDevice(const string& name, Device** device) const;

  // Clears given containers of all devices if 'container' is
  // non-empty. Otherwise, clears default containers of all devices.
  void ClearContainers(gtl::ArraySlice<string> containers) const;

  int NumDeviceType(const string& type) const;

 private:
  typedef gtl::InlinedVector<Device*, 8> DeviceVec;
  DeviceVec devices_;
  std::unordered_map<string, Device*> device_map_;
  std::unordered_map<string, int> device_type_counts_;

  TF_DISALLOW_COPY_AND_ASSIGN(DeviceMgr);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_COMMON_RUNTIME_DEVICE_MGR_H_
