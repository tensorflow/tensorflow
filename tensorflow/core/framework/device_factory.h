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

#ifndef TENSORFLOW_CORE_FRAMEWORK_DEVICE_FACTORY_H_
#define TENSORFLOW_CORE_FRAMEWORK_DEVICE_FACTORY_H_

#include <string>
#include <vector>

#include "absl/base/attributes.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

class Device;
struct SessionOptions;

class DeviceFactory {
 public:
  virtual ~DeviceFactory() {}
  static void Register(const std::string& device_type,
                       std::unique_ptr<DeviceFactory> factory, int priority,
                       bool is_pluggable_device);
  ABSL_DEPRECATED("Use the `Register` function above instead")
  static void Register(const std::string& device_type, DeviceFactory* factory,
                       int priority, bool is_pluggable_device) {
    Register(device_type, std::unique_ptr<DeviceFactory>(factory), priority,
             is_pluggable_device);
  }
  static DeviceFactory* GetFactory(const std::string& device_type);

  // Append to "*devices" CPU devices.
  static absl::Status AddCpuDevices(
      const SessionOptions& options, const std::string& name_prefix,
      std::vector<std::unique_ptr<Device>>* devices);

  // Append to "*devices" all suitable devices, respecting
  // any device type specific properties/counts listed in "options".
  //
  // CPU devices are added first.
  static absl::Status AddDevices(const SessionOptions& options,
                                 const std::string& name_prefix,
                                 std::vector<std::unique_ptr<Device>>* devices);

  // Helper for tests.  Create a single device of type "type".  The
  // returned device is always numbered zero, so if creating multiple
  // devices of the same type, supply distinct name_prefix arguments.
  static std::unique_ptr<Device> NewDevice(const string& type,
                                           const SessionOptions& options,
                                           const string& name_prefix);

  // Iterate through all device factories and build a list of all of the
  // possible physical devices.
  //
  // CPU is are added first.
  static absl::Status ListAllPhysicalDevices(std::vector<string>* devices);

  // Iterate through all device factories and build a list of all of the
  // possible pluggable physical devices.
  static absl::Status ListPluggablePhysicalDevices(
      std::vector<string>* devices);

  // Get details for a specific device among all device factories.
  // 'device_index' indexes into devices from ListAllPhysicalDevices.
  static absl::Status GetAnyDeviceDetails(
      int device_index, std::unordered_map<string, string>* details);

  // For a specific device factory list all possible physical devices.
  virtual absl::Status ListPhysicalDevices(std::vector<string>* devices) = 0;

  // Get details for a specific device for a specific factory. Subclasses
  // can store arbitrary device information in the map. 'device_index' indexes
  // into devices from ListPhysicalDevices.
  virtual absl::Status GetDeviceDetails(
      int device_index, std::unordered_map<string, string>* details) {
    return absl::OkStatus();
  }

  // Most clients should call AddDevices() instead.
  virtual absl::Status CreateDevices(
      const SessionOptions& options, const std::string& name_prefix,
      std::vector<std::unique_ptr<Device>>* devices) = 0;

  // Return the device priority number for a "device_type" string.
  //
  // Higher number implies higher priority.
  //
  // In standard TensorFlow distributions, GPU device types are
  // preferred over CPU, and by default, custom devices that don't set
  // a custom priority during registration will be prioritized lower
  // than CPU.  Custom devices that want a higher priority can set the
  // 'priority' field when registering their device to something
  // higher than the packaged devices.  See calls to
  // REGISTER_LOCAL_DEVICE_FACTORY to see the existing priorities used
  // for built-in devices.
  static int32 DevicePriority(const std::string& device_type);

  // Returns true if 'device_type' is registered from plugin. Returns false if
  // 'device_type' is a first-party device.
  static bool IsPluggableDevice(const std::string& device_type);
};

namespace dfactory {

template <class Factory>
class Registrar {
 public:
  // Multiple registrations for the same device type with different priorities
  // are allowed.  Priorities are used in two different ways:
  //
  // 1) When choosing which factory (that is, which device
  //    implementation) to use for a specific 'device_type', the
  //    factory registered with the highest priority will be chosen.
  //    For example, if there are two registrations:
  //
  //      Registrar<CPUFactory1>("CPU", 125);
  //      Registrar<CPUFactory2>("CPU", 150);
  //
  //    then CPUFactory2 will be chosen when
  //    DeviceFactory::GetFactory("CPU") is called.
  //
  // 2) When choosing which 'device_type' is preferred over other
  //    DeviceTypes in a DeviceSet, the ordering is determined
  //    by the 'priority' set during registration.  For example, if there
  //    are two registrations:
  //
  //      Registrar<CPUFactory>("CPU", 100);
  //      Registrar<GPUFactory>("GPU", 200);
  //
  //    then DeviceType("GPU") will be prioritized higher than
  //    DeviceType("CPU").
  //
  // The default priority values for built-in devices is:
  // GPU: 210
  // GPUCompatibleCPU: 70
  // ThreadPoolDevice: 60
  // Default: 50
  explicit Registrar(const std::string& device_type, int priority = 50) {
    DeviceFactory::Register(device_type, std::make_unique<Factory>(), priority,
                            /*is_pluggable_device*/ false);
  }
};

}  // namespace dfactory

#define REGISTER_LOCAL_DEVICE_FACTORY(device_type, device_factory, ...) \
  INTERNAL_REGISTER_LOCAL_DEVICE_FACTORY(device_type, device_factory,   \
                                         __COUNTER__, ##__VA_ARGS__)

#define INTERNAL_REGISTER_LOCAL_DEVICE_FACTORY(device_type, device_factory, \
                                               ctr, ...)                    \
  static ::tensorflow::dfactory::Registrar<device_factory>                  \
  INTERNAL_REGISTER_LOCAL_DEVICE_FACTORY_NAME(ctr)(device_type, ##__VA_ARGS__)

// __COUNTER__ must go through another macro to be properly expanded
#define INTERNAL_REGISTER_LOCAL_DEVICE_FACTORY_NAME(ctr) ___##ctr##__object_

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_FRAMEWORK_DEVICE_FACTORY_H_
