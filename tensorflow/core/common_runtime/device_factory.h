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

#ifndef TENSORFLOW_COMMON_RUNTIME_DEVICE_FACTORY_H_
#define TENSORFLOW_COMMON_RUNTIME_DEVICE_FACTORY_H_

#include <string>
#include <vector>

#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

class Device;
struct SessionOptions;

class DeviceFactory {
 public:
  virtual ~DeviceFactory() {}
  static void Register(const string& device_type, DeviceFactory* factory,
                       int priority);
  static DeviceFactory* GetFactory(const string& device_type);

  // Append to "*devices" all suitable devices, respecting
  // any device type specific properties/counts listed in "options".
  //
  // CPU devices are added first.
  static Status AddDevices(const SessionOptions& options,
                           const string& name_prefix,
                           std::vector<Device*>* devices);

  // Helper for tests.  Create a single device of type "type".  The
  // returned device is always numbered zero, so if creating multiple
  // devices of the same type, supply distinct name_prefix arguments.
  static Device* NewDevice(const string& type, const SessionOptions& options,
                           const string& name_prefix);

  // Most clients should call AddDevices() instead.
  virtual Status CreateDevices(const SessionOptions& options,
                               const string& name_prefix,
                               std::vector<Device*>* devices) = 0;
};

namespace dfactory {

template <class Factory>
class Registrar {
 public:
  // Multiple registrations for the same device type with different priorities
  // are allowed. The registration with the highest priority will be used.
  explicit Registrar(const string& device_type, int priority = 0) {
    DeviceFactory::Register(device_type, new Factory(), priority);
  }
};

}  // namespace dfactory

#define REGISTER_LOCAL_DEVICE_FACTORY(device_type, device_factory, ...) \
  INTERNAL_REGISTER_LOCAL_DEVICE_FACTORY(device_type, device_factory,   \
                                         __COUNTER__, ##__VA_ARGS__)

#define INTERNAL_REGISTER_LOCAL_DEVICE_FACTORY(device_type, device_factory, \
                                               ctr, ...)                    \
  static ::tensorflow::dfactory::Registrar<device_factory>                  \
      INTERNAL_REGISTER_LOCAL_DEVICE_FACTORY_NAME(ctr)(device_type,         \
                                                       ##__VA_ARGS__)

// __COUNTER__ must go through another macro to be properly expanded
#define INTERNAL_REGISTER_LOCAL_DEVICE_FACTORY_NAME(ctr) ___##ctr##__object_

}  // namespace tensorflow

#endif  // TENSORFLOW_COMMON_RUNTIME_DEVICE_FACTORY_H_
