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

#include "tensorflow/core/common_runtime/device_factory.h"

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {

namespace {

static mutex* get_device_factory_lock() {
  static mutex device_factory_lock;
  return &device_factory_lock;
}

struct FactoryItem {
  std::unique_ptr<DeviceFactory> factory;
  int priority;
};

std::unordered_map<string, FactoryItem>& device_factories() {
  static std::unordered_map<string, FactoryItem>* factories =
      new std::unordered_map<string, FactoryItem>;
  return *factories;
}

}  // namespace

// static
int32 DeviceFactory::DevicePriority(const string& device_type) {
  mutex_lock l(*get_device_factory_lock());
  std::unordered_map<string, FactoryItem>& factories = device_factories();
  auto iter = factories.find(device_type);
  if (iter != factories.end()) {
    return iter->second.priority;
  }

  return -1;
}

// static
void DeviceFactory::Register(const string& device_type, DeviceFactory* factory,
                             int priority) {
  mutex_lock l(*get_device_factory_lock());
  std::unique_ptr<DeviceFactory> factory_ptr(factory);
  std::unordered_map<string, FactoryItem>& factories = device_factories();
  auto iter = factories.find(device_type);
  if (iter == factories.end()) {
    factories[device_type] = {std::move(factory_ptr), priority};
  } else {
    if (iter->second.priority < priority) {
      iter->second = {std::move(factory_ptr), priority};
    } else if (iter->second.priority == priority) {
      LOG(FATAL) << "Duplicate registration of device factory for type "
                 << device_type << " with the same priority " << priority;
    }
  }
}

DeviceFactory* DeviceFactory::GetFactory(const string& device_type) {
  mutex_lock l(*get_device_factory_lock());  // could use reader lock
  auto it = device_factories().find(device_type);
  if (it == device_factories().end()) {
    return nullptr;
  }
  return it->second.factory.get();
}

Status DeviceFactory::AddDevices(const SessionOptions& options,
                                 const string& name_prefix,
                                 std::vector<Device*>* devices) {
  // CPU first. A CPU device is required.
  auto cpu_factory = GetFactory("CPU");
  if (!cpu_factory) {
    return errors::NotFound(
        "CPU Factory not registered.  Did you link in threadpool_device?");
  }
  size_t init_size = devices->size();
  TF_RETURN_IF_ERROR(cpu_factory->CreateDevices(options, name_prefix, devices));
  if (devices->size() == init_size) {
    return errors::NotFound("No CPU devices are available in this process");
  }

  // Then the rest (including GPU).
  mutex_lock l(*get_device_factory_lock());
  for (auto& p : device_factories()) {
    auto factory = p.second.factory.get();
    if (factory != cpu_factory) {
      TF_RETURN_IF_ERROR(factory->CreateDevices(options, name_prefix, devices));
    }
  }

  return Status::OK();
}

Device* DeviceFactory::NewDevice(const string& type,
                                 const SessionOptions& options,
                                 const string& name_prefix) {
  auto device_factory = GetFactory(type);
  if (!device_factory) {
    return nullptr;
  }
  SessionOptions opt = options;
  (*opt.config.mutable_device_count())[type] = 1;
  std::vector<Device*> devices;
  TF_CHECK_OK(device_factory->CreateDevices(opt, name_prefix, &devices));
  CHECK_EQ(devices.size(), size_t{1});
  return devices[0];
}

}  // namespace tensorflow
