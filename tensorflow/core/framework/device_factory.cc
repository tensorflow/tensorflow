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

#include "tensorflow/core/framework/device_factory.h"

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "tensorflow/core/framework/device.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/util/device_name_utils.h"
#include "tensorflow/core/util/env_var.h"

namespace tensorflow {

namespace {

static mutex* get_device_factory_lock() {
  static mutex device_factory_lock(LINKER_INITIALIZED);
  return &device_factory_lock;
}

struct FactoryItem {
  std::unique_ptr<DeviceFactory> factory;
  int priority;
  bool is_pluggable_device;
};

std::unordered_map<string, FactoryItem>& device_factories() {
  static std::unordered_map<string, FactoryItem>* factories =
      new std::unordered_map<string, FactoryItem>;
  return *factories;
}

bool IsDeviceFactoryEnabled(const string& device_type) {
  std::vector<string> enabled_devices;
  TF_CHECK_OK(tensorflow::ReadStringsFromEnvVar(
      /*env_var_name=*/"TF_ENABLED_DEVICE_TYPES", /*default_val=*/"",
      &enabled_devices));
  if (enabled_devices.empty()) {
    return true;
  }
  return std::find(enabled_devices.begin(), enabled_devices.end(),
                   device_type) != enabled_devices.end();
}
}  // namespace

// static
int32 DeviceFactory::DevicePriority(const string& device_type) {
  tf_shared_lock l(*get_device_factory_lock());
  std::unordered_map<string, FactoryItem>& factories = device_factories();
  auto iter = factories.find(device_type);
  if (iter != factories.end()) {
    return iter->second.priority;
  }

  return -1;
}

bool DeviceFactory::IsPluggableDevice(const string& device_type) {
  tf_shared_lock l(*get_device_factory_lock());
  std::unordered_map<string, FactoryItem>& factories = device_factories();
  auto iter = factories.find(device_type);
  if (iter != factories.end()) {
    return iter->second.is_pluggable_device;
  }
  return false;
}

// static
void DeviceFactory::Register(const string& device_type,
                             std::unique_ptr<DeviceFactory> factory,
                             int priority, bool is_pluggable_device) {
  if (!IsDeviceFactoryEnabled(device_type)) {
    LOG(INFO) << "Device factory '" << device_type << "' disabled by "
              << "TF_ENABLED_DEVICE_TYPES environment variable.";
    return;
  }
  mutex_lock l(*get_device_factory_lock());
  std::unordered_map<string, FactoryItem>& factories = device_factories();
  auto iter = factories.find(device_type);
  if (iter == factories.end()) {
    factories[device_type] = {std::move(factory), priority,
                              is_pluggable_device};
  } else {
    if (iter->second.priority < priority) {
      iter->second = {std::move(factory), priority, is_pluggable_device};
    } else if (iter->second.priority == priority) {
      LOG(FATAL) << "Duplicate registration of device factory for type "
                 << device_type << " with the same priority " << priority;
    }
  }
}

DeviceFactory* DeviceFactory::GetFactory(const string& device_type) {
  tf_shared_lock l(*get_device_factory_lock());
  auto it = device_factories().find(device_type);
  if (it == device_factories().end()) {
    return nullptr;
  } else if (!IsDeviceFactoryEnabled(device_type)) {
    LOG(FATAL) << "Device type " << device_type  // Crash OK
               << " had factory registered but was explicitly disabled by "
               << "`TF_ENABLED_DEVICE_TYPES`. This environment variable needs "
               << "to be set at program startup.";
  }
  return it->second.factory.get();
}

Status DeviceFactory::ListAllPhysicalDevices(std::vector<string>* devices) {
  // CPU first. A CPU device is required.
  // TODO(b/183974121): Consider merge the logic into the loop below.
  auto cpu_factory = GetFactory("CPU");
  if (!cpu_factory) {
    return errors::NotFound(
        "CPU Factory not registered. Did you link in threadpool_device?");
  }

  size_t init_size = devices->size();
  TF_RETURN_IF_ERROR(cpu_factory->ListPhysicalDevices(devices));
  if (devices->size() == init_size) {
    return errors::NotFound("No CPU devices are available in this process");
  }

  // Then the rest (including GPU).
  tf_shared_lock l(*get_device_factory_lock());
  for (auto& p : device_factories()) {
    auto factory = p.second.factory.get();
    if (factory != cpu_factory) {
      TF_RETURN_IF_ERROR(factory->ListPhysicalDevices(devices));
    }
  }

  return OkStatus();
}

Status DeviceFactory::ListPluggablePhysicalDevices(
    std::vector<string>* devices) {
  tf_shared_lock l(*get_device_factory_lock());
  for (auto& p : device_factories()) {
    if (p.second.is_pluggable_device) {
      auto factory = p.second.factory.get();
      TF_RETURN_IF_ERROR(factory->ListPhysicalDevices(devices));
    }
  }
  return OkStatus();
}

Status DeviceFactory::GetAnyDeviceDetails(
    int device_index, std::unordered_map<string, string>* details) {
  if (device_index < 0) {
    return errors::InvalidArgument("Device index out of bounds: ",
                                   device_index);
  }
  const int orig_device_index = device_index;

  // Iterate over devices in the same way as in ListAllPhysicalDevices.
  auto cpu_factory = GetFactory("CPU");
  if (!cpu_factory) {
    return errors::NotFound(
        "CPU Factory not registered. Did you link in threadpool_device?");
  }

  // TODO(b/183974121): Consider merge the logic into the loop below.
  std::vector<string> devices;
  TF_RETURN_IF_ERROR(cpu_factory->ListPhysicalDevices(&devices));
  if (device_index < devices.size()) {
    return cpu_factory->GetDeviceDetails(device_index, details);
  }
  device_index -= devices.size();

  // Then the rest (including GPU).
  tf_shared_lock l(*get_device_factory_lock());
  for (auto& p : device_factories()) {
    auto factory = p.second.factory.get();
    if (factory != cpu_factory) {
      devices.clear();
      // TODO(b/146009447): Find the factory size without having to allocate a
      // vector with all the physical devices.
      TF_RETURN_IF_ERROR(factory->ListPhysicalDevices(&devices));
      if (device_index < devices.size()) {
        return factory->GetDeviceDetails(device_index, details);
      }
      device_index -= devices.size();
    }
  }

  return errors::InvalidArgument("Device index out of bounds: ",
                                 orig_device_index);
}

Status DeviceFactory::AddCpuDevices(
    const SessionOptions& options, const string& name_prefix,
    std::vector<std::unique_ptr<Device>>* devices) {
  auto cpu_factory = GetFactory("CPU");
  if (!cpu_factory) {
    return errors::NotFound(
        "CPU Factory not registered. Did you link in threadpool_device?");
  }
  size_t init_size = devices->size();
  TF_RETURN_IF_ERROR(cpu_factory->CreateDevices(options, name_prefix, devices));
  if (devices->size() == init_size) {
    return errors::NotFound("No CPU devices are available in this process");
  }

  return OkStatus();
}

Status DeviceFactory::AddDevices(
    const SessionOptions& options, const string& name_prefix,
    std::vector<std::unique_ptr<Device>>* devices) {
  // CPU first. A CPU device is required.
  // TODO(b/183974121): Consider merge the logic into the loop below.
  TF_RETURN_IF_ERROR(AddCpuDevices(options, name_prefix, devices));

  absl::flat_hash_set<std::string> allowed_device_types;
  for (const auto& device_filter : options.config.device_filters()) {
    DeviceNameUtils::ParsedName parsed;
    if (!DeviceNameUtils::ParseFullOrLocalName(device_filter, &parsed)) {
      return errors::InvalidArgument(
          absl::StrCat("Invalid device filter: ", device_filter));
    }
    if (parsed.has_type) {
      allowed_device_types.insert(parsed.type);
    }
  }

  // Then Stream Devices
  auto stream_gpu_factory = GetFactory("STREAM_GPU");
  auto stream_cpu_factory = GetFactory("STREAM_CPU");
  if (!stream_gpu_factory) {
    return errors::NotFound(
        "STREAM_GPU Factory not registered. Did you link in "
        "threadpool_device?");
  }
  size_t init_size = devices->size();
  TF_RETURN_IF_ERROR(
      stream_gpu_factory->CreateDevices(options, name_prefix, devices));
  if (devices->size() == init_size) {
    LOG(INFO) << "No STREAM_GPU devices are available in this process";
  } else {
    init_size = devices->size();
    TF_RETURN_IF_ERROR(
        stream_cpu_factory->CreateDevices(options, name_prefix, devices));
    if (devices->size() == init_size) {
      LOG(INFO) << "No STREAM_CPU devices are available in this process";
    }
  }

  auto cpu_factory = GetFactory("CPU");
  // Then the rest (including GPU).
  mutex_lock l(*get_device_factory_lock());
  for (auto& p : device_factories()) {
    if (!allowed_device_types.empty() &&
        !allowed_device_types.contains(p.first)) {
      continue;  // Skip if the device type is not found from the device filter.
    }
    auto factory = p.second.factory.get();
    if (factory != cpu_factory && factory != stream_gpu_factory &&
        factory != stream_cpu_factory) {
      TF_RETURN_IF_ERROR(factory->CreateDevices(options, name_prefix, devices));
    }
  }

  return OkStatus();
}

std::unique_ptr<Device> DeviceFactory::NewDevice(const string& type,
                                                 const SessionOptions& options,
                                                 const string& name_prefix) {
  auto device_factory = GetFactory(type);
  if (!device_factory) {
    return nullptr;
  }
  SessionOptions opt = options;
  (*opt.config.mutable_device_count())[type] = 1;
  std::vector<std::unique_ptr<Device>> devices;
  TF_CHECK_OK(device_factory->CreateDevices(opt, name_prefix, &devices));
  int expected_num_devices = 1;
  auto iter = options.config.device_count().find(type);
  if (iter != options.config.device_count().end()) {
    expected_num_devices = iter->second;
  }
  DCHECK_EQ(devices.size(), static_cast<size_t>(expected_num_devices));
  return std::move(devices[0]);
}

}  // namespace tensorflow
