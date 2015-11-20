/* Copyright 2015 Google Inc. All Rights Reserved.

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

#include "tensorflow/stream_executor/machine_manager.h"

#include "tensorflow/stream_executor/platform/port.h"

#include "tensorflow/stream_executor/dso_loader.h"
#include "tensorflow/stream_executor/lib/error.h"
#include "tensorflow/stream_executor/platform/logging.h"
#include "tensorflow/stream_executor/platform/mutex.h"
#include "tensorflow/stream_executor/platform/port.h"

namespace perftools {
namespace gputools {

mutex MachineManager::mu_{LINKER_INITIALIZED};

MachineManager *MachineManager::singleton_ = nullptr;

PlatformKind MachineManager::DetectPreferredPlatform() {
// TODO(leary) for KNC card experiments, figure out a legitimate way to
// determine this. For now, we use a compile-time hint so we can compile tests
// for both.
#if defined TENSORFLOW_STREAM_EXECUTOR_MACHINE_MANAGER_PREFER_OPENCL
  return PlatformKind::kOpenCL;
#elif defined TENSORFLOW_STREAM_EXECUTOR_MACHINE_MANAGER_PREFER_HOST
  return PlatformKind::kHost;
#else
  return PlatformKind::kCuda;
#endif
}

/* static */ port::StatusOr<std::unique_ptr<MachineManager>>
MachineManager::Create(PlatformKind kind, DeviceOptions options,
                       const PluginConfig &config) {
  std::unique_ptr<MachineManager> machine_manager{
      new MachineManager{kind, options, config}};
  auto init_status = machine_manager->Init();
  if (!init_status.ok()) {
    return init_status;
  }

  return std::move(machine_manager);
}

MachineManager::MachineManager(PlatformKind platform,
                               DeviceOptions device_options,
                               const PluginConfig &config)
    : platform_(platform),
      device_options_(device_options),
      plugin_config_(config),
      min_numa_node_(0),
      limit_numa_node_(0) {}

port::Status MachineManager::Init() {
  // Initialize the first StreamExecutor, then use that platform interface to
  // grab the device count.
  executors_.resize(1);
  executors_[0].reset(new StreamExecutor{platform_, plugin_config_});
  auto status = executors_[0]->Init(0 /* = device_ordinal */, device_options_);
  if (!status.ok()) {
    return port::Status{
        port::error::FAILED_PRECONDITION,
        port::StrCat(
            "failed to initialize StreamExecutor for device ordinal 0: ",
            status.ToString())};
  }
  int device_count = executors_[0]->PlatformDeviceCount();
  if (device_count == 0) {
    LOG(WARNING) << "no devices found for platform "
                 << PlatformKindString(platform_);
    min_numa_node_ = limit_numa_node_ = 0;
    return port::Status::OK();
  }

  streams_.resize(device_count);
  streams_[0].reset(new Stream(executors_[0].get()));
  if (!streams_[0]->Init().ok()) {
    return port::Status{
        port::error::FAILED_PRECONDITION,
        "failed to initialize default stream for device ordinal 0"};
  }

  min_numa_node_ = executors_[0]->GetDeviceDescription().numa_node();
  limit_numa_node_ = min_numa_node_ + 1;

  executors_.resize(device_count);
  for (int device_ordinal = 1; device_ordinal < device_count;
       ++device_ordinal) {
    StreamExecutor *stream_exec = new StreamExecutor{platform_, plugin_config_};
    executors_[device_ordinal].reset(stream_exec);
    auto status = stream_exec->Init(device_ordinal, device_options_);
    if (!status.ok()) {
      return port::Status(
          port::error::FAILED_PRECONDITION,
          port::StrCat(
              "failed to initialize StreamExecutor for device ordinal ",
              device_ordinal, ": ", status.ToString()));
    }

    min_numa_node_ = std::min(min_numa_node_,
                              stream_exec->GetDeviceDescription().numa_node());
    limit_numa_node_ = std::max(
        limit_numa_node_, stream_exec->GetDeviceDescription().numa_node() + 1);

    if (!stream_exec->GetDeviceDescription().ecc_enabled()) {
      LOG(WARNING) << "ECC not enabled for device ordinal: " << device_ordinal;
    }

    streams_[device_ordinal].reset(
        new Stream(executors_[device_ordinal].get()));
    if (!streams_[device_ordinal]->Init().ok()) {
      return port::Status(
          port::error::FAILED_PRECONDITION,
          port::StrCat(
              "failed to initialize default stream for device ordinal ",
              device_ordinal));
    }
  }

  return port::Status::OK();
}

int MachineManager::device_count() const { return executors_.size(); }

port::Status MachineManager::EnablePeerAccess() {
  auto peer_access_map = GetPeerAccessMap();
  for (const auto &access : *peer_access_map) {
    auto devices = access.first;
    if (access.second) {
      StreamExecutor *from = executors_[devices.first].get();
      StreamExecutor *to = executors_[devices.second].get();
      auto status = from->EnablePeerAccessTo(to);
      if (!status.ok()) {
        return status;
      }
    } else {
      LOG(INFO) << "cannot enable peer access from device ordinal "
                << devices.first << " to device ordinal " << devices.second;
    }
  }
  return port::Status::OK();
}

std::unique_ptr<std::map<std::pair<int, int>, bool>>
MachineManager::GetPeerAccessMap() {
  auto *map = new std::map<std::pair<int, int>, bool>;
  for (int i = 0; i < device_count(); ++i) {
    for (int j = 0; j < device_count(); ++j) {
      StreamExecutor *from = executors_[i].get();
      StreamExecutor *to = executors_[j].get();
      (*map)[{i, j}] = from->CanEnablePeerAccessTo(to);
    }
  }

  return std::unique_ptr<std::map<std::pair<int, int>, bool>>{map};
}

StreamExecutor *MachineManager::executor_for_device(int device_ordinal) const {
  CHECK_GE(device_ordinal, 0) << "device ordinal must be non-negative";
  CHECK(0 <= device_ordinal && device_ordinal < device_count())
      << "device " << device_ordinal << " out of range with device count "
      << device_count();
  StreamExecutor *executor = executors_[device_ordinal].get();
  CHECK(executor != nullptr);
  return executor;
}

int MachineManager::ExecutorToBus(const StreamExecutor *stream_exec) const {
  return stream_exec->GetDeviceDescription().numa_node() - min_numa_node_;
}

int MachineManager::DeviceToBus(int device_ordinal) const {
  return ExecutorToBus(executor_for_device(device_ordinal));
}

int MachineManager::ExecutorToNumaNode(
    const StreamExecutor *stream_exec) const {
  return stream_exec->GetDeviceDescription().numa_node();
}

int MachineManager::DeviceToNumaNode(int device_ordinal) const {
  return ExecutorToNumaNode(executor_for_device(device_ordinal));
}

StreamExecutor *MachineManager::first_executor_for_bus(int bus_ordinal) {
  CHECK_LT(bus_ordinal, bus_count()) << "bus ordinal out of available range";
  for (auto &executor : executors_) {
    if (ExecutorToBus(executor.get()) == bus_ordinal) {
      return executor.get();
    }
  }

  LOG(WARNING) << "could not find executor requested for bus ordinal: "
               << bus_ordinal;
  return nullptr;
}

StreamExecutor *MachineManager::first_executor_for_numa_node(int numa_node) {
  for (auto &executor : executors_) {
    if (ExecutorToNumaNode(executor.get()) == numa_node) {
      return executor.get();
    }
  }

  LOG(WARNING) << "could not find executor requested for numa_node: "
               << numa_node;
  return nullptr;
}

Stream *MachineManager::stream_for_device(int device_ordinal) {
  CHECK(0 <= device_ordinal && device_ordinal < device_count());
  Stream *stream = streams_[device_ordinal].get();
  CHECK(stream != nullptr);
  return stream;
}

/* static */ port::StatusOr<MachineManager *>
MachineManager::CreateSingletonInternal(PlatformKind platform,
                                        DeviceOptions options,
                                        const PluginConfig &config) {
  if (singleton_ != nullptr) {
    return port::Status{
        port::error::ALREADY_EXISTS,
        "cannot create machine manager singleton; one already exists"};
  }

  auto create_status = Create(platform, options, config);
  if (!create_status.ok()) {
    return create_status.status();
  }

  singleton_ = create_status.ConsumeValueOrDie().release();

  VLOG(1) << "machine manager singleton is " << singleton_ << " with platform "
          << PlatformKindString(platform) << " and device options "
          << options.ToString();

  return singleton_;
}

/* static */ MachineManager *MachineManager::CreateSingletonOrDie(
    PlatformKind platform, DeviceOptions options, const PluginConfig &config) {
  auto status = CreateSingleton(platform, options, config);
  if (!status.ok()) {
    LOG(FATAL) << "failed to create MachineManager singleton: "
               << status.status();
  }
  return status.ValueOrDie();
}

/* static */ port::StatusOr<MachineManager *> MachineManager::CreateSingleton(
    PlatformKind platform, DeviceOptions device_options,
    const PluginConfig &config) {
  mutex_lock lock{mu_};
  return CreateSingletonInternal(platform, device_options, config);
}

/* static */ MachineManager *MachineManager::singleton() {
  mutex_lock lock{mu_};
  if (singleton_ == nullptr) {
    PlatformKind platform = DetectPreferredPlatform();
    DeviceOptions options = DeviceOptions::Default();
    auto status = CreateSingletonInternal(platform, options, PluginConfig());
    if (!status.ok()) {
      LOG(FATAL)
          << "failed to create MachineManager singleton: "
             "singleton accessor attempted lazy construction but failed: "
          << status.status();
    }
    return status.ValueOrDie();
  }

  return singleton_;
}

}  // namespace gputools
}  // namespace perftools
