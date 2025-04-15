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

#define EIGEN_USE_THREADS

#include "tensorflow/core/framework/device_base.h"

#include <algorithm>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/synchronization/notification.h"
#include "unsupported/Eigen/CXX11/Tensor"  // from @eigen_archive
#include "tensorflow/core/util/work_sharder.h"

namespace tensorflow {

DeviceBase::~DeviceBase() {
  for (auto& temp : eigen_cpu_devices_) {
    delete temp;
  }
  eigen_cpu_devices_.clear();
}

absl::Status DeviceContext::CopyDeviceTensorToCPUSync(
    const Tensor* device_tensor, absl::string_view tensor_name, Device* device,
    Tensor* cpu_tensor) {
  absl::Notification n;
  absl::Status status;
  CopyDeviceTensorToCPU(device_tensor, tensor_name, device, cpu_tensor,
                        [&](const absl::Status& s) {
                          status = s;
                          n.Notify();
                        });
  n.WaitForNotification();
  return status;
}

absl::Status DeviceContext::CopyCPUTensorToDeviceSync(
    const Tensor* cpu_tensor, Device* device, Tensor* device_tensor) const {
  absl::Notification n;
  absl::Status status;
  CopyCPUTensorToDevice(cpu_tensor, device, device_tensor,
                        [&](const absl::Status& s) {
                          status = s;
                          n.Notify();
                        });
  n.WaitForNotification();
  return status;
}

const DeviceAttributes& DeviceBase::attributes() const {
  LOG(FATAL) << "DeviceBase does not implement attributes()";  // Crash OK
  std::abort();
}

const string& DeviceBase::name() const {
  LOG(FATAL) << "DeviceBase does not implement name()";  // Crash OK
  std::abort();
}

const DeviceNameUtils::ParsedName& DeviceBase::parsed_name() const {
  LOG(FATAL) << "DeviceBase does not implement parsed_name()";  // Crash OK
  std::abort();
}

const std::string& DeviceBase::device_type() const {
  LOG(FATAL) << "DeviceBase does not implement device_type()";  // Crash OK
  std::abort();
}

void DeviceBase::set_eigen_cpu_device(Eigen::ThreadPoolDevice* d) {
  // Eigen::ThreadPoolDevice is a very cheap struct (two pointers and
  // an int).  Therefore, we can afford a pre-allocated array of
  // Eigen::ThreadPoolDevice.  Here, we ensure that
  // Eigen::ThreadPoolDevices in eigen_cpu_devices_ has increasingly
  // larger numThreads.
  for (int i = 1; i <= d->numThreads(); ++i) {
    eigen_cpu_devices_.push_back(new Eigen::ThreadPoolDevice(
        d->getPool(), i /* numThreads() */, d->allocator()));
  }
}

const Eigen::ThreadPoolDevice* DeviceBase::eigen_cpu_device() {
  // Based on GetPerThreadMaxParallelism(), we return a different
  // pre-allocated Eigen::ThreadPoolDevice. All these ThreadPoolDevice
  // use the same underlying threadpool. But they use different
  // nominal numThreads() hoping that the user of the returned
  // Eigen::ThreadPoolDevice may not aggressively occupy all the
  // threads in the underlying threadpool.
  const int parallelism = std::max<int>(
      1,
      std::min<int>(GetPerThreadMaxParallelism(), eigen_cpu_devices_.size()));
  return eigen_cpu_devices_[parallelism - 1];
}

namespace {

absl::flat_hash_set<std::string>* GetSymbolicDeviceList() {
  static absl::flat_hash_set<std::string>* symbolic_device_list =
      new absl::flat_hash_set<std::string>();
  return symbolic_device_list;
}

}  // namespace

void AddSymbolicExecutionDevice(const absl::string_view device_name) {
  GetSymbolicDeviceList()->insert(std::string(device_name));
}

bool IsSymbolicExecutionDevice(const absl::string_view device_name) {
  absl::flat_hash_set<std::string>* symbolic_devices = GetSymbolicDeviceList();
  if (symbolic_devices->contains(device_name)) {
    return true;
  } else {
    return false;
  }
}

}  // namespace tensorflow
