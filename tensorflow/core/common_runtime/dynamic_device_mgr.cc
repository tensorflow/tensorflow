/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include <atomic>
#include <iterator>
#include <memory>
#include <vector>

#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/local_device.h"
#include "tensorflow/core/framework/device_attributes.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/util/device_name_utils.h"

namespace tensorflow {

mutex DynamicDeviceMgr::mgrs_mu_;
std::unordered_map<const Device*,
                   std::unique_ptr<DynamicDeviceMgr::StreamGroupMgr>>
    DynamicDeviceMgr::stream_group_mgrs_;
size_t DynamicDeviceMgr::max_stream_num_;

DynamicDeviceMgr::DynamicDeviceMgr() : cpu_device_(nullptr) {}

DynamicDeviceMgr::DynamicDeviceMgr(
    std::vector<std::unique_ptr<Device>>&& devices)
    : cpu_device_(nullptr) {
  Status status = AddDevices(std::move(devices));
  CHECK(status.ok());  // Crash OK
  InitStreamDevice();
  mutex_lock l(devices_mu_);
  // Initialize cpu_device_.
  for (const auto& it : dynamic_devices_) {
    Device* d = it.first;
    if (d->device_type() == DEVICE_CPU && d->parsed_name().id == 0) {
      cpu_device_ = d;
      break;
    }
  }
}

DynamicDeviceMgr::DynamicDeviceMgr(std::unique_ptr<Device>&& device)
    : DynamicDeviceMgr([&device] {
        std::vector<std::unique_ptr<Device>> vector;
        vector.push_back(std::move(device));
        return vector;
      }()) {}

DynamicDeviceMgr::~DynamicDeviceMgr() {
  // Release resources ahead of destroying the device manager as the resource
  // destructors (e.g. ~IteratorResource) assume devices still exist.
  mutex_lock l(devices_mu_);
  for (const auto& it : dynamic_devices_) {
    // TODO(tf-runtime-team): clear devices' resource mgr in devices'
    // destructor.
    it.first->ClearResourceMgr();
  }
}

void DynamicDeviceMgr::ListDeviceAttributes(
    std::vector<DeviceAttributes>* devices) const {
  tf_shared_lock l(devices_mu_);
  devices->reserve(dynamic_devices_.size());
  for (const auto& it : dynamic_devices_) {
    devices->emplace_back(it.first->attributes());
  }
}

std::vector<Device*> DynamicDeviceMgr::ListDevices() const {
  tf_shared_lock l(devices_mu_);
  std::vector<Device*> devices;
  devices.reserve(dynamic_devices_.size());
  for (const auto& it : dynamic_devices_) {
    devices.emplace_back(it.first);
  }
  return devices;
}

string DynamicDeviceMgr::DebugString() const {
  string out;
  tf_shared_lock l(devices_mu_);
  for (const auto& it : dynamic_devices_) {
    strings::StrAppend(&out, it.first->name(), "\n");
  }
  return out;
}

string DynamicDeviceMgr::DeviceMappingString() const {
  string out;
  tf_shared_lock l(devices_mu_);
  for (const auto& it : dynamic_devices_) {
    auto d = it.first;
    if (!d->attributes().physical_device_desc().empty()) {
      strings::StrAppend(&out, d->name(), " -> ",
                         d->attributes().physical_device_desc(), "\n");
    }
  }
  return out;
}

Status DynamicDeviceMgr::LookupDevice(StringPiece name, Device** device) const {
  tf_shared_lock l(devices_mu_);
  auto iter = device_map_.find(string(name));
  if (iter == device_map_.end()) {
    std::vector<StringPiece> device_names;
    for (auto&& itr : device_map_) {
      device_names.push_back(itr.first);
    }
    VLOG(1) << "Unknown device: " << name
            << " all devices: " << absl::StrJoin(device_names, ", ");
    return errors::InvalidArgument(name, " unknown device.");
  }
  *device = iter->second;
  return OkStatus();
}

bool DynamicDeviceMgr::ContainsDevice(int64_t device_incarnation) const {
  tf_shared_lock l(devices_mu_);
  return device_incarnation_set_.contains(device_incarnation);
}

void DynamicDeviceMgr::ClearContainers(
    gtl::ArraySlice<string> containers) const {
  Status s;
  tf_shared_lock l(devices_mu_);
  for (const auto& it : dynamic_devices_) {
    auto d = it.first;
    if (containers.empty()) {
      s.Update(d->resource_manager()->Cleanup(
          d->resource_manager()->default_container()));
    } else {
      for (const string& c : containers) {
        s.Update(d->resource_manager()->Cleanup(c));
      }
    }
    if (!s.ok()) {
      LOG(WARNING) << s;
    }
  }
}

int DynamicDeviceMgr::NumDeviceType(const string& type) const {
  tf_shared_lock l(devices_mu_);
  auto iter = device_type_counts_.find(type);
  if (iter != device_type_counts_.end()) return iter->second;
  return 0;
}

int DynamicDeviceMgr::NumDevices() const {
  tf_shared_lock l(devices_mu_);
  return dynamic_devices_.size();
}

Status DynamicDeviceMgr::AddDevices(
    std::vector<std::unique_ptr<Device>> devices) {
  mutex_lock l(devices_mu_);
  for (auto& d : devices) {
    if (device_map_.find(d->name()) != device_map_.end()) {
      return errors::InvalidArgument(
          "Trying to add device ", d->name(),
          " to manager but its name conflicts with an existing device.");
    }
    // Register under the (1) full name and (2) canonical name.
    for (const string& name :
         DeviceNameUtils::GetNamesForDeviceMappings(d->parsed_name())) {
      device_map_[name] = d.get();
    }
    // Register under the (3) local name and (4) legacy local name.
    for (const string& name :
         DeviceNameUtils::GetLocalNamesForDeviceMappings(d->parsed_name())) {
      device_map_[name] = d.get();
    }
    device_type_counts_[d->device_type()]++;
    device_incarnation_set_.insert(d->attributes().incarnation());
    dynamic_devices_.emplace(d.get(), std::move(d));
  }
  return OkStatus();
}

Status DynamicDeviceMgr::RemoveDevices(const std::vector<Device*>& devices) {
  mutex_lock l(devices_mu_);

  for (const auto& d : devices) {
    if (d == cpu_device_) {
      TF_RETURN_IF_ERROR(
          errors::InvalidArgument("Can not remove HostCPU device ", d->name()));
    }
    const auto it = dynamic_devices_.find(d);
    if (it == dynamic_devices_.end()) {
      return errors::InvalidArgument("Unknown device ", d->name());
    }
  }

  for (const auto& d : devices) {
    // Clear registration of (1) full name and (2) canonical name
    for (const string& name :
         DeviceNameUtils::GetNamesForDeviceMappings(d->parsed_name())) {
      device_map_.erase(name);
    }
    // Clear registration of (3) local name and (4) legacy local name
    for (const string& name :
         DeviceNameUtils::GetLocalNamesForDeviceMappings(d->parsed_name())) {
      device_map_.erase(name);
    }
    device_type_counts_[d->device_type()]--;
    device_incarnation_set_.erase(d->attributes().incarnation());

    auto it = dynamic_devices_.find(d);
    if (it == dynamic_devices_.end()) {
      return errors::InvalidArgument("Unknown device ", d->name());
    }
    // There shouldn't be unknown devices at this point.
    CHECK(it != dynamic_devices_.end());  // Crash OK
    stale_devices_.add(std::move(it->second));
    dynamic_devices_.erase(it);
  }
  return OkStatus();
}

Status DynamicDeviceMgr::RemoveDevicesByName(
    const std::vector<string>& device_names) {
  std::vector<Device*> devices_to_remove;
  for (const string& name : device_names) {
    Device* device;
    TF_RETURN_IF_ERROR(LookupDevice(name, &device));
    devices_to_remove.emplace_back(device);
  }
  return RemoveDevices(devices_to_remove);
}

Device* DynamicDeviceMgr::HostCPU() const {
  Device* device = cpu_device_.load(std::memory_order_relaxed);

  // Host CPU device can't be removed, so if we found valid device once, we
  // do not need to check that it is still in the device list.
  if (device != nullptr) return device;

  mutex_lock l(devices_mu_);
  for (const auto& it : dynamic_devices_) {
    Device* d = it.first;
    if (d->device_type() == DEVICE_CPU && d->parsed_name().id == 0) {
      cpu_device_ = d;
      break;
    }
  }

  return cpu_device_.load(std::memory_order_relaxed);
}

void DynamicDeviceMgr::InitStreamDevice() {
  // Counts how many StreamDevices there are within a GPU.
  std::unordered_map<int, size_t> gpu_id2num;
  for (auto& item : dynamic_devices_) {
    Device* d = item.first;
    tsl::StatusOr<int> idx =
        DeviceNameUtils::DecodeDeviceFromStreamDeviceName(d->name());
    if (idx.ok()) {
      if (d->parsed_name().type.find("STREAM_GPU_") != string::npos) {
        if (gpu_id2num.find(idx.value()) == gpu_id2num.end()) {
          gpu_id2num[idx.value()] = 1;
        } else {
          ++gpu_id2num[idx.value()];
        }
      }
    }
  }

  mutex_lock l(mgrs_mu_);
  // Create stream group map and managers.
  Device* gpu;
  for (auto& item : gpu_id2num) {
    TF_CHECK_OK(
        LookupDevice(strings::StrCat("/device:GPU:", item.first), &gpu));
    stream_device_map_[gpu] = std::vector<Device*>(item.second);
    max_stream_num_ =
        max_stream_num_ > item.second ? max_stream_num_ : item.second;
    if (stream_group_mgrs_.find(gpu) == stream_group_mgrs_.end()) {
      stream_group_mgrs_[gpu] = absl::make_unique<StreamGroupMgr>(item.second);
    }
  }

  // Fill in the stream group map and set real device.
  Device* real_device;
  for (auto& item : dynamic_devices_) {
    Device* d = item.first;
    tsl::StatusOr<string> name =
        DeviceNameUtils::GetDeviceNameFromStreamDeviceName(d->name());
    if (name.ok()) {
      TF_CHECK_OK(LookupDevice(name.value(), &real_device));
      stream_device_map_[real_device][d->parsed_name().id] = d;
      d->SetRealDevice(real_device);
    }
  }
}

size_t DynamicDeviceMgr::GetMaxStreamNum() const {
  tf_shared_lock l(mgrs_mu_);
  return max_stream_num_;
}

size_t DynamicDeviceMgr::GetStreamNum(const Device* device) const {
  tf_shared_lock l(mgrs_mu_);
  if (stream_device_map_.find(device) == stream_device_map_.end()) {
    return 0;
  }
  return stream_device_map_.at(device).size();
}

Device* DynamicDeviceMgr::LookupStream(const Device* device,
                                      const int stream_id) const {
  tf_shared_lock l(mgrs_mu_);
  if (stream_id < 0 ||
      stream_device_map_.find(device) == stream_device_map_.end() ||
      stream_device_map_.at(device).size() <= stream_id) {
    return const_cast<Device*>(device);
  }
  return stream_device_map_.at(device).at(stream_id);
}

int DynamicDeviceMgr::RequireStreamGroup(const Device* device) const {
  if (device->parsed_name().type != "GPU" &&
      device->parsed_name().type != "gpu") {
    return -1;
  }
  tf_shared_lock l(mgrs_mu_);
  return stream_group_mgrs_.find(device) == stream_group_mgrs_.end()
             ? -1
             : stream_group_mgrs_[device]->RequireStreamGroup();
}

void DynamicDeviceMgr::ReleaseStreamGroup(const Device* device,
                                         const int stream_id) const {
  if (device->parsed_name().type == "GPU" ||
      device->parsed_name().type == "gpu") {
    tf_shared_lock l(mgrs_mu_);
    if (stream_group_mgrs_.find(device) != stream_group_mgrs_.end()) {
      DCHECK_NE(stream_id, -1);
      stream_group_mgrs_[device]->ReleaseStreamGroup(stream_id);
    }
  }
}

DynamicDeviceMgr::StreamGroupMgr::StreamGroupMgr(const size_t total_num)
    : total_num_(total_num) {
  stream_group_heap_.resize(total_num);
  for (int i = 0; i < total_num; ++i) {
    stream_group_heap_[i] = absl::make_unique<StreamGroupNode>(i);
    id2heap_map_.insert(std::make_pair(i, i));
  }
}

void DynamicDeviceMgr::StreamGroupMgr::swap(const size_t idx1,
                                           const size_t idx2) {
  id2heap_map_[stream_group_heap_[idx1]->id_] = idx2;
  id2heap_map_[stream_group_heap_[idx2]->id_] = idx1;
  std::swap(stream_group_heap_[idx1], stream_group_heap_[idx2]);
}

void DynamicDeviceMgr::StreamGroupMgr::reset_accumulators() {
  VLOG(2) << "One of the Stream Group Node reaches access limit"
          << ", reset...";
  for (auto& node : stream_group_heap_) {
    node->accumulator_ = 0;
  }
}

int DynamicDeviceMgr::StreamGroupMgr::RequireStreamGroup() {
  mutex_lock l(mu_);
  int ret(stream_group_heap_[0]->id_);
  ++stream_group_heap_[0]->workload_;
  if (++stream_group_heap_[0]->accumulator_ == 0xFFFFFFFFFFFFFFFFull) {
    reset_accumulators();
  }
  size_t ptr(0);
  while (true) {
    if (2 * ptr + 2 >= total_num_) {
      if (2 * ptr + 2 == total_num_ &&
          stream_group_heap_[ptr]->workload_ >
              stream_group_heap_[2 * ptr + 1]->workload_) {
        swap(ptr, 2 * ptr + 1);
      }
      break;
    }
    if (stream_group_heap_[2 * ptr + 1]->workload_ <
        stream_group_heap_[2 * ptr + 2]->workload_) {
      if (stream_group_heap_[ptr]->workload_ >
          stream_group_heap_[2 * ptr + 1]->workload_) {
        swap(ptr, 2 * ptr + 1);
        ptr = 2 * ptr + 1;
      } else {
        break;
      }
    } else if (stream_group_heap_[2 * ptr + 1]->workload_ >
               stream_group_heap_[2 * ptr + 2]->workload_) {
      if (stream_group_heap_[ptr]->workload_ >
          stream_group_heap_[2 * ptr + 2]->workload_) {
        swap(ptr, 2 * ptr + 2);
        ptr = 2 * ptr + 2;
      } else {
        break;
      }
    } else {
      if (stream_group_heap_[ptr]->workload_ >
          stream_group_heap_[2 * ptr + 1]->workload_) {
        if (stream_group_heap_[2 * ptr + 1]->accumulator_ <
            stream_group_heap_[2 * ptr + 2]->accumulator_) {
          swap(ptr, 2 * ptr + 1);
          ptr = 2 * ptr + 1;
        } else {
          swap(ptr, 2 * ptr + 2);
          ptr = 2 * ptr + 2;
        }
      } else {
        break;
      }
    }
  }
  return ret;
}

void DynamicDeviceMgr::StreamGroupMgr::ReleaseStreamGroup(const int stream_id) {
  mutex_lock l(mu_);
  size_t ptr = id2heap_map_[stream_id];
  --stream_group_heap_[ptr]->workload_;
  while (ptr != 0) {
    size_t parent = (ptr + 1) / 2 - 1;
    if (stream_group_heap_[ptr]->workload_ <
        stream_group_heap_[parent]->workload_) {
      swap(ptr, parent);
      ptr = parent;
    } else {
      break;
    }
  }
}

}  // namespace tensorflow
