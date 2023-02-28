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

#include "tensorflow/core/common_runtime/device_mgr.h"

#include <memory>
#include <vector>

#include "tensorflow/core/common_runtime/local_device.h"
#include "tensorflow/core/framework/device_attributes.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/util/device_name_utils.h"

namespace tensorflow {

DeviceMgr::~DeviceMgr() {}

mutex StaticDeviceMgr::mgrs_mu_;
std::unordered_map<const Device*,
                   std::unique_ptr<StaticDeviceMgr::StreamGroupMgr>>
    StaticDeviceMgr::stream_group_mgrs_;
int32 StaticDeviceMgr::max_stream_num_;

StaticDeviceMgr::StaticDeviceMgr(std::vector<std::unique_ptr<Device>> devices)
    : devices_(std::move(devices)),
      name_backing_store_(128),
      cpu_device_(nullptr) {
  for (auto& d : devices_) {
    // Register under the (1) full name and (2) canonical name.
    for (const string& name :
         DeviceNameUtils::GetNamesForDeviceMappings(d->parsed_name())) {
      device_map_[CopyToBackingStore(name)] = d.get();
    }
    // Register under the (3) local name and (4) legacy local name.
    for (const string& name :
         DeviceNameUtils::GetLocalNamesForDeviceMappings(d->parsed_name())) {
      device_map_[CopyToBackingStore(name)] = d.get();
    }
    const auto& t = d->device_type();
    device_type_counts_[t]++;
    device_incarnation_set_.insert(d->attributes().incarnation());
    if (cpu_device_ == nullptr && t == "CPU" && d->parsed_name().id == 0) {
      cpu_device_ = d.get();
    }
  }
  InitStreamDevice();
}

StaticDeviceMgr::StaticDeviceMgr(std::unique_ptr<Device> device)
    : StaticDeviceMgr([&device] {
        std::vector<std::unique_ptr<Device>> vector;
        vector.push_back(std::move(device));
        return vector;
      }()) {}

StaticDeviceMgr::~StaticDeviceMgr() {
  // Release resources ahead of destroying the device manager as the resource
  // destructors (e.g. ~IteratorResource) assume devices still exist.
  for (auto& device : devices_) {
    device->ClearResourceMgr();
  }
}

StringPiece StaticDeviceMgr::CopyToBackingStore(StringPiece s) {
  size_t n = s.size();
  char* space = name_backing_store_.Alloc(n);
  memcpy(space, s.data(), n);
  return StringPiece(space, n);
}

void StaticDeviceMgr::ListDeviceAttributes(
    std::vector<DeviceAttributes>* devices) const {
  devices->reserve(devices_.size());
  for (const auto& dev : devices_) {
    devices->emplace_back(dev->attributes());
  }
}

std::vector<Device*> StaticDeviceMgr::ListDevices() const {
  std::vector<Device*> devices(devices_.size());
  for (size_t i = 0; i < devices_.size(); ++i) {
    devices[i] = devices_[i].get();
  }
  return devices;
}

string StaticDeviceMgr::DebugString() const {
  string out;
  for (const auto& dev : devices_) {
    strings::StrAppend(&out, dev->name(), "\n");
  }
  return out;
}

string StaticDeviceMgr::DeviceMappingString() const {
  string out;
  for (const auto& dev : devices_) {
    if (!dev->attributes().physical_device_desc().empty()) {
      strings::StrAppend(&out, dev->name(), " -> ",
                         dev->attributes().physical_device_desc(), "\n");
    }
  }
  return out;
}

Status StaticDeviceMgr::LookupDevice(StringPiece name, Device** device) const {
  auto iter = device_map_.find(name);
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

bool StaticDeviceMgr::ContainsDevice(int64_t device_incarnation) const {
  return device_incarnation_set_.contains(device_incarnation);
}

void StaticDeviceMgr::ClearContainers(
    gtl::ArraySlice<string> containers) const {
  Status s;
  for (const auto& dev : devices_) {
    if (containers.empty()) {
      s.Update(dev->resource_manager()->Cleanup(
          dev->resource_manager()->default_container()));
    } else {
      for (const string& c : containers) {
        s.Update(dev->resource_manager()->Cleanup(c));
      }
    }
    if (!s.ok()) {
      LOG(WARNING) << s;
    }
  }
}

int StaticDeviceMgr::NumDeviceType(const string& type) const {
  auto iter = device_type_counts_.find(type);
  if (iter != device_type_counts_.end()) return iter->second;
  return 0;
}

int StaticDeviceMgr::NumDevices() const { return devices_.size(); }

Device* StaticDeviceMgr::HostCPU() const { return cpu_device_; }

void StaticDeviceMgr::InitStreamDevice() {
  // get how many stream device for a gpu and cpu
  std::unordered_map<int, int32> gpu_id2num;
  std::unordered_map<int, int32> cpu_id2num;
  for (auto& d : devices_) {
    if (d->parsed_name().type.find("STREAM_GPU") != string::npos) {
      int idx = std::stoi(d->parsed_name().type.substr(11));
      if (gpu_id2num.find(idx) == gpu_id2num.end()) {
        gpu_id2num[idx] = 1;
      } else {
        ++gpu_id2num[idx];
      }
    } else if (d->parsed_name().type.find("STREAM_CPU") != string::npos) {
      int idx = std::stoi(d->parsed_name().type.substr(11));
      if (cpu_id2num.find(idx) == cpu_id2num.end()) {
        cpu_id2num[idx] = 1;
      } else {
        ++cpu_id2num[idx];
      }
    }
  }

  mutex_lock l(mgrs_mu_);
  // Deal with GPU
  Device* gpu;
  // create stream group mgrs
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
  // create stream_device_map_
  for (auto& d : devices_) {
    if (d->parsed_name().type.find("STREAM_GPU") != string::npos) {
      int idx = std::stoi(d->parsed_name().type.substr(11));
      TF_CHECK_OK(LookupDevice(strings::StrCat("/device:GPU:", idx), &gpu));
      stream_device_map_[gpu][d->parsed_name().id] = d.get();
      d->SetRealDevice(gpu);
    }
  }

  // Deal with CPU
  Device* cpu;
  // don't create stream group mgrs for CPU
  for (auto& item : cpu_id2num) {
    TF_CHECK_OK(
        LookupDevice(strings::StrCat("/device:CPU:", item.first), &cpu));
    stream_device_map_[cpu] = std::vector<Device*>(item.second);
  }
  // create stream_device_map_
  for (auto& d : devices_) {
    if (d->parsed_name().type.find("STREAM_CPU") != string::npos) {
      int idx = std::stoi(d->parsed_name().type.substr(11));
      TF_CHECK_OK(LookupDevice(strings::StrCat("/device:CPU:", idx), &cpu));
      stream_device_map_[cpu][d->parsed_name().id] = d.get();
      d->SetRealDevice(cpu);
    }
  }
}

int32 StaticDeviceMgr::GetMaxStreamNum() const {
  tf_shared_lock l(mgrs_mu_);
  return max_stream_num_;
}

int32 StaticDeviceMgr::GetStreamNum(const Device* device) const {
  tf_shared_lock l(mgrs_mu_);
  if (stream_device_map_.find(device) == stream_device_map_.end()) {
    return 0;
  }
  return stream_device_map_.at(device).size();
}

Device* StaticDeviceMgr::LookupStream(const Device* device,
                                      const int32 stream_id) const {
  tf_shared_lock l(mgrs_mu_);
  if (stream_id < 0 ||
      stream_device_map_.find(device) == stream_device_map_.end() ||
      stream_device_map_.at(device).size() <= stream_id) {
    return const_cast<Device*>(device);
  }
  return stream_device_map_.at(device).at(stream_id);
}

int32 StaticDeviceMgr::RequireStreamGroup(const Device* device) const {
  if (device->parsed_name().type != "GPU" &&
      device->parsed_name().type != "gpu") {
    return -1;
  }
  tf_shared_lock l(mgrs_mu_);
  return stream_group_mgrs_.find(device) == stream_group_mgrs_.end()
             ? -1
             : stream_group_mgrs_[device]->RequireStreamGroup();
}

void StaticDeviceMgr::ReleaseStreamGroup(const Device* device,
                                         const int32 stream_id) const {
  if (device->parsed_name().type == "GPU" ||
      device->parsed_name().type == "gpu") {
    tf_shared_lock l(mgrs_mu_);
    if (stream_group_mgrs_.find(device) != stream_group_mgrs_.end()) {
      DCHECK_NE(stream_id, -1);
      stream_group_mgrs_[device]->ReleaseStreamGroup(stream_id);
    }
  }
}

StaticDeviceMgr::StreamGroupMgr::StreamGroupMgr(const int32 total_num)
    : total_num_(total_num) {
  stream_group_heap_.resize(total_num);
  for (int32 i = 0; i < total_num; ++i) {
    stream_group_heap_[i] = absl::make_unique<StreamGroupNode>(i);
    id2heap_map_.insert(std::make_pair(i, i));
  }
}

void StaticDeviceMgr::StreamGroupMgr::swap(const int32 idx1, const int32 idx2) {
  id2heap_map_[stream_group_heap_[idx1]->id_] = idx2;
  id2heap_map_[stream_group_heap_[idx2]->id_] = idx1;
  std::swap(stream_group_heap_[idx1], stream_group_heap_[idx2]);
}

void StaticDeviceMgr::StreamGroupMgr::reset_accumulators() {
  VLOG(2) << "One of the Stream Group Node reaches access limit"
          << ", reset...";
  for (auto& node : stream_group_heap_) {
    node->accumulator_ = 0;
  }
}

int32 StaticDeviceMgr::StreamGroupMgr::RequireStreamGroup() {
  mutex_lock l(mu_);
  int32 ret(stream_group_heap_[0]->id_);
  ++stream_group_heap_[0]->workload_;
  if (++stream_group_heap_[0]->accumulator_ == 0xffffffffffffffff) {
    reset_accumulators();
  }
  int32 ptr(0);
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
      } else
        break;
    } else if (stream_group_heap_[2 * ptr + 1]->workload_ >
               stream_group_heap_[2 * ptr + 2]->workload_) {
      if (stream_group_heap_[ptr]->workload_ >
          stream_group_heap_[2 * ptr + 2]->workload_) {
        swap(ptr, 2 * ptr + 2);
        ptr = 2 * ptr + 2;
      } else
        break;
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
      } else
        break;
    }
  }
  return ret;
}

void StaticDeviceMgr::StreamGroupMgr::ReleaseStreamGroup(
    const int32 stream_id) {
  mutex_lock l(mu_);
  int32 ptr(id2heap_map_[stream_id]);
  --stream_group_heap_[ptr]->workload_;
  while (ptr != 0) {
    int32 parent = (ptr + 1) / 2 - 1;
    if (stream_group_heap_[ptr]->workload_ <
        stream_group_heap_[parent]->workload_) {
      swap(ptr, parent);
      ptr = parent;
    } else
      break;
  }
}

}  // namespace tensorflow
