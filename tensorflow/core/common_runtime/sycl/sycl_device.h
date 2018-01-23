/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#if !TENSORFLOW_USE_SYCL
#error This file must only be included when building TensorFlow with SYCL support
#endif

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_SYCL_SYCL_DEVICE_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_SYCL_SYCL_DEVICE_H_

#include "tensorflow/core/common_runtime/local_device.h"
#include "tensorflow/core/common_runtime/sycl/sycl_allocator.h"
#include "tensorflow/core/common_runtime/sycl/sycl_device_context.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {

class GSYCLInterface {
  std::vector<Eigen::QueueInterface*> m_queue_interface_;  // owned
  std::vector<Allocator*> m_cpu_allocator_;                // not owned
  std::vector<SYCLAllocator*> m_sycl_allocator_;           // owned
  std::vector<SYCLDeviceContext*> m_sycl_context_;         // ref counted
  GSYCLInterface() {
    bool found_device = false;
    auto device_list = Eigen::get_sycl_supported_devices();
    // Obtain list of supported devices from Eigen
    for (const auto& device : device_list) {
      if (device.is_gpu()) {
        // returns first found GPU
        AddDevice(device);
        found_device = true;
      }
    }

    if (!found_device) {
      // Currently Intel GPU is not supported
      LOG(WARNING) << "No OpenCL GPU found that is supported by "
                   << "ComputeCpp/triSYCL, trying OpenCL CPU";
    }

    for (const auto& device : device_list) {
      if (device.is_cpu()) {
        // returns first found CPU
        AddDevice(device);
        found_device = true;
      }
    }

    if (!found_device) {
      LOG(WARNING) << "No OpenCL CPU found that is supported by "
                   << "ComputeCpp/triSYCL, checking for host sycl device";
    }

    for (const auto& device : device_list) {
      // triSYCL only supports the host device for now
      if (device.is_host()) {
        LOG(WARNING) << "Found SYCL host device";
        AddDevice(device);
        found_device = true;
      }
    }

    if (!found_device) {
      // Currently Intel GPU is not supported
      LOG(FATAL) << "No SYCL host and no OpenCL GPU nor CPU"
                 << " supported by ComputeCPP/triSYCL was found";
    } else {
      LOG(INFO) << "Found following OpenCL devices:";
      for (int i = 0; i < device_list.size(); i++) {
        LOG(INFO) << GetShortDeviceDescription(i);
      }
    }
  }

  ~GSYCLInterface() {
    m_cpu_allocator_.clear();

    for (auto p : m_sycl_allocator_) {
      p->Synchronize();
      p->ClearSYCLDevice();
      // Cannot delete the Allocator instances, as the Allocator lifetime
      // needs to exceed any Tensor created by it. There is no way of
      // knowing when all Tensors have been deallocated, as they are
      // RefCounted and wait until all instances of a Tensor have been
      // destroyed before calling Allocator.Deallocate. This could happen at
      // program exit, which can set up a race condition between destroying
      // Tensors and Allocators when the program is cleaning up.
    }
    m_sycl_allocator_.clear();

    for (auto p : m_sycl_context_) {
      p->Unref();
    }
    m_sycl_context_.clear();

    for (auto p : m_queue_interface_) {
      p->deallocate_all();
      delete p;
    }
    m_queue_interface_.clear();
  }

  void AddDevice(const cl::sycl::device& d) {
    m_queue_interface_.push_back(new Eigen::QueueInterface(d));
    m_cpu_allocator_.push_back(cpu_allocator());
    m_sycl_allocator_.push_back(new SYCLAllocator(m_queue_interface_.back()));
    m_sycl_context_.push_back(new SYCLDeviceContext());
  }

 public:
  static const GSYCLInterface* instance() {
    // c++11 guarantees that this will be constructed in a thread safe way
    static const GSYCLInterface instance;
    return &instance;
  }

  Eigen::QueueInterface* GetQueueInterface(size_t i = 0) const {
    if (!m_queue_interface_.empty()) {
      return m_queue_interface_[i];
    } else {
      std::cerr << "No cl::sycl::device has been added" << std::endl;
      return nullptr;
    }
  }

  SYCLAllocator* GetSYCLAllocator(size_t i = 0) const {
    if (!m_sycl_allocator_.empty()) {
      return m_sycl_allocator_[i];
    } else {
      std::cerr << "No cl::sycl::device has been added" << std::endl;
      return nullptr;
    }
  }

  Allocator* GetCPUAllocator(size_t i = 0) const {
    if (!m_cpu_allocator_.empty()) {
      return m_cpu_allocator_[i];
    } else {
      std::cerr << "No cl::sycl::device has been added" << std::endl;
      return nullptr;
    }
  }

  SYCLDeviceContext* GetSYCLContext(size_t i = 0) const {
    if (!m_sycl_context_.empty()) {
      return m_sycl_context_[i];
    } else {
      std::cerr << "No cl::sycl::device has been added" << std::endl;
      return nullptr;
    }
  }

  string GetShortDeviceDescription(int device_id = 0) const {
    Eigen::QueueInterface* queue_ptr = GetQueueInterface(device_id);
    if (!queue_ptr) {
      LOG(ERROR)
          << "Device name cannot be given after Eigen QueueInterface destroyed";
      return "";
    }
    auto device = queue_ptr->sycl_queue().get_device();
    auto name = device.get_info<cl::sycl::info::device::name>();
    auto vendor = device.get_info<cl::sycl::info::device::vendor>();
    auto profile = device.get_info<cl::sycl::info::device::profile>();

    std::string type;
    if (device.is_host()) {
      type = "Host";
    } else if (device.is_cpu()) {
      type = "CPU";
    } else if (device.is_gpu()) {
      type = "GPU";
    } else if (device.is_accelerator()) {
      type = "Accelerator";
    } else {
      type = "Unknown";
    }

    return strings::StrCat("id: ", device_id, ", type: ", type, ", name: ",
                           name.c_str(), ", vendor: ", vendor.c_str(),
                           ", profile: ", profile.c_str());
  }
};

class SYCLDevice : public LocalDevice {
 public:
  SYCLDevice(const SessionOptions& options, const string& name,
             Bytes memory_limit, const DeviceLocality& locality,
             const string& physical_device_desc, SYCLAllocator* sycl_allocator,
             Allocator* cpu_allocator, SYCLDeviceContext* ctx)
      : LocalDevice(options, Device::BuildDeviceAttributes(
                                 name, DEVICE_SYCL, memory_limit, locality,
                                 physical_device_desc)),
        cpu_allocator_(cpu_allocator),
        sycl_allocator_(sycl_allocator),
        device_context_(ctx) {
    set_eigen_sycl_device(sycl_allocator->getSyclDevice());
  }

  ~SYCLDevice() override;

  void Compute(OpKernel* op_kernel, OpKernelContext* context) override;
  Allocator* GetAllocator(AllocatorAttributes attr) override;
  Status MakeTensorFromProto(const TensorProto& tensor_proto,
                             const AllocatorAttributes alloc_attrs,
                             Tensor* tensor) override;

  Status FillContextMap(const Graph* graph,
                        DeviceContextMap* device_context_map) override;

  Status Sync() override;

 private:
  Allocator* cpu_allocator_;           // not owned
  SYCLAllocator* sycl_allocator_;      // not owned
  SYCLDeviceContext* device_context_;  // not owned
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_SYCL_SYCL_DEVICE_H_
