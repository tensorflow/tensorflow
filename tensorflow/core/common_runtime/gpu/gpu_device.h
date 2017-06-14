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

#if !GOOGLE_CUDA
#error This file must only be included when building with Cuda support
#endif

#ifndef TENSORFLOW_COMMON_RUNTIME_GPU_GPU_DEVICE_H_
#define TENSORFLOW_COMMON_RUNTIME_GPU_GPU_DEVICE_H_

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/gpu/gpu_event_mgr.h"
#include "tensorflow/core/common_runtime/gpu_device_context.h"
#include "tensorflow/core/common_runtime/local_device.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/stream_executor.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {

class BaseGPUDevice : public LocalDevice {
 public:
  BaseGPUDevice(const SessionOptions& options, const string& name,
                Bytes memory_limit, const DeviceLocality& locality, int gpu_id,
                const string& physical_device_desc, Allocator* gpu_allocator,
                Allocator* cpu_allocator, bool sync_every_op,
                int32 max_streams);

  ~BaseGPUDevice() override;

  // Initialize the device and return the status of initialization.
  Status Init(const SessionOptions& options);

  // GPU devices require the Op Compute method to save a reference to
  // any temporary tensors that are allocated until the Op execution
  // completes.
  bool RequiresRecordingAccessedTensors() const override;

  void ConsumeListOfAccessedTensors(
      DeviceContext* device_context,
      const TensorReferenceVector& tensor_refs) override;

  Status FillContextMap(const Graph* graph,
                        DeviceContextMap* device_context_map);

  void Compute(OpKernel* op_kernel, OpKernelContext* context) override;

  Status Sync() override;

  void ComputeAsync(AsyncOpKernel* op_kernel, OpKernelContext* context,
                    AsyncOpKernel::DoneCallback done) override;

  Status MakeTensorFromProto(const TensorProto& tensor_proto,
                             const AllocatorAttributes alloc_attrs,
                             Tensor* tensor) override;

  // The caller owns the returned device.
  PerOpGpuDevice* MakeGpuDevice() override;

  void ReinitializeGpuDevice(OpKernelContext* context, PerOpGpuDevice* device,
                             DeviceContext* dc, Allocator* allocator) override;

  // Returns the id of this device within the native driver system; e.g., for
  // CUDA this is the ordinal of the GPU within the system.
  int gpu_id() const { return gpu_id_; }

  // The executor that provides control for the device; e.g., for CUDA this
  // corresponds to the cuda context.
  gpu::StreamExecutor* executor() const { return executor_; }

 protected:
  Allocator* gpu_allocator_;  // not owned
  Allocator* cpu_allocator_;  // not owned

  gpu::StreamExecutor* executor_;  // not owned

 private:
  struct StreamGroup {
    gpu::Stream* compute = nullptr;
    gpu::Stream* host_to_device = nullptr;
    gpu::Stream* device_to_host = nullptr;
    gpu::Stream* device_to_device = nullptr;
  };
  class StreamGroupFactory;

  gtl::InlinedVector<StreamGroup*, 4> streams_;
  gtl::InlinedVector<char*, 4> scratch_;
  std::vector<GPUDeviceContext*> device_contexts_;
  GpuDeviceInfo* gpu_device_info_ = nullptr;
  mutex trace_mu_;
  int gpu_id_ = -1;
  const bool sync_every_op_ = false;
  const int32 max_streams_;
  std::unique_ptr<EventMgr> em_;

  void ReinitializeDevice(OpKernelContext* context, PerOpGpuDevice* device,
                          int stream_id, Allocator* allocator);

  void ComputeHelper(OpKernel* op_kernel, OpKernelContext* context);
};

class BaseGPUDeviceFactory : public DeviceFactory {
 public:
  Status CreateDevices(const SessionOptions& options, const string& name_prefix,
                       std::vector<Device*>* devices) override;

 private:
  Status CreateGPUDevice(const SessionOptions& options, const string& name,
                         int gpu_id, BaseGPUDevice** out_device);

  virtual BaseGPUDevice* CreateGPUDevice(const SessionOptions& options,
                                         const string& name, Bytes memory_limit,
                                         const DeviceLocality& locality,
                                         int gpu_id,
                                         const string& physical_device_desc,
                                         Allocator* gpu_allocator,
                                         Allocator* cpu_allocator) = 0;

  // Returns into 'ids' the list of valid GPU ids, in the order that
  // they should map to logical gpu ids "/gpu:0", "/gpu:1", etc, based
  // upon 'visible_device_list', a comma-separated list of 'visible
  // gpu ids'.
  Status GetValidDeviceIds(const string& visible_device_list,
                           std::vector<int>* ids);

  // visible_gpu_initialized_[gpu_id] is true if visible GPU gpu_id
  // has been initialized by the process.
  std::unordered_map<int, bool> visible_gpu_initialized_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_COMMON_RUNTIME_GPU_GPU_DEVICE_H_
