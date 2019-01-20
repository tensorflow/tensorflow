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

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_GPU_GPU_DEVICE_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_GPU_GPU_DEVICE_H_

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/gpu/gpu_event_mgr.h"
#include "tensorflow/core/common_runtime/gpu/gpu_id.h"
#include "tensorflow/core/common_runtime/gpu/gpu_id_manager.h"
#include "tensorflow/core/common_runtime/gpu/gpu_id_utils.h"
#include "tensorflow/core/common_runtime/gpu_device_context.h"
#include "tensorflow/core/common_runtime/local_device.h"
#include "tensorflow/core/common_runtime/scoped_allocator_mgr.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/stream_executor.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {

class BaseGPUDevice : public LocalDevice {
 public:
  BaseGPUDevice(const SessionOptions& options, const string& name,
                Bytes memory_limit, const DeviceLocality& locality,
                TfGpuId tf_gpu_id, const string& physical_device_desc,
                Allocator* gpu_allocator, Allocator* cpu_allocator,
                bool sync_every_op, int32 max_streams);

  ~BaseGPUDevice() override;

  // Initialize the device and return the status of initialization.
  Status Init(const SessionOptions& options);

  // GPU devices require the Op Compute method to save a reference to
  // any temporary tensors that are allocated until the Op execution
  // completes.
  bool RequiresRecordingAccessedTensors() const override;

  // GPU kernel execution requires us to use `tracing::ScopedAnnotation()`
  // rather than `tracing::ScopedActivity()`, in order to relate asynchronously
  // launched GPU kernels to the OpKernel.
  bool TraceUsingAnnotations() const { return true; }

  void ConsumeListOfAccessedTensors(
      DeviceContext* device_context,
      const TensorReferenceVector& tensor_refs) override;

  Status FillContextMap(const Graph* graph,
                        DeviceContextMap* device_context_map) override;

  void Compute(OpKernel* op_kernel, OpKernelContext* context) override;

  Status Sync() override;

  void ComputeAsync(AsyncOpKernel* op_kernel, OpKernelContext* context,
                    AsyncOpKernel::DoneCallback done) override;

  Status MakeTensorFromProto(const TensorProto& tensor_proto,
                             const AllocatorAttributes alloc_attrs,
                             Tensor* tensor) override;

  // The caller owns the returned device.
  PerOpGpuDevice* MakeGpuDevice() override;

  Status ReinitializeGpuDevice(OpKernelContext* context, PerOpGpuDevice* device,
                               DeviceContext* dc,
                               Allocator* allocator) override;

  // Returns the platform GPU id of this device within the native driver system;
  // e.g., for CUDA this is the ordinal of the GPU within the system.
  int gpu_id() const {
    PlatformGpuId platform_gpu_id;
    TF_CHECK_OK(GpuIdManager::TfToPlatformGpuId(tf_gpu_id_, &platform_gpu_id));
    return platform_gpu_id.value();
  }

  // The executor that provides control for the device; e.g., for CUDA this
  // corresponds to the cuda context.
  se::StreamExecutor* executor() const { return executor_; }

  Allocator* GetScopedAllocator(AllocatorAttributes attr,
                                int64 step_id) override;

  ScopedAllocatorMgr* GetScopedAllocatorMgr() const override {
    return scoped_allocator_mgr_.get();
  }

 protected:
  Allocator* gpu_allocator_;  // not owned
  Allocator* cpu_allocator_;  // not owned

  se::StreamExecutor* executor_;  // not owned
  std::unique_ptr<ScopedAllocatorMgr> scoped_allocator_mgr_;

 private:
  struct StreamGroup {
    se::Stream* compute = nullptr;
    se::Stream* host_to_device = nullptr;
    se::Stream* device_to_host = nullptr;
    gtl::InlinedVector<se::Stream*, 4> device_to_device;
  };
  class StreamGroupFactory;

  gtl::InlinedVector<StreamGroup*, 4> streams_;
  mutex scratch_init_mutex_;
  gtl::InlinedVector<char*, 4> scratch_;
  std::vector<GPUDeviceContext*> device_contexts_;
  GpuDeviceInfo* gpu_device_info_ = nullptr;
  mutex trace_mu_;
  TfGpuId tf_gpu_id_;
  const bool sync_every_op_ = false;
  const int32 max_streams_;
  std::unique_ptr<EventMgr> em_;
  std::unique_ptr<thread::ThreadPool> thread_pool_;

  // Initialize scractch buffers used by Eigen.
  Status InitScratchBuffers();

  void ReinitializeDevice(OpKernelContext* context, PerOpGpuDevice* device,
                          int stream_id, Allocator* allocator);

  void ComputeHelper(OpKernel* op_kernel, OpKernelContext* context);

  string ComputeOpKernelDebugString(const OpKernel& op_kernel,
                                    const int& stream_id);

  // This method returns an initialization status, in addition to
  // calling the "done" StatusCallback, if there is a failure to
  // allocate memory or if the tensor "from" is not DMA-copyable.
  // If there is no error prior to enqueueing the copy, an OK status
  // is returned.
  Status MaybeCopyTensorToGPU(const AllocatorAttributes& alloc_attrs,
                              const Tensor& from, Tensor* to,
                              StatusCallback done);
};

class BaseGPUDeviceFactory : public DeviceFactory {
 public:
  Status CreateDevices(const SessionOptions& options, const string& name_prefix,
                       std::vector<std::unique_ptr<Device>>* devices) override;

  struct InterconnectMap {
    // Name of interconnect technology, if known.
    string name;
    // If possible, strength should approximate Gb/sec bandwidth rate.
    // Where architecture-specific subclassing is not done that won't
    // always be possible.  The minimum expectation is that
    // faster links should have a higher value than slower links.
    int32 strength;
    static const int kSameDeviceStrength;
    static const int kStreamExecutorStrength;
    std::set<std::pair<PlatformGpuId, PlatformGpuId>> directed_links;
  };

 protected:
  // Populates *maps with interconnect maps for all local direct access
  // pathways between GPUs.
  virtual Status GetInterconnectMaps(
      const std::vector<PlatformGpuId>& visible_gpu_order,
      se::Platform* gpu_manager, std::vector<InterconnectMap>* maps);

  struct TfGpuIdHash {
    std::size_t operator()(const TfGpuId& id) const noexcept {
      return std::hash<int>{}(id.value());
    }
  };
  typedef std::unordered_map<TfGpuId, DeviceLocality, TfGpuIdHash> LocalityMap;
  // Populates *localities with the DeviceLocality descriptor for
  // every TfGpuId.
  virtual Status GetDeviceLocalities(
      int num_tf_gpus, const std::vector<InterconnectMap>& interconnects,
      LocalityMap* localities);

 private:
  // Creates a BaseGPUDevice associated with 'tf_gpu_id', allocates (strictly)
  // 'memory_limit' bytes of GPU memory to it, and adds it to the 'devices'
  // vector.
  Status CreateGPUDevice(const SessionOptions& options,
                         const string& name_prefix, TfGpuId tf_gpu_id,
                         int64 memory_limit, const DeviceLocality& dev_locality,
                         std::vector<std::unique_ptr<Device>>* devices);

  virtual std::unique_ptr<BaseGPUDevice> CreateGPUDevice(
      const SessionOptions& options, const string& name, Bytes memory_limit,
      const DeviceLocality& dev_locality, TfGpuId tf_gpu_id,
      const string& physical_device_desc, Allocator* gpu_allocator,
      Allocator* cpu_allocator) = 0;

  // Returns into 'ids' the list of valid platform GPU ids, in the order that
  // they should map to TF GPU ids "/device:GPU:0", "/device:GPU:1", etc,
  // based upon 'visible_gpu_order' which was generated by parsing
  // GPUOptions::visible_device_list which is a comma-separated list of CUDA GPU
  // ids.
  Status GetValidDeviceIds(const std::vector<PlatformGpuId>& visible_gpu_order,
                           std::vector<PlatformGpuId>* ids);

  // visible_gpu_initialized_[platform_gpu_id] is true if visible GPU
  // platform_gpu_id has been initialized by the process.
  std::unordered_map<int, bool> visible_gpu_initialized_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_GPU_GPU_DEVICE_H_
