#if !GOOGLE_CUDA
#error This file must only be included when building with Cuda support
#endif

#ifndef TENSORFLOW_COMMON_RUNTIME_GPU_GPU_DEVICE_H_
#define TENSORFLOW_COMMON_RUNTIME_GPU_GPU_DEVICE_H_

#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/gpu_device_context.h"
#include "tensorflow/core/common_runtime/local_device.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/port.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/public/status.h"
#include "tensorflow/core/public/tensor.h"
#include "tensorflow/core/common_runtime/gpu/gpu_event_mgr.h"
#include "tensorflow/stream_executor/stream.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace tensorflow {

class EigenAllocator;

class BaseGPUDevice : public LocalDevice {
 public:
  BaseGPUDevice(const SessionOptions& options, const string& name,
                Bytes memory_limit, BusAdjacency bus_adjacency, int gpu_id,
                const string& physical_device_desc, Allocator* gpu_allocator,
                Allocator* cpu_allocator);

  ~BaseGPUDevice() override;

  // GPU devices require the Op Compute method to save a reference to
  // any temporary tensors that are allocated until the Op execution
  // completes.
  bool SaveTemporaryTensors() const override { return true; }

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
  const PerOpGpuDevice* MakeGpuDevice(DeviceContext* dc,
                                      Allocator* allocator) override;

 protected:
  Allocator* gpu_allocator_;  // not owned
  Allocator* cpu_allocator_;  // not owned

 private:
  std::vector<gpu::Stream*> streams_;
  std::vector<GPUDeviceContext*> device_contexts_;
  GpuDeviceInfo* gpu_device_info_ = nullptr;
  mutex trace_mu_;
  int gpu_id_ = -1;
  std::unique_ptr<EventMgr> em_;

  const PerOpGpuDevice* NewDevice(int stream_id, Allocator* allocator);
};

class BaseGPUDeviceFactory : public DeviceFactory {
 public:
  void CreateDevices(const SessionOptions& options, const string& name_prefix,
                     std::vector<Device*>* devices) override;

 private:
  LocalDevice* CreateGPUDevice(const SessionOptions& options,
                               const string& name, int gpu_id);

  virtual LocalDevice* CreateGPUDevice(const SessionOptions& options,
                                       const string& name, Bytes memory_limit,
                                       BusAdjacency bus_adjacency, int gpu_id,
                                       const string& physical_device_desc,
                                       Allocator* gpu_allocator,
                                       Allocator* cpu_allocator) = 0;

  void GetValidDeviceIds(std::vector<int>* ids);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_COMMON_RUNTIME_GPU_GPU_DEVICE_H_
