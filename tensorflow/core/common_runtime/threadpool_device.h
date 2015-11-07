#ifndef TENSORFLOW_COMMON_RUNTIME_THREADPOOL_DEVICE_H_
#define TENSORFLOW_COMMON_RUNTIME_THREADPOOL_DEVICE_H_

#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/local_device.h"

namespace tensorflow {

// CPU device implementation.
class ThreadPoolDevice : public LocalDevice {
 public:
  ThreadPoolDevice(const SessionOptions& options, const string& name,
                   Bytes memory_limit, BusAdjacency bus_adjacency,
                   Allocator* allocator);
  ~ThreadPoolDevice() override;

  void Compute(OpKernel* op_kernel, OpKernelContext* context) override;
  Allocator* GetAllocator(AllocatorAttributes attr) override;
  Status MakeTensorFromProto(const TensorProto& tensor_proto,
                             const AllocatorAttributes alloc_attrs,
                             Tensor* tensor) override;

  Status Sync() override { return Status::OK(); }

 private:
  Allocator* allocator_;  // Not owned
};

}  // namespace tensorflow

#endif  // TENSORFLOW_COMMON_RUNTIME_THREADPOOL_DEVICE_H_
