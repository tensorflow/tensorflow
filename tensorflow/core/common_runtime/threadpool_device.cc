#include "tensorflow/core/common_runtime/threadpool_device.h"

#include "tensorflow/core/common_runtime/local_device.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/graph/types.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/platform/port.h"
#include "tensorflow/core/platform/tracing.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {

ThreadPoolDevice::ThreadPoolDevice(const SessionOptions& options,
                                   const string& name, Bytes memory_limit,
                                   BusAdjacency bus_adjacency,
                                   Allocator* allocator)
    : LocalDevice(options, Device::BuildDeviceAttributes(
                               name, DEVICE_CPU, memory_limit, bus_adjacency),
                  allocator),
      allocator_(allocator) {}

ThreadPoolDevice::~ThreadPoolDevice() {}

void ThreadPoolDevice::Compute(OpKernel* op_kernel, OpKernelContext* context) {
  if (port::Tracing::IsActive()) {
    // TODO(pbar) We really need a useful identifier of the graph node.
    const uint64 id = Hash64(op_kernel->name());
    port::Tracing::ScopedActivity region(port::Tracing::EventCategory::kCompute,
                                         id);
    op_kernel->Compute(context);
  } else {
    op_kernel->Compute(context);
  }
}

Allocator* ThreadPoolDevice::GetAllocator(AllocatorAttributes attr) {
  return allocator_;
}

Status ThreadPoolDevice::MakeTensorFromProto(
    const TensorProto& tensor_proto, const AllocatorAttributes alloc_attrs,
    Tensor* tensor) {
  Tensor parsed(tensor_proto.dtype());
  if (!parsed.FromProto(cpu_allocator(), tensor_proto)) {
    return errors::InvalidArgument("Cannot parse tensor from proto: ",
                                   tensor_proto.DebugString());
  }
  *tensor = parsed;
  return Status::OK();
}

}  // namespace tensorflow
