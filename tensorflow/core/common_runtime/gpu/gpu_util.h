#ifndef TENSORFLOW_COMMON_RUNTIME_GPU_GPU_UTIL_H_
#define TENSORFLOW_COMMON_RUNTIME_GPU_GPU_UTIL_H_

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/public/tensor.h"
#include "tensorflow/core/public/status.h"
#include "tensorflow/core/common_runtime/gpu/dma_helper.h"
#include "tensorflow/stream_executor/device_memory.h"

#include "tensorflow/stream_executor/stream.h"

namespace tensorflow {

class RecvTensorResponse;
class TensorProto;

namespace gpu = ::perftools::gputools;

class GPUUtil {
 public:
  // "tensor" is GPU-local.  "dev" is the hosting GPU.
  // "device_context" should be the context of the GPU "_Send" op
  // which provides the Tensor.
  // Sets all necessasry fields of "proto" by transferring value
  // bytes from GPU to CPU RAM. "is_dead" indicates that the
  // tensor is dead with an uninit value.
  static void SetProtoFromGPU(const Tensor& tensor, Device* dev,
                              const DeviceContext* device_context,
                              TensorProto* proto, bool is_dead,
                              StatusCallback done);

  // Copies "input" to "output" between devices accessible to the
  // local process via some DMA-like method.  "edge_name" is the name
  // of the tensor being copied, for debugging purposes. Depending on
  // the type of devices and memory in use, the copy may be performed
  // synchronously or asynchronously.  'done' will be invoked only
  // after the copy is actually complete.
  static void CopyViaDMA(const string& edge_name,
                         DeviceContext* send_dev_context,
                         DeviceContext* recv_dev_context, Device* src,
                         Device* dst, const AllocatorAttributes src_alloc_attr,
                         const AllocatorAttributes dst_alloc_attr,
                         const Tensor* input, Tensor* output,
                         StatusCallback done);

  // Copies the data in 'gpu_tensor' into 'cpu_tensor'.
  // 'gpu_tensor''s backing memory must be on 'gpu_device' and
  // 'cpu_tensor' must be allocated to be of the same size as
  // 'gpu_tensor'. Synchronous: may block.
  static void CopyGPUTensorToCPU(Device* gpu_device,
                                 const DeviceContext* device_context,
                                 const Tensor* gpu_tensor, Tensor* cpu_tensor,
                                 StatusCallback done);

  // Blocks until all operations queued on the stream associated with
  // "gpu_device" at the time of the call have completed.  Returns any
  // error pending on the stream at completion.
  static Status Sync(Device* gpu_device);

  // Blocks until all operations queued on all streams associated with the
  // corresponding GPU device at the time of call have completed.
  // Returns any error pending on the stream at completion.
  static Status SyncAll(Device* gpu_device);

  // For debugging purpose, given a "device" and a "tensor" allocated
  // on the device, return a string printing each byte in the tensor
  // (up to a limit).  "device" can be either a CPU or a GPU device.
  static string MemoryDebugString(const Device* device, Tensor* tensor);

  static perftools::gputools::DeviceMemory<float> AsGPUFloat(const Tensor& t);

  // Computes a checksum over the contents of "tensor", which is allocated
  // on "gpu_device".
  static uint64 Checksum(Device* gpu_device,
                         const DeviceContext* device_context,
                         const Tensor& tensor);

  // Computes a checksum over the contents of "tensor", which is allocated
  // in local CPU RAM.
  static uint64 Checksum(const Tensor& tensor);

  static void CopyCPUTensorToGPU(const Tensor* cpu_tensor,
                                 const DeviceContext* device_context,
                                 Device* gpu_device, Tensor* gpu_tensor,
                                 StatusCallback done);
};

}  // namespace tensorflow
#endif  // TENSORFLOW_COMMON_RUNTIME_GPU_GPU_UTIL_H_
