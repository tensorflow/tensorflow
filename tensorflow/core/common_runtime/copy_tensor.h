#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_COPY_TENSOR_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_COPY_TENSOR_H_

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

class CopyTensor {
 public:
  typedef void (*CopyFunction)(DeviceContext* send_dev_context,
                               DeviceContext* recv_dev_context, Device* src,
                               Device* dst,
                               const AllocatorAttributes src_alloc_attr,
                               const AllocatorAttributes dst_alloc_attr,
                               const Tensor* input, Tensor* output,
                               StatusCallback done);

  // Copies "input" to "output" between devices accessible to the
  // local process via some DMA-like method.  "edge_name" is the name
  // of the tensor being copied, for debugging purposes. Depending on
  // the type of devices and memory in use, the copy may be performed
  // synchronously or asynchronously.  'done' will be invoked only
  // after the copy is actually complete.
  static void ViaDMA(const string& edge_name, DeviceContext* send_dev_context,
                     DeviceContext* recv_dev_context, Device* src, Device* dst,
                     const AllocatorAttributes src_alloc_attr,
                     const AllocatorAttributes dst_alloc_attr,
                     const Tensor* input, Tensor* output, StatusCallback done);

  // Register a function for copying between two specific DeviceTypes.
  static Status Register(DeviceType sender_device_type,
                         DeviceType receiver_device_type,
                         CopyFunction copy_function);

  // Object used to call Register() at static-initialization time.
  class Registration {
   public:
    Registration(DeviceType sender_device_type, DeviceType receiver_device_type,
                 CopyFunction copy_function) {
      TF_QCHECK_OK(
          Register(sender_device_type, receiver_device_type, copy_function));
    }
  };
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_COPY_TENSOR_H_
