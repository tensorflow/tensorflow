#ifndef GDR_MEMORY_MANAGER_H_
#define GDR_MEMORY_MANAGER_H_

#include "tensorflow/core/lib/core/status.h"

namespace google {
namespace protobuf {
class Any;
}
}

namespace tensorflow {

class Device;
class DeviceContext;
class Tensor;

// Abstract interface that handles out-of-band tensor transport.
//
// The transport options are encoded into a protocol buffer and transmitted via
// some other communication channels like RPC.
// See RecvTensorRequest in tensorflow/core/protobuf/worker.proto
class RemoteMemoryManager {
 public:
  virtual Status Init() = 0;
  virtual void Run() = 0;
  virtual void Stop() = 0;

  // Encodes the tensor information to an arbitrary protocol buffer
  // The protocol buffer needs to be transmitted via some other channel
  virtual Status TransportOptionsFromTensor(
      ::google::protobuf::Any* mutable_transport_options, const Tensor& tensor,
      Device* device, DeviceContext* device_context, bool on_host) = 0;

  // Retrieve the tensor from the encoded protocol buffer
  // Note that the tensor has to be allocated, but not initialized
  virtual Status TensorFromTransportOptions(
      Tensor* tensor, const ::google::protobuf::Any& transport_options,
      Device* device, DeviceContext* device_context, bool on_host) = 0;
};

RemoteMemoryManager* CreateRemoteMemoryManager(const string& host,
                                               const string& port);

}  // namespace tensorflow

#endif  // GDR_MEMORY_MANAGER_H_
