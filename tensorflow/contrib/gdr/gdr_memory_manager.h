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

class RemoteMemoryManager {
 public:
  virtual Status Init() = 0;
  virtual void Run() = 0;
  virtual void Stop() = 0;

  virtual Status TransportOptionsFromTensor(
      ::google::protobuf::Any* mutable_transport_options, const Tensor& tensor,
      Device* device, DeviceContext* device_context, bool on_host) = 0;

  virtual Status TensorFromTransportOptions(
      Tensor* tensor, const ::google::protobuf::Any& transport_options,
      Device* device, DeviceContext* device_context, bool on_host) = 0;
};

RemoteMemoryManager* CreateRemoteMemoryManager(const string& host,
                                               const string& port);

}  // namespace tensorflow

#endif  // GDR_MEMORY_MANAGER_H_
