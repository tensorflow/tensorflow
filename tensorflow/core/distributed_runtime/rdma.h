#ifndef THIRD_PARTY_TENSORFLOW_DISTRIBUTED_RUNTIME_RDMA_H_
#define THIRD_PARTY_TENSORFLOW_DISTRIBUTED_RUNTIME_RDMA_H_

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

class RdmaClient {
 public:
  using Any = ::google::protobuf::Any;
  virtual Status ReadTensorViaDMA(Tensor* tensor,
                                  Device* dst_device,
                                  DeviceContext* dst_device_context,
                                  bool on_host,
                                  const Any& transport_options);
};

class RdmaServer {
 public:
  virtual Status Init();

  virtual void Run();

  virtual void Stop();

  using Any = ::google::protobuf::Any;
  virtual Status RegisterTensorDMA(const Tensor& tensor,
                                   Device* src_device,
                                   DeviceContext* src_device_context,
                                   bool on_host,
                                   Any* mutable_transport_options);
};

}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_DISTRIBUTED_RUNTIME_RDMA_H_
