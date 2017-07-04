#ifndef THIRD_PARTY_TENSORFLOW_DISTRIBUTED_RUNTIME_RDMA_H_
#define THIRD_PARTY_TENSORFLOW_DISTRIBUTED_RUNTIME_RDMA_H_

#include "tensorflow/core/lib/core/status.h"

namespace google {
namespace protobuf {
class Any;
}
}

namespace tensorflow {

class TensorBuffer;

class RdmaClient {
 public:
  using Any = ::google::protobuf::Any;
  virtual Status ReadTensorViaDMA(const TensorBuffer* buffer,
                                  const Any& transport_options) = 0;
};

class RdmaServer {
 public:
  virtual Status Init() = 0;

  virtual void Run() = 0;

  virtual void Stop() = 0;

  using Any = ::google::protobuf::Any;
  virtual Status RegisterTensorDMA(const TensorBuffer* buffer,
                                   Any* mutable_transport_options) = 0;
};

}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_DISTRIBUTED_RUNTIME_RDMA_H_
