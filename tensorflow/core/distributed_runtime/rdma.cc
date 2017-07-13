#include "tensorflow/core/distributed_runtime/rdma.h"

#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {

Status RdmaClient::ReadTensorViaDMA(Tensor*,
                                    Device*,
                                    DeviceContext*,
                                    bool,
                                    const Any&) {
  return errors::Unavailable("RDMA not available");
}

Status RdmaServer::Init() {
  return errors::Unavailable("RDMA not available");
}

void RdmaServer::Run() {}

void RdmaServer::Stop() {}

Status RdmaServer::RegisterTensorDMA(const Tensor&,
                                     Device*,
                                     DeviceContext*,
                                     bool,
                                     Any*) {
  return errors::Unavailable("RDMA not available");
}

}  // namespace tensorflow
