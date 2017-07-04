#ifndef THIRD_PARTY_TENSORFLOW_DISTRIBUTED_RUNTIME_RPC_RDMA_H_
#define THIRD_PARTY_TENSORFLOW_DISTRIBUTED_RUNTIME_RPC_RDMA_H_

#include "tensorflow/core/distributed_runtime/rdma.h"

namespace tensorflow {

RdmaServer* NewRdmaServer(const string& host, const string& port);

RdmaClient* NewRdmaClient();

}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_DISTRIBUTED_RUNTIME_RPC_RDMA_H_
