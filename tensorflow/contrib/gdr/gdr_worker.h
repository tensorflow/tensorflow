#ifndef GDR_WORKER_H_
#define GDR_WORKER_H_

#include "tensorflow/contrib/gdr/gdr_memory_manager.h"

#include "tensorflow/core/distributed_runtime/rpc/grpc_worker_service.h"

namespace tensorflow {

class GdrWorker : public GrpcWorker {
 public:
  GdrWorker(WorkerEnv* env, RemoteMemoryManager* remote_memory_manager);

  // Serve the RecvTensorRequest but omit the tensor content and transmit it
  // out-of-band using GPU Direct RDMA whenever possible.
  // If it's not possible, it falls back to gRPC in-band tensor transport by
  // encoding the tensor content into the grpc::ByteBuffer.
  // The RecvTensorResponse will carry the necessary information for RDMA.
  virtual void GrpcRecvTensorAsync(CallOptions* opts,
                                   const RecvTensorRequest* request,
                                   ::grpc::ByteBuffer* response,
                                   StatusCallback done) override;

 private:
  RemoteMemoryManager* remote_memory_manager_;  // Not owned
};

}  // namespace tensorflow

#endif  // GDR_WORKER_H_
