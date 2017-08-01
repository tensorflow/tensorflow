#ifndef GDR_WORKER_H_
#define GDR_WORKER_H_

#include "tensorflow/contrib/gdr/gdr_memory_manager.h"

#include "tensorflow/core/distributed_runtime/rpc/grpc_worker_service.h"

namespace tensorflow {

class GdrWorker : public GrpcWorker {
 public:
  GdrWorker(WorkerEnv* env, RemoteMemoryManager* remote_memory_manager);

  virtual void RecvTensorAsync(CallOptions* opts,
                               const RecvTensorRequest* request,
                               ::grpc::ByteBuffer* response,
                               StatusCallback done) override;

 private:
  RemoteMemoryManager* remote_memory_manager_;  // Not owned
};

}  // namespace tensorflow

#endif  // GDR_WORKER_H_
