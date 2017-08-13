#ifndef GDR_SERVER_LIB_H_
#define GDR_SERVER_LIB_H_

#include "tensorflow/contrib/gdr/gdr_memory_manager.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_server_lib.h"

namespace tensorflow {

class GdrServer : public GrpcServer {
 protected:
  GdrServer(const ServerDef& server_def, Env* env);

 public:
  static Status Create(const ServerDef& server_def, Env* env,
                       std::unique_ptr<ServerInterface>* out_server);

  virtual ~GdrServer() override;

  virtual Status Start() override;

  virtual Status Stop() override;

  virtual Status Join() override;

 protected:
  Status Init();

 private:
  mutex mu_;

  std::unique_ptr<RemoteMemoryManager> remote_memory_manager_;
  std::unique_ptr<Thread> gdr_thread_ GUARDED_BY(mu_);
};

}  // namespace tensorflow

#endif  // GDR_SERVER_LIB_H_
