#ifndef TENSORFLOW_CONTRIB_SEASTAR_SEASTAR_SERVER_LIB_H_
#define TENSORFLOW_CONTRIB_SEASTAR_SEASTAR_SERVER_LIB_H_

#include "tensorflow/core/distributed_runtime/rpc/async_service_interface.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_server_lib.h"
#include "tensorflow/core/platform/env.h"

namespace tensorflow {
class SeastarEngine;
class SeastarWorker;
class SeastarWorkerService;
class SeastarChannelSpec;
class SeastarPortMgr;

class SeastarServer : public GrpcServer {
 protected:
  SeastarServer(const ServerDef& server_def, Env* env);

 public:
  static Status Create(const ServerDef& server_def, Env* env,
                       std::unique_ptr<ServerInterface>* out_server);
  virtual ~SeastarServer();
  Status Init();

 protected:
  Status ParseChannelSpec(const WorkerCacheFactoryOptions& options,
                          SeastarChannelSpec* channel_spec);
  Status WorkerCacheFactory(const WorkerCacheFactoryOptions& options,
                            WorkerCacheInterface** worker_cache) override;

 private:
  int seastar_bound_port_ = 0;
  std::unique_ptr<SeastarWorker> seastar_worker_impl_;
  SeastarWorkerService* seastar_worker_service_ = nullptr;
  SeastarEngine* seastar_engine_ = nullptr;
  SeastarPortMgr* seastar_port_mgr_ = nullptr;
};
}

#endif
