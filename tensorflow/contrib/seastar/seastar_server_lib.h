#ifndef TENSORFLOW_CONTRIB_SEASTAR_SEASTAR_SERVER_LIB_H_
#define TENSORFLOW_CONTRIB_SEASTAR_SEASTAR_SERVER_LIB_H_

#include "grpc++/grpc++.h"
#include "grpc++/security/credentials.h"
#include "tensorflow/core/common_runtime/process_util.h"
#include "tensorflow/core/distributed_runtime/master_env.h"
#include "tensorflow/core/distributed_runtime/rpc/async_service_interface.h"
#include "tensorflow/core/distributed_runtime/server_lib.h"
#include "tensorflow/core/distributed_runtime/session_mgr.h"
#include "tensorflow/core/distributed_runtime/worker_env.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/platform/env.h"

namespace tensorflow {
class Master;
class SeastarEngine;
class SeastarWorker;
class SeastarWorkerService;
class SeastarChannelSpec;
class SeastarPortMgr;

class SeastarServer : public ServerInterface {
protected:
  SeastarServer(const ServerDef& server_def, Env* env);

public:
  static Status Create(const ServerDef& server_def, Env* env,
                       std::unique_ptr<ServerInterface>* out_server);

  virtual ~SeastarServer();

  Status Start() override;
  Status Stop() override;
  Status Join() override;
  const string target() const;

  Status Init();

protected:
  Status ParseChannelSpec(const WorkerCacheFactoryOptions& options,
                          SeastarChannelSpec* channel_spec);

  size_t ParseServers(const WorkerCacheFactoryOptions& options);

  Status SeastarWorkerCacheFactory(const WorkerCacheFactoryOptions& options,
                                   WorkerCacheInterface** worker_cache);

  virtual std::shared_ptr<::grpc::ServerCredentials> GetServerCredentials(
          const ServerDef& server_def) const;
  virtual std::unique_ptr<Master> CreateMaster(MasterEnv* master_env);

  int bound_port() const { return bound_port_; }
  WorkerEnv* worker_env() { return &worker_env_; }
  const ServerDef& server_def() const { return server_def_; }

private:
  const ServerDef server_def_;
  Env* env_;

  int bound_port_ = 0;
  int seastar_bound_port_ = 0;

  mutex mu_;
  enum State { NEW, STARTED, STOPPED };
  State state_ GUARDED_BY(mu_);

  // master part, still using grpc
  MasterEnv master_env_;
  std::unique_ptr<Master> master_impl_;
  AsyncServiceInterface* master_service_ = nullptr;
  std::unique_ptr<Thread> master_thread_ GUARDED_BY(mu_);
  std::unique_ptr<::grpc::Server> server_ GUARDED_BY(mu_);

  WorkerEnv worker_env_;
  std::unique_ptr<SeastarWorker> worker_impl_;
  SeastarWorkerService* worker_service_ = nullptr;
  SeastarEngine* seastar_engine_ = nullptr;
  SeastarPortMgr* seastar_port_mgr_ = nullptr;
};
}

#endif
