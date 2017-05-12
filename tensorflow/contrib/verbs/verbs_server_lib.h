/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef THIRD_PARTY_TENSORFLOW_CONTRIB_VERBS_VERBS_SERVER_LIB_H_
#define THIRD_PARTY_TENSORFLOW_CONTRIB_VERBS_VERBS_SERVER_LIB_H_

#ifdef TENSORFLOW_USE_VERBS

#include "tensorflow/contrib/verbs/grpc_verbs_service.h"
#include "tensorflow/contrib/verbs/rdma_mgr.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_server_lib.h"

namespace tensorflow {

class VerbsServer : public GrpcServer {
 protected:
  VerbsServer(const ServerDef& server_def, Env* env);

 public:
  static Status Create(const ServerDef& server_def, Env* env,
                       std::unique_ptr<ServerInterface>* out_server);

  // Destruction is only supported in the factory method. Clean
  // shutdown is not currently implemented for this server type.
  virtual ~VerbsServer() override;

  // Implementations of ServerInterface methods.
  Status Start() override;
  Status Join() override;

 protected:
  Status Init(ServiceInitFunction service_func,
              RendezvousMgrCreationFunction rendezvous_mgr_func);
  Status ChannelCacheFactory(const ServerDef& server_def,
                             GrpcChannelCache** channel_cache);

 private:
  RdmaMgr* rdma_mgr_;

  // Guards state transitions.
  mutex mu_;

  enum State { DISCONNECTED, CONNECTED };
  State verbs_state_ GUARDED_BY(mu_);

  GrpcVerbsService* verbs_service_ = nullptr;
  std::unique_ptr<Thread> verbs_thread_ GUARDED_BY(mu_);
  GrpcChannelCache* channel_cache_ = nullptr;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_USE_VERBS
#endif  // THIRD_PARTY_TENSORFLOW_CONTRIB_VERBS_VERBS_SERVER_LIB_H_
