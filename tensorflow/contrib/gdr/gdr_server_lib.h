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

#ifndef TENSORFLOW_CONTRIB_GDR_GDR_SERVER_LIB_H_
#define TENSORFLOW_CONTRIB_GDR_GDR_SERVER_LIB_H_

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

#endif  // TENSORFLOW_CONTRIB_GDR_GDR_SERVER_LIB_H_
