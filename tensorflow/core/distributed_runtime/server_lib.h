/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_SERVER_LIB_H_
#define TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_SERVER_LIB_H_

#include <memory>

#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/protobuf/tensorflow_server.pb.h"

namespace tensorflow {

class CoordinationServiceAgent;
class DeviceMgr;
class EagerContext;
class WorkerEnv;
class MasterEnv;

// This library supports a registration/factory-based mechanism for
// creating TensorFlow server objects. Each server implementation must
// have an accompanying implementation of ServerFactory, and create a
// static "registrar" object that calls `ServerFactory::Register()`
// with an instance of the factory class. See "rpc/grpc_server_lib.cc"
// for an example.

// Represents a single TensorFlow server that exports Master and Worker
// services.
class ServerInterface {
 public:
  ServerInterface() {}
  virtual ~ServerInterface() {}

  // Starts the server running asynchronously. Returns OK on success, otherwise
  // returns an error.
  virtual Status Start() = 0;

  // Stops the server asynchronously. Returns OK on success, otherwise returns
  // an error.
  //
  // After calling `Stop()`, the caller may call `Join()` to block until the
  // server has stopped.
  virtual Status Stop() = 0;

  // Blocks until the server has stopped. Returns OK on success, otherwise
  // returns an error.
  virtual Status Join() = 0;

  // Returns a target string that can be used to connect to this server using
  // `tensorflow::NewSession()`.
  virtual const string target() const = 0;

  virtual WorkerEnv* worker_env() = 0;
  virtual MasterEnv* master_env() = 0;

  // Update the set of workers that can be reached by the server
  virtual Status UpdateServerDef(const ServerDef& server_def) = 0;

  // Functions to operate on service-specific properties.
  //
  // Add master eager context to local eager service in order to handle enqueue
  // requests from remote workers.
  virtual Status AddMasterEagerContextToEagerService(
      const tensorflow::uint64 context_id, EagerContext* context) = 0;
  // Set coordination service agent instance to coordination service RPC handler
  virtual Status SetCoordinationServiceAgentInstance(
      CoordinationServiceAgent* agent) = 0;

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(ServerInterface);
};

class ServerFactory {
 public:
  struct Options {
    // Local DeviceMgr to use.
    tensorflow::DeviceMgr* local_device_mgr;
  };
  // Creates a new server based on the given `server_def`, and stores
  // it in `*out_server`. Returns OK on success, otherwise returns an
  // error.
  virtual Status NewServer(const ServerDef& server_def, const Options& options,
                           std::unique_ptr<ServerInterface>* out_server) = 0;

  // Returns true if and only if this factory can create a server
  // based on the given `server_def`.
  virtual bool AcceptsOptions(const ServerDef& server_def) = 0;

  virtual ~ServerFactory() {}

  // For each `ServerFactory` subclass, an instance of that class must
  // be registered by calling this method.
  //
  // The `server_type` must be unique to the server factory.
  static void Register(const string& server_type, ServerFactory* factory);

  // Looks up a factory that can create a server based on the given
  // `server_def`, and stores it in `*out_factory`. Returns OK on
  // success, otherwise returns an error.
  static Status GetFactory(const ServerDef& server_def,
                           ServerFactory** out_factory);
};

// Creates a server based on the given `server_def`, and stores it in
// `*out_server`. Returns OK on success, otherwise returns an error.
Status NewServer(const ServerDef& server_def,
                 std::unique_ptr<ServerInterface>* out_server);
Status NewServerWithOptions(const ServerDef& server_def,
                            const ServerFactory::Options& options,
                            std::unique_ptr<ServerInterface>* out_server);

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_SERVER_LIB_H_
