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

#ifndef TENSORFLOW_CORE_DATA_SERVICE_SERVER_LIB_H_
#define TENSORFLOW_CORE_DATA_SERVICE_SERVER_LIB_H_

#include "grpcpp/server.h"
#include "grpcpp/server_builder.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
namespace data {

// Forward declared because transitively depending on .grpc.pb.h files causes
// issues in the pywrap build.
class GrpcMasterImpl;
class GrpcWorkerImpl;

// A grpc server for the dataset service.
class GrpcDataServerBase {
 public:
  // Constructs a dataset server with the specified port. If the port is 0, the
  // server will find an available port in `Start()`. The chosen port can be
  // found in the output of `Target()`.
  //
  // master_address is only needed for worker data servers.
  GrpcDataServerBase(int requested_port, const std::string& protocol);
  virtual ~GrpcDataServerBase() {}

  // Starts the server running asynchronously.
  Status Start();

  // Stops the server. This will block until all outstanding requests complete.
  void Stop();

  // Blocks until the server stops.
  void Join();

  // Returns the target string for the server. Only valid after calling Start().
  std::string Target();

 protected:
  virtual void AddServiceToBuilder(::grpc::ServerBuilder* builder) = 0;
  // Starts the service. This will be called after building the service, so
  // bound_port() will return the actual bound port.
  virtual Status StartServiceInternal() = 0;

  int bound_port() { return bound_port_; }

  const int requested_port_;
  const std::string protocol_;

 private:
  int bound_port_;

  std::unique_ptr<grpc::Server> server_;
};

class MasterGrpcDataServer : public GrpcDataServerBase {
 public:
  MasterGrpcDataServer(int requested_port, const std::string& protocol);
  ~MasterGrpcDataServer() override;

  // Returns the number of tasks created by the master.
  Status NumTasks(int* num_tasks);

 protected:
  void AddServiceToBuilder(grpc::ServerBuilder* builder) override;
  Status StartServiceInternal() override { return Status::OK(); }

 private:
  // Owned. We use a raw pointer because GrpcMasterImpl is forward-declared.
  GrpcMasterImpl* service_;
};

class WorkerGrpcDataServer : public GrpcDataServerBase {
 public:
  WorkerGrpcDataServer(int requested_port, const std::string& protocol,
                       const std::string& master_address);
  ~WorkerGrpcDataServer() override;

 protected:
  void AddServiceToBuilder(grpc::ServerBuilder* builder) override;
  Status StartServiceInternal() override;

 private:
  const std::string master_address_;
  // Owned. We use a raw pointer because GrpcWorkerImpl is forward-declared.
  GrpcWorkerImpl* service_;
};

// Creates a master dataset server and stores it in `*out_server`.
Status NewMasterServer(int port, const std::string& protocol,
                       std::unique_ptr<MasterGrpcDataServer>* out_server);

// Creates a worker dataset server and stores it in `*out_server`.
Status NewWorkerServer(int port, const std::string& protocol,
                       const std::string& master_address,
                       std::unique_ptr<WorkerGrpcDataServer>* out_server);

}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DATA_SERVICE_SERVER_LIB_H_
