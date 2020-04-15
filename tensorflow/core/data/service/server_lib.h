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

#include "grpcpp/impl/codegen/service_type.h"
#include "grpcpp/server.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
namespace data {

// A grpc server for the dataset service.
class GrpcDataServer {
 public:
  // Constructs a dataset server with the specified port. If the port is 0, the
  // server will find an available port in `Start()`. The chosen port can be
  // found in the output of `Target()`.
  //
  // master_address is only needed for worker data servers.
  explicit GrpcDataServer(int requested_port, const std::string& protocol,
                          bool is_master, const std::string& master_address);

  // Starts the server running asynchronously.
  Status Start();

  // Stops the server. This will block until all outstanding requests complete.
  void Stop();

  // Blocks until the server stops.
  void Join();

  // Returns the target string for the server. Only valid after calling Start().
  std::string Target();

 private:
  const int requested_port_;
  const std::string protocol_;
  const bool is_master_;
  const std::string master_address_;

  int bound_port_;

  std::unique_ptr<grpc::Service> service_;
  std::unique_ptr<grpc::Server> server_;
};

// Creates a master dataset server and stores it in `*out_server`.
Status NewMasterServer(int port, const std::string& protocol,
                       std::unique_ptr<GrpcDataServer>* out_server);

// Creates a worker dataset server and stores it in `*out_server`.
Status NewWorkerServer(int port, const std::string& protocol,
                       const std::string& master_address,
                       std::unique_ptr<GrpcDataServer>* out_server);

}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DATA_SERVICE_SERVER_LIB_H_
