/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_DATA_SERVICE_DATA_TRANSFER_H_
#define TENSORFLOW_CORE_DATA_SERVICE_DATA_TRANSFER_H_

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "absl/strings/string_view.h"
#include "absl/types/optional.h"
#include "tensorflow/core/data/service/worker.pb.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/dataset.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/protobuf/service_config.pb.h"

namespace tensorflow {
namespace data {

// The result of a GetElement request. Exactly one of the following will be
// true: (1) `components` is nonempty (2) `end_of_sequence` is true (3) `skip`
// is true.
struct GetElementResult {
  GetElementResult() = default;
  GetElementResult(const GetElementResult&) = delete;
  GetElementResult& operator=(const GetElementResult&) = delete;
  GetElementResult(GetElementResult&&) = default;
  GetElementResult& operator=(GetElementResult&&) = default;

  // Creates a copy of this result. This is used to create multiple copies of
  // the same cached value.
  GetElementResult Copy() const;

  // Estimated memory used by this object, measured in bytes.
  size_t EstimatedMemoryUsageBytes() const;

  // A dataset element produced by a GetElement request.
  std::vector<Tensor> components;
  // The element's index within the task it came from.
  int64_t element_index = 0;
  // If true, indicates that there is no more data to read.
  bool end_of_sequence = false;
  // If true, indicates that there is still data, but the caller should skip
  // reading from the worker. This is used for load balancing when doing round
  // robin reads.
  bool skip = false;
};

// Client for communicating with the tf.data service transfer server.
class DataTransferClient {
 public:
  struct Config {
    absl::string_view protocol;
    std::string address;
    const DeviceBase::AcceleratorDeviceInfo* accelerator_device_info;
    Allocator* allocator;
  };
  using ClientFactoryT =
      std::function<absl::Status(Config, std::unique_ptr<DataTransferClient>*)>;
  virtual ~DataTransferClient() = default;

  // Fetches the next element.
  virtual absl::Status GetElement(const GetElementRequest& req,
                                  GetElementResult& result) = 0;

  // Makes a best effort to cancel all outstanding calls in progress for the
  // client, and causes further calls to return Cancelled status.
  virtual void TryCancel() = 0;

  // Registers a DataTransferClient factory under `name`.
  static void Register(std::string name, ClientFactoryT factory);

  // Builds a DataTransferClient from the factory registered under `name`.
  static absl::Status Build(std::string name, Config config,
                            std::unique_ptr<DataTransferClient>* out);

  // Returns a string describing properties of the client relevant for checking
  // compatibility with a server for a given protocol.
  virtual absl::StatusOr<std::string> GetCompatibilityInfo() const {
    return std::string();
  }

  // Returns an error if the client is incompatible with a server which has the
  // properties described in `server_compatibility_info`.
  virtual absl::Status CheckCompatibility(
      const std::string& server_compatibility_info) const {
    return absl::OkStatus();
  }

 protected:
  Env* const env_ = Env::Default();
};

// Server for communicating with the tf.data service transfer client.
class DataTransferServer {
 public:
  using GetElementT =
      std::function<absl::Status(const GetElementRequest*, GetElementResult*)>;
  using ServerFactoryT = std::function<absl::Status(
      GetElementT, std::shared_ptr<DataTransferServer>*)>;
  virtual ~DataTransferServer() = default;

  // Starts DataTransferServer, it should be available for requests afterwards.
  virtual absl::Status Start(const experimental::WorkerConfig& config) = 0;

  // Return the port that this server is listening on.
  virtual int Port() const = 0;

  // Register a DataTransferServer factory under `name`.
  static void Register(std::string name, ServerFactoryT factory);

  // Builds a DataTransferServer from the factory registered with `name`.
  static absl::Status Build(std::string name, GetElementT get_element,
                            std::shared_ptr<DataTransferServer>* out);

  // Returns a string describing properties of the server relevant for checking
  // compatibility with a client for a given protocol.
  virtual absl::StatusOr<std::string> GetCompatibilityInfo() const {
    return std::string();
  }

  // If `true`, data service clients should fall back to gRPC for this server if
  // they fail to create a data transfer client for it.
  virtual bool FallBackToGrpcAtClientCreationTime() const { return true; }

  // If `true`, data service clients should fall back to gRPC for this server if
  // it nonretryably fails to transfer an element.
  virtual bool FallBackToGrpcAtGetElementTime() const { return true; }
};

}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DATA_SERVICE_DATA_TRANSFER_H_
