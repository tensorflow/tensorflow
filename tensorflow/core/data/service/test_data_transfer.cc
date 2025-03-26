/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#include <memory>

#include "absl/status/status.h"
#include "tensorflow/core/data/service/data_transfer.h"

namespace tensorflow {
namespace data {

// Fake alternative data transfer protocols:
//
// - good: No errors or fallback.
//
// - bad_with_primary_fallback: Fails at data transfer client creation time and
// falls back to gRPC.
//
// - bad_without_primary_fallback: Fails at data transfer client creation time
// and doesn't fall back, taking down the entire data service client.
//
// - bad_with_secondary_fallback: Fails at get element time and falls back to
// gRPC.
//
// - bad_without_secondary_fallback: Fails at get element time and doesn't fall
// back, taking down the entire data service client.
//
constexpr const char kGoodProtocol[] = "good";
constexpr const char kBadProtocolWithPrimaryFallback[] =
    "bad_with_primary_fallback";
constexpr const char kBadProtocolWithoutPrimaryFallback[] =
    "bad_without_primary_fallback";
constexpr const char kBadProtocolWithSecondaryFallback[] =
    "bad_with_secondary_fallback";
constexpr const char kBadProtocolWithoutSecondaryFallback[] =
    "bad_without_secondary_fallback";

// A server that works.
class GoodTestServer : public DataTransferServer {
 public:
  explicit GoodTestServer(DataTransferServer::GetElementT get_element,
                          bool fall_back_to_grpc_at_client_creation_time = true,
                          bool fall_back_to_grpc_at_get_element_time = true)
      : get_element_(get_element),
        fall_back_to_grpc_at_client_creation_time_(
            fall_back_to_grpc_at_client_creation_time),
        fall_back_to_grpc_at_get_element_time_(
            fall_back_to_grpc_at_get_element_time) {}

  virtual absl::Status GetElement(const GetElementRequest& req,
                                  GetElementResult& result) {
    return get_element_(&req, &result);
  }

  bool FallBackToGrpcAtClientCreationTime() const override {
    return fall_back_to_grpc_at_client_creation_time_;
  }

  bool FallBackToGrpcAtGetElementTime() const override {
    return fall_back_to_grpc_at_get_element_time_;
  }

  absl::Status Start(const experimental::WorkerConfig& config) override {
    return absl::OkStatus();
  }

  int Port() const override { return -1; }

 private:
  DataTransferServer::GetElementT get_element_;
  bool fall_back_to_grpc_at_client_creation_time_;
  bool fall_back_to_grpc_at_get_element_time_;
};

// A server that doesn't work (by failing at get element time).
class BadTestServer : public GoodTestServer {
 public:
  explicit BadTestServer(DataTransferServer::GetElementT get_element,
                         bool fall_back_to_grpc_at_client_creation_time = true,
                         bool fall_back_to_grpc_at_get_element_time = true)
      : GoodTestServer(get_element, fall_back_to_grpc_at_client_creation_time,
                       fall_back_to_grpc_at_get_element_time) {}

  absl::Status GetElement(const GetElementRequest& req,
                          GetElementResult& result) override {
    return absl::InternalError("Bad get element.");
  }
};

// A working client for a server that may or may not work.
template <typename TestServerT>
class TestClient : public DataTransferClient {
 public:
  explicit TestClient(std::shared_ptr<TestServerT> server) : server_(server) {}

  absl::Status GetElement(const GetElementRequest& req,
                          GetElementResult& result) override {
    return server_->GetElement(req, result);
  }

  void TryCancel() override {}

 private:
  std::shared_ptr<TestServerT> server_;
};

class DataTransferRegistrar {
 public:
  DataTransferRegistrar() {
    // "good".
    RegisterServer<GoodTestServer>(kGoodProtocol, good_);
    RegisterClient<GoodTestServer>(kGoodProtocol, good_);

    // "bad_with_primary_fallback".
    RegisterUnusedServerForBadClient(
        kBadProtocolWithPrimaryFallback,
        /*fall_back_to_grpc_at_client_creation_time=*/true);
    RegisterBadClient(kBadProtocolWithPrimaryFallback);

    // "bad_without_primary_fallback".
    RegisterUnusedServerForBadClient(
        kBadProtocolWithoutPrimaryFallback,
        /*fall_back_to_grpc_at_client_creation_time=*/false);
    RegisterBadClient(kBadProtocolWithoutPrimaryFallback);

    // "bad_with_secondary_fallback".
    RegisterServer<BadTestServer>(
        kBadProtocolWithSecondaryFallback, bad_with_secondary_fallback_,
        /*fall_back_to_grpc_at_get_element_time=*/true);
    RegisterClient<BadTestServer>(kBadProtocolWithSecondaryFallback,
                                  bad_with_secondary_fallback_);

    // "bad_without_secondary_fallback".
    RegisterServer<BadTestServer>(
        kBadProtocolWithoutSecondaryFallback, bad_without_secondary_fallback_,
        /*fall_back_to_grpc_at_get_element_time=*/false);
    RegisterClient<BadTestServer>(kBadProtocolWithoutSecondaryFallback,
                                  bad_without_secondary_fallback_);
  }

 private:
  // Registers a server that may or may not work.
  template <typename TestServerT>
  void RegisterServer(const std::string& protocol,
                      std::shared_ptr<TestServerT>& my_server,
                      bool fall_back_to_grpc_at_get_element_time = true) {
    DataTransferServer::Register(
        protocol, [&my_server, fall_back_to_grpc_at_get_element_time](
                      DataTransferServer::GetElementT get_element,
                      std::shared_ptr<DataTransferServer>* server) {
          my_server = std::make_shared<TestServerT>(
              get_element, /*fall_back_to_grpc_at_client_creation_time=*/true,
              fall_back_to_grpc_at_get_element_time);
          *server = my_server;
          return absl::OkStatus();
        });
  }

  // Registers a working client for a server that may or may not work.
  template <typename TestServerT>
  void RegisterClient(const std::string& protocol,
                      std::shared_ptr<TestServerT>& my_server) {
    DataTransferClient::Register(
        protocol, [&](DataTransferClient::Config config,
                      std::unique_ptr<DataTransferClient>* client) {
          *client = std::make_unique<TestClient<TestServerT>>(my_server);
          return absl::OkStatus();
        });
  }

  // Registers a working server that shouldn't get used (because its client
  // should fail first, which may or may not result in a fall back).
  void RegisterUnusedServerForBadClient(
      const std::string& protocol,
      bool fall_back_to_grpc_at_client_creation_time) {
    DataTransferServer::Register(
        protocol, [fall_back_to_grpc_at_client_creation_time](
                      DataTransferServer::GetElementT get_element,
                      std::shared_ptr<DataTransferServer>* server) {
          *server = std::make_shared<GoodTestServer>(
              get_element, fall_back_to_grpc_at_client_creation_time);
          return absl::OkStatus();
        });
  }

  // Registers a nonworking client (via a client creation callback that fails).
  void RegisterBadClient(const std::string& protocol) {
    DataTransferClient::Register(
        protocol, [](DataTransferClient::Config config,
                     std::unique_ptr<DataTransferClient>* client) {
          return absl::InternalError("Bad client.");
        });
  }

  std::shared_ptr<GoodTestServer> good_ = nullptr;
  std::shared_ptr<BadTestServer> bad_with_secondary_fallback_ = nullptr;
  std::shared_ptr<BadTestServer> bad_without_secondary_fallback_ = nullptr;
};

static DataTransferRegistrar data_transfer_registrar;

}  // namespace data
}  // namespace tensorflow
