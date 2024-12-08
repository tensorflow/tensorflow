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
namespace {

// Fake alternative data transfer protocols:
// - good: Does not error.
// - bad_with_primary_fallback: Errors at initialization and then falls back.
// - bad_with_secondary_fallback: Initializes successfully, but errors at
// `GetElement` and then falls back.
constexpr const char kGoodProtocol[] = "good";
constexpr const char kBadProtocolWithPrimaryFallback[] =
    "bad_with_primary_fallback";
constexpr const char kBadProtocolWithSecondaryFallback[] =
    "bad_with_secondary_fallback";

class TestServer : public DataTransferServer {
 public:
  explicit TestServer(DataTransferServer::GetElementT get_element)
      : get_element_(get_element) {}

  virtual absl::Status GetElement(const GetElementRequest& req,
                                  GetElementResult& result) {
    return get_element_(&req, &result);
  }

  absl::Status Start(const experimental::WorkerConfig& config) override {
    return absl::OkStatus();
  }

  int Port() const override { return -1; }

 private:
  DataTransferServer::GetElementT get_element_;
};

class TestServerFailsWithSecondaryFallback : public TestServer {
 public:
  explicit TestServerFailsWithSecondaryFallback(
      DataTransferServer::GetElementT get_element)
      : TestServer(get_element) {}

  absl::Status GetElement(const GetElementRequest& req,
                          GetElementResult& result) override {
    return absl::InternalError("Bad get element.");
  }
};

class TestClient : public DataTransferClient {
 public:
  explicit TestClient(std::shared_ptr<TestServer> server) : server_(server) {}

  absl::Status GetElement(const GetElementRequest& req,
                          GetElementResult& result) override {
    return server_->GetElement(req, result);
  }

  void TryCancel() override {}

 private:
  std::shared_ptr<TestServer> server_;
};

class DataTransferRegistrar {
 public:
  DataTransferRegistrar() {
    // "good".
    RegisterServer<TestServer>(kGoodProtocol, good_);
    RegisterClient<TestServer>(kGoodProtocol, good_);

    // "bad_with_primary_fallback".
    RegisterDummyServer(kBadProtocolWithPrimaryFallback);
    RegisterBadClient(kBadProtocolWithPrimaryFallback);

    // "bad_with_secondary_fallback".
    RegisterServer<TestServerFailsWithSecondaryFallback>(
        kBadProtocolWithSecondaryFallback, bad_with_secondary_fallback_);
    RegisterClient<TestServerFailsWithSecondaryFallback>(
        kBadProtocolWithSecondaryFallback, bad_with_secondary_fallback_);
  }

 private:
  template <typename T>
  void RegisterServer(const std::string& protocol,
                      std::shared_ptr<T> my_server) {
    DataTransferServer::Register(
        protocol, [&my_server](DataTransferServer::GetElementT get_element,
                               std::shared_ptr<DataTransferServer>* server) {
          my_server = std::make_shared<T>(get_element);
          *server = my_server;
          return absl::OkStatus();
        });
  }

  template <typename T>
  void RegisterClient(const std::string& protocol,
                      std::shared_ptr<T> my_server) {
    DataTransferClient::Register(
        protocol, [&my_server](DataTransferClient::Config config,
                               std::unique_ptr<DataTransferClient>* client) {
          *client = std::make_unique<TestClient>(my_server);
          return absl::OkStatus();
        });
  }

  void RegisterDummyServer(const std::string& protocol) {
    DataTransferServer::Register(
        protocol, [](DataTransferServer::GetElementT get_element,
                     std::shared_ptr<DataTransferServer>* server) {
          *server = std::make_shared<TestServer>(get_element);
          return absl::OkStatus();
        });
  }

  void RegisterBadClient(const std::string& protocol) {
    DataTransferClient::Register(
        protocol, [](DataTransferClient::Config config,
                     std::unique_ptr<DataTransferClient>* client) {
          return absl::InternalError("Bad client.");
        });
  }

  std::shared_ptr<TestServer> good_ = nullptr;
  std::shared_ptr<TestServerFailsWithSecondaryFallback>
      bad_with_secondary_fallback_ = nullptr;
};

static DataTransferRegistrar data_transfer_registrar;

}  // namespace
}  // namespace data
}  // namespace tensorflow
