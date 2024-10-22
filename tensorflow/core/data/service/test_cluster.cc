/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/data/service/test_cluster.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "tensorflow/core/data/service/export.pb.h"
#include "tensorflow/core/data/service/server_lib.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/protobuf/data_service.pb.h"
#include "tensorflow/core/protobuf/service_config.pb.h"
#include "tsl/platform/env.h"

namespace tensorflow {
namespace data {
namespace {
constexpr const char kProtocol[] = "grpc";
}  // namespace

TestCluster::TestCluster(int num_workers,
                         std::optional<std::string> data_transfer_protocol)
    : num_workers_(num_workers),
      data_transfer_protocol_(data_transfer_protocol) {}

TestCluster::TestCluster(const TestCluster::Config& config)
    : num_workers_(config.num_workers), config_(config) {}

TestCluster::~TestCluster() {
  if (!config_.work_dir.empty()) {
    int64_t undeleted_files, undeleted_dirs;
    tsl::Env::Default()
        ->DeleteRecursively(config_.work_dir, &undeleted_files, &undeleted_dirs)
        .IgnoreError();
  }
}

absl::Status TestCluster::Initialize() {
  if (initialized_) {
    return errors::FailedPrecondition(
        "Test cluster has already been initialized.");
  }
  initialized_ = true;
  experimental::DispatcherConfig dispatcher_config;
  if (!config_.work_dir.empty()) {
    dispatcher_config.set_work_dir(config_.work_dir);
    dispatcher_config.set_fault_tolerant_mode(true);
  }
  dispatcher_config.set_protocol(kProtocol);
  for (int i = 0; i < num_workers_; ++i) {
    dispatcher_config.add_worker_addresses("localhost");
  }
  dispatcher_config.set_deployment_mode(DEPLOYMENT_MODE_COLOCATED);
  dispatcher_config.set_job_gc_check_interval_ms(
      config_.job_gc_check_interval_ms);
  dispatcher_config.set_job_gc_timeout_ms(config_.job_gc_timeout_ms);
  dispatcher_config.set_client_timeout_ms(config_.client_timeout_ms);
  dispatcher_config.set_worker_max_concurrent_snapshots(
      config_.worker_max_concurrent_snapshots);
  TF_RETURN_IF_ERROR(NewDispatchServer(dispatcher_config, dispatcher_));
  TF_RETURN_IF_ERROR(dispatcher_->Start());
  dispatcher_address_ = absl::StrCat("localhost:", dispatcher_->BoundPort());
  workers_.reserve(num_workers_);
  worker_addresses_.reserve(num_workers_);
  for (int i = 0; i < num_workers_; ++i) {
    TF_RETURN_IF_ERROR(
        AddWorker(/*port=*/std::nullopt, data_transfer_protocol_));
  }
  return absl::OkStatus();
}

absl::Status TestCluster::AddWorker(
    std::optional<int> port,
    std::optional<std::string> data_transfer_protocol) {
  std::unique_ptr<WorkerGrpcDataServer> worker;
  experimental::WorkerConfig config;
  if (port.has_value()) {
    config.set_port(*port);
  }
  config.set_protocol(kProtocol);
  if (data_transfer_protocol.has_value()) {
    config.set_data_transfer_protocol(*data_transfer_protocol);
  }
  config.set_dispatcher_address(dispatcher_address_);
  std::string worker_address =
      port.has_value() ? absl::StrCat("localhost:", *port) : "localhost:%port%";
  config.set_worker_address(worker_address);
  config.set_heartbeat_interval_ms(config_.worker_heartbeat_interval_ms);
  TF_RETURN_IF_ERROR(NewWorkerServer(config, worker));
  TF_RETURN_IF_ERROR(worker->Start());
  worker_addresses_.push_back(absl::StrCat("localhost:", worker->BoundPort()));
  workers_.push_back(std::move(worker));
  return absl::OkStatus();
}

std::string TestCluster::DispatcherAddress() const {
  return dispatcher_address_;
}

std::string TestCluster::WorkerAddress(int index) const {
  DCHECK_GE(index, 0);
  DCHECK_LT(index, worker_addresses_.size());
  return worker_addresses_[index];
}

void TestCluster::StopWorker(size_t index) {
  DCHECK_GE(index, 0);
  DCHECK_LT(index, worker_addresses_.size());
  workers_[index]->Stop();
}

void TestCluster::StopWorkers() {
  for (std::unique_ptr<WorkerGrpcDataServer>& worker : workers_) {
    worker->Stop();
  }
}

ServerStateExport TestCluster::ExportDispatcherState() const {
  return dispatcher_->ExportState();
}

ServerStateExport TestCluster::ExportWorkerState(size_t index) const {
  return workers_[index]->ExportState();
}

}  // namespace data
}  // namespace tensorflow
