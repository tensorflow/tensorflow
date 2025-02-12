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

#include <memory>
#include <string>
#include <vector>

#include "grpcpp/server.h"
#include "grpcpp/server_builder.h"
#include "tensorflow/core/data/service/common.pb.h"
#include "tensorflow/core/data/service/data_transfer.h"
#include "tensorflow/core/data/service/export.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/profiler/rpc/profiler_service_impl.h"
#include "tensorflow/core/protobuf/service_config.pb.h"

namespace tensorflow {
namespace data {

// Forward declared because transitively depending on .grpc.pb.h files causes
// issues in the pywrap build.
class GrpcDispatcherImpl;
class GrpcWorkerImpl;

// A grpc server for the tf.data service.
class GrpcDataServerBase {
 public:
  // Constructs a tf.data server with the specified port. If the port is 0, the
  // server will find an available port in `Start()`. The chosen port can be
  // found by calling `BoundPort()`.
  GrpcDataServerBase(
      int requested_port, const std::string& protocol,
      const std::string& server_type,
      std::vector<std::unique_ptr<::grpc::ServerBuilderOption>> options = {});
  virtual ~GrpcDataServerBase() = default;

  // Starts the server running asynchronously.
  absl::Status Start();

  // Stops the server. This will block until all outstanding requests complete.
  void Stop();

  // Blocks until the server stops.
  void Join();

  // Returns the port bound by the server. Only valid after calling Start().
  int BoundPort();

  // Exports the server state to improve debuggability.
  virtual ServerStateExport ExportState() const = 0;

 protected:
  virtual void AddDataServiceToBuilder(::grpc::ServerBuilder& builder) = 0;
  void AddProfilerServiceToBuilder(::grpc::ServerBuilder& builder);
  // Starts the service. This will be called after building the service, so
  // bound_port() will return the actual bound port.
  virtual absl::Status StartServiceInternal() = 0;
  virtual void StopServiceInternal() {}

  int bound_port() { return bound_port_; }

  const int requested_port_;
  const std::string protocol_;
  const std::string server_type_;

 private:
  int bound_port_;
  bool started_ = false;
  bool stopped_ = false;

  std::unique_ptr<::grpc::Server> server_;
  // TensorFlow profiler service implementation.
  std::unique_ptr<grpc::ProfilerService::Service> profiler_service_ = nullptr;
  std::vector<std::unique_ptr<::grpc::ServerBuilderOption>> server_options_;
};

// A wrapper for `SnapshotStreamInfo` for use with pybind.
struct SnapshotStreamInfoWrapper {
  SnapshotStreamInfoWrapper() = default;
  explicit SnapshotStreamInfoWrapper(const SnapshotStreamInfo& info)
      : index(info.index()), state(info.state()) {}
  int64_t index;
  int64_t state;
};

class DispatchGrpcDataServer : public GrpcDataServerBase {
 public:
  explicit DispatchGrpcDataServer(
      const experimental::DispatcherConfig& config,
      std::vector<std::unique_ptr<::grpc::ServerBuilderOption>> options = {});
  ~DispatchGrpcDataServer() override;

  // Returns the number of workers registered with the dispatcher.
  absl::Status NumWorkers(int* num_workers);
  // Returns the number of active (non-finished) iterations running on the
  // dispatcher.
  size_t NumActiveIterations();
  // Returns information about all the streams for the snapshot at `path`.
  absl::Status SnapshotStreams(const std::string& path,
                               std::vector<SnapshotStreamInfoWrapper>* streams);

  ServerStateExport ExportState() const override;

 protected:
  void AddDataServiceToBuilder(::grpc::ServerBuilder& builder) override;
  absl::Status StartServiceInternal() override;
  void StopServiceInternal() override;

 private:
  const experimental::DispatcherConfig config_;
  // Owned. We use a raw pointer because GrpcDispatcherImpl is forward-declared.
  GrpcDispatcherImpl* service_;
};

// A wrapper for `SnapshotTaskProgress` for use with pybind.
struct SnapshotTaskProgressWrapper {
  SnapshotTaskProgressWrapper() = default;
  explicit SnapshotTaskProgressWrapper(const SnapshotTaskProgress& progress)
      : snapshot_task_base_path(progress.snapshot_task().base_path()),
        snapshot_task_stream_index(progress.snapshot_task().stream_index()),
        completed(progress.completed()) {}
  std::string snapshot_task_base_path;
  int64_t snapshot_task_stream_index;
  bool completed;
};

class WorkerGrpcDataServer : public GrpcDataServerBase {
 public:
  explicit WorkerGrpcDataServer(
      const experimental::WorkerConfig& config,
      std::vector<std::unique_ptr<::grpc::ServerBuilderOption>> options = {});
  ~WorkerGrpcDataServer() override;

  // Returns the number of tasks currently being executed by the worker.
  absl::Status NumTasks(int* num_tasks);

  // Returns the progresses of the snapshot tasks currently being executed by
  // the worker.
  absl::Status SnapshotTaskProgresses(
      std::vector<SnapshotTaskProgressWrapper>* snapshot_task_progresses);

  ServerStateExport ExportState() const override;

 protected:
  void AddDataServiceToBuilder(::grpc::ServerBuilder& builder) override;
  absl::Status StartServiceInternal() override;
  void StopServiceInternal() override;

 private:
  // If an alternative data transfer protocol is configured, tries to start a
  // transfer server for it, adding an entry to `transfer_servers` if
  // successful.
  void MaybeStartAlternativeDataTransferServer(
      std::vector<DataTransferServerInfo>& transfer_servers);

  const experimental::WorkerConfig config_;
  // Owned. We use a raw pointer because GrpcWorkerImpl is forward-declared.
  GrpcWorkerImpl* service_;
  std::shared_ptr<DataTransferServer> transfer_server_;
};

// Creates a dispatch tf.data server and stores it in `out_server`.
absl::Status NewDispatchServer(
    const experimental::DispatcherConfig& config,
    std::unique_ptr<DispatchGrpcDataServer>& out_server);

// Creates a worker tf.data server and stores it in `out_server`.
absl::Status NewWorkerServer(const experimental::WorkerConfig& config,
                             std::unique_ptr<WorkerGrpcDataServer>& out_server);

}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DATA_SERVICE_SERVER_LIB_H_
