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

#ifndef TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_MESSAGE_WRAPPERS_H_
#define TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_MESSAGE_WRAPPERS_H_

#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/cost_graph.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/step_stats.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor.pb_text.h"
#include "tensorflow/core/framework/versions.pb.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/protobuf/master.pb.h"
#include "tensorflow/core/protobuf/worker.pb.h"

namespace tensorflow {

////////////////////////////////////////////////////////////////////////////////
//
// Wrapper classes for the `MasterService.RunStep` request message.
//
// The `RunStepRequest` message can contain potentially large tensor
// data as part of its `feed` submessages. Here we provide specialized
// wrappers that avoid copying the tensor data wherever possible.
//
// See `RunStepRequest` in tensorflow/core/protobuf/master.proto for the
// protocol buffer definition.
//
////////////////////////////////////////////////////////////////////////////////

// Abstract interface for an immutable RunStepRequest message.
//
// This interface is typically used by server-side components in the
// TensorFlow master.
class RunStepRequestWrapper {
 public:
  virtual ~RunStepRequestWrapper() {}

  // REQUIRED: session_handle must be returned by a CreateSession call
  // to the same master service.
  virtual const string& session_handle() const = 0;

  // Partial run handle (optional). If specified, this will be a partial run
  // execution, run up to the specified fetches.
  virtual const string& partial_run_handle() const = 0;

  // Tensors to be fed in the step. Each feed is a named tensor.
  virtual size_t num_feeds() const = 0;
  virtual const string& feed_name(size_t i) const = 0;

  // Stores the content of the feed value at index `i` in `tensor`.
  virtual Status FeedValue(size_t i, Tensor* out_tensor) const = 0;
  virtual Status FeedValue(size_t i, TensorProto* out_tensor) const = 0;

  // Fetches. A list of tensor names. The caller expects a tensor to
  // be returned for each fetch[i] (see RunStepResponse.tensor). The
  // order of specified fetches does not change the execution order.
  virtual size_t num_fetches() const = 0;
  virtual const string& fetch_name(size_t i) const = 0;

  // Target Nodes. A list of node names. The named nodes will be run
  // to but their outputs will not be fetched.
  virtual size_t num_targets() const = 0;
  virtual const string& target_name(size_t i) const = 0;

  // Options for the run call.
  virtual const RunOptions& options() const = 0;

  // If true then some errors, e.g., execution errors that have long
  // error messages, may return an OK RunStepResponse with the actual
  // error saved in the status_code/status_error_message fields of the
  // response body. This is a workaround since the RPC subsystem may
  // truncate long metadata messages.
  virtual bool store_errors_in_response_body() const = 0;

  // Returns a human-readable representation of this message for debugging.
  virtual string DebugString() const = 0;

  // Returns the wrapped data as a protocol buffer message.
  virtual const RunStepRequest& ToProto() const = 0;
};

// Abstract interface for a mutable RunStepRequest message.
//
// See `RunStepRequestWrapper` above for a description of the fields.
class MutableRunStepRequestWrapper : public RunStepRequestWrapper {
 public:
  virtual void set_session_handle(const string& handle) = 0;
  virtual void set_partial_run_handle(const string& handle) = 0;
  virtual void add_feed(const string& name, const Tensor& value) = 0;
  virtual void add_fetch(const string& name) = 0;
  virtual void add_target(const string& name) = 0;
  virtual RunOptions* mutable_options() = 0;
  virtual void set_store_errors_in_response_body(bool store_errors) = 0;
};

// Specialized (and mutable) wrapper for RunStep requests between a client and
// master in the same address space.
class InMemoryRunStepRequest : public MutableRunStepRequestWrapper {
 public:
  // RunStepRequestWrapper methods.
  const string& session_handle() const override;
  const string& partial_run_handle() const override;
  size_t num_feeds() const override;
  const string& feed_name(size_t i) const override;
  Status FeedValue(size_t i, Tensor* out_tensor) const override;
  Status FeedValue(size_t i, TensorProto* out_tensor) const override;
  size_t num_fetches() const override;
  const string& fetch_name(size_t i) const override;
  size_t num_targets() const override;
  const string& target_name(size_t i) const override;
  const RunOptions& options() const override;
  string DebugString() const override;
  const RunStepRequest& ToProto() const override;
  bool store_errors_in_response_body() const override;

  // MutableRunStepRequestWrapper methods.
  void set_session_handle(const string& handle) override;
  void set_partial_run_handle(const string& handle) override;
  void add_feed(const string& name, const Tensor& value) override;
  void add_fetch(const string& name) override;
  void add_target(const string& name) override;
  RunOptions* mutable_options() override;
  void set_store_errors_in_response_body(bool store_errors) override;

 private:
  string session_handle_;
  string partial_run_handle_;
  gtl::InlinedVector<std::pair<string, Tensor>, 4> feeds_;
  gtl::InlinedVector<string, 4> fetches_;
  gtl::InlinedVector<string, 4> targets_;
  RunOptions options_;
  bool store_errors_in_response_body_ = false;

  // Holds a cached and owned representation of the proto
  // representation of this request, if needed, so that `ToProto()`
  // can return a const RunStepRequest&.
  // NOTE(mrry): Although calls to `ToProto()` on this class are
  // expected to be rare, retaining ownership of the returned message
  // makes it easier to return a reference from the proto-backed
  // representations.
  mutable std::unique_ptr<RunStepRequest> proto_version_;
};

// Wrapper for mutable RunStep requests that uses a protobuf message.
//
// This wrapper class should be used for RunStep requests between a
// client and master in different address spaces.
class MutableProtoRunStepRequest : public MutableRunStepRequestWrapper {
 public:
  // RunStepRequestWrapper methods.
  const string& session_handle() const override;
  const string& partial_run_handle() const override;
  size_t num_feeds() const override;
  const string& feed_name(size_t i) const override;
  Status FeedValue(size_t i, Tensor* out_tensor) const override;
  Status FeedValue(size_t i, TensorProto* out_tensor) const override;
  size_t num_fetches() const override;
  const string& fetch_name(size_t i) const override;
  size_t num_targets() const override;
  const string& target_name(size_t i) const override;
  const RunOptions& options() const override;
  string DebugString() const override;
  const RunStepRequest& ToProto() const override;
  bool store_errors_in_response_body() const override;

  // MutableRunStepRequestWrapper methods.
  void set_session_handle(const string& handle) override;
  void set_partial_run_handle(const string& handle) override;
  void add_feed(const string& name, const Tensor& value) override;
  void add_fetch(const string& name) override;
  void add_target(const string& name) override;
  RunOptions* mutable_options() override;
  void set_store_errors_in_response_body(bool store_errors) override;

 private:
  RunStepRequest request_;
};

// Wrapper for immutable RunStep requests that use a non-owned
// protobuf message.
//
// This interface is typically used by server-side components in the
// TensorFlow master, where the incoming message is a (possibly const)
// `RunStepRequest*`.
class ProtoRunStepRequest : public RunStepRequestWrapper {
 public:
  ProtoRunStepRequest(const RunStepRequest* request);

  // RunStepRequestWrapper methods.
  const string& session_handle() const override;
  const string& partial_run_handle() const override;
  size_t num_feeds() const override;
  const string& feed_name(size_t i) const override;
  Status FeedValue(size_t i, Tensor* out_tensor) const override;
  Status FeedValue(size_t i, TensorProto* out_tensor) const override;
  size_t num_fetches() const override;
  const string& fetch_name(size_t i) const override;
  size_t num_targets() const override;
  const string& target_name(size_t i) const override;
  const RunOptions& options() const override;
  string DebugString() const override;
  const RunStepRequest& ToProto() const override;
  bool store_errors_in_response_body() const override;

 private:
  const RunStepRequest* const request_;  // Not owned.
};

////////////////////////////////////////////////////////////////////////////////
//
// Wrapper classes for the `WorkerService.RunGraph` request message.
//
// The `RunGraphRequest` message can contain potentially large tensor
// data as part of its `send` submessages. Here we provide specialized
// wrappers that avoid copying the tensor data wherever possible.
//
// See `RunGraphRequest` in tensorflow/core/protobuf/worker.proto for the
// protocol buffer definition.
//
////////////////////////////////////////////////////////////////////////////////

// Abstract interface for an immutable RunStepRequest message.
//
// This interface is typically used by server-side components in the
// TensorFlow worker.
class RunGraphRequestWrapper {
 public:
  virtual ~RunGraphRequestWrapper() {}

  // The session handle used to register the graph. If empty, a single global
  // namespace is used.
  virtual const string& session_handle() const = 0;

  // REQUIRED: graph_handle must be returned by a RegisterGraph call
  // to the same WorkerService.
  virtual const string& graph_handle() const = 0;

  // A unique ID to distinguish different runs of the same graph.
  //
  // The master generates a global unique `step_id` to distinguish
  // different runs of the graph computation. Subgraphs communicate
  // (e.g., send/recv ops) with each other using `step_id` to
  // distinguish tensors generated by different runs.
  virtual int64 step_id() const = 0;

  // Options for this step.
  virtual const ExecutorOpts& exec_opts() const = 0;

  // Sends the tensors in "send" into the graph before the run.
  virtual size_t num_sends() const = 0;
  virtual const string& send_key(size_t i) const = 0;
  virtual Status SendValue(size_t i, Tensor* out_tensor) const = 0;

  // Fetches the keys into `RunGraphResponse.recv` after the run.
  virtual size_t num_recvs() const = 0;
  virtual const string& recv_key(size_t i) const = 0;

  // True if the RunGraphRequest is a partial run request.
  virtual bool is_partial() const = 0;

  // True if this is the last partial run request in a sequence of requests.
  virtual bool is_last_partial_run() const = 0;

  // If true then some errors, e.g., execution errors that have long
  // error messages, may return an OK RunStepResponse with the actual
  // error saved in the status_code/status_error_message fields of the
  // response body. This is a workaround since the RPC subsystem may
  // truncate long metadata messages.
  virtual bool store_errors_in_response_body() const = 0;

  // Returns the wrapped data as a protocol buffer message.
  virtual const RunGraphRequest& ToProto() const = 0;
};

// Abstract interface for a mutable RunGraphRequest message.
//
// See `RunGraphRequestWrapper` above for a description of the fields.
class MutableRunGraphRequestWrapper : public RunGraphRequestWrapper {
 public:
  virtual void set_session_handle(const string& handle) = 0;
  virtual void set_graph_handle(const string& handle) = 0;
  virtual void set_step_id(int64 step_id) = 0;
  virtual ExecutorOpts* mutable_exec_opts() = 0;

  // Stores the i^{th} feed value in `run_step_request` in this
  // request with the given `send_key`.
  virtual Status AddSendFromRunStepRequest(
      const RunStepRequestWrapper& run_step_request, size_t i,
      const string& send_key) = 0;

  virtual void add_recv_key(const string& recv_key) = 0;
  virtual void set_is_partial(bool is_partial) = 0;
  virtual void set_is_last_partial_run(bool is_last_partial_run) = 0;
  virtual void set_store_errors_in_response_body(bool store_errors) = 0;
};

class InMemoryRunGraphRequest : public MutableRunGraphRequestWrapper {
 public:
  // RunGraphRequestWrapper methods.
  const string& session_handle() const override;
  const string& graph_handle() const override;
  int64 step_id() const override;
  const ExecutorOpts& exec_opts() const override;
  size_t num_sends() const override;
  const string& send_key(size_t i) const override;
  Status SendValue(size_t i, Tensor* out_tensor) const override;
  size_t num_recvs() const override;
  const string& recv_key(size_t i) const override;
  bool is_partial() const override;
  bool is_last_partial_run() const override;
  const RunGraphRequest& ToProto() const override;
  bool store_errors_in_response_body() const override;

  // MutableRunGraphRequestWrapper methods.
  void set_session_handle(const string& handle) override;
  void set_graph_handle(const string& handle) override;
  void set_step_id(int64 step_id) override;
  ExecutorOpts* mutable_exec_opts() override;
  Status AddSendFromRunStepRequest(
      const RunStepRequestWrapper& run_step_request, size_t i,
      const string& send_key) override;
  void add_recv_key(const string& recv_key) override;
  void set_is_partial(bool is_partial) override;
  void set_is_last_partial_run(bool is_last_partial_run) override;
  void set_store_errors_in_response_body(bool store_errors) override;

 private:
  string session_handle_;
  string graph_handle_;
  int64 step_id_;
  ExecutorOpts exec_opts_;
  gtl::InlinedVector<std::pair<string, Tensor>, 4> sends_;
  gtl::InlinedVector<string, 4> recvs_;
  bool is_partial_ = false;
  bool is_last_partial_run_ = false;
  bool store_errors_in_response_body_ = false;

  // Holds a cached and owned representation of the proto
  // representation of this request, if needed, so that `ToProto()`
  // can return a const RunGraphRequest&.
  // NOTE(mrry): Although calls to `ToProto()` on this class are
  // expected to be rare, retaining ownership of the returned message
  // makes it easier to return a reference from the proto-backed
  // representations.
  mutable std::unique_ptr<RunGraphRequest> proto_version_;
};

class MutableProtoRunGraphRequest : public MutableRunGraphRequestWrapper {
 public:
  // RunGraphRequestWrapper methods.
  const string& session_handle() const override;
  const string& graph_handle() const override;
  int64 step_id() const override;
  const ExecutorOpts& exec_opts() const override;
  size_t num_sends() const override;
  const string& send_key(size_t i) const override;
  Status SendValue(size_t i, Tensor* out_tensor) const override;
  size_t num_recvs() const override;
  const string& recv_key(size_t i) const override;
  bool is_partial() const override;
  bool is_last_partial_run() const override;
  bool store_errors_in_response_body() const override;
  const RunGraphRequest& ToProto() const override;

  // MutableRunGraphRequestWrapper methods.
  void set_session_handle(const string& handle) override;
  void set_graph_handle(const string& handle) override;
  void set_step_id(int64 step_id) override;
  ExecutorOpts* mutable_exec_opts() override;
  Status AddSendFromRunStepRequest(
      const RunStepRequestWrapper& run_step_request, size_t i,
      const string& send_key) override;
  void add_recv_key(const string& recv_key) override;
  void set_is_partial(bool is_partial) override;
  void set_is_last_partial_run(bool is_last_partial_run) override;
  void set_store_errors_in_response_body(bool store_errors) override;

 private:
  RunGraphRequest request_;
};

class ProtoRunGraphRequest : public RunGraphRequestWrapper {
 public:
  ProtoRunGraphRequest(const RunGraphRequest* request);

  // RunGraphRequestWrapper methods.
  const string& session_handle() const override;
  const string& graph_handle() const override;
  int64 step_id() const override;
  const ExecutorOpts& exec_opts() const override;
  size_t num_sends() const override;
  const string& send_key(size_t i) const override;
  Status SendValue(size_t i, Tensor* out_tensor) const override;
  size_t num_recvs() const override;
  const string& recv_key(size_t i) const override;
  bool is_partial() const override;
  bool is_last_partial_run() const override;
  bool store_errors_in_response_body() const override;
  const RunGraphRequest& ToProto() const override;

 private:
  const RunGraphRequest* const request_;  // Not owned.
};

////////////////////////////////////////////////////////////////////////////////
//
// Wrapper classes for the `WorkerService.RunGraph` response message.
//
// The `RunGraphResponse` message can contain potentially large tensor
// data as part of its `recv` submessages. Here we provide specialized
// wrappers that avoid copying the tensor data wherever possible.
//
// See `RunGraphResponse` in tensorflow/core/protobuf/worker.proto for the
// protocol buffer definition.
//
////////////////////////////////////////////////////////////////////////////////

// Abstract interface for a mutable RunGraphResponse message.
//
// Note that there is no corresponding (immutable)
// RunGraphResponseWrapper class, because the RunGraphResponse object
// is always used as a mutable pointer.
class MutableRunGraphResponseWrapper {
 public:
  virtual ~MutableRunGraphResponseWrapper() {}

  // A list of tensors corresponding to those requested by
  // `RunGraphRequest.recv_key`.
  virtual size_t num_recvs() const = 0;
  virtual const string& recv_key(size_t i) const = 0;
  // NOTE: The following methods may perform a destructive read, for
  // efficiency.
  virtual Status RecvValue(size_t i, TensorProto* out_tensor) = 0;
  virtual Status RecvValue(size_t i, Tensor* out_tensor) = 0;
  virtual void AddRecv(const string& key, const Tensor& value) = 0;

  // Submessages that store performance statistics about the subgraph
  // execution, if necessary.
  virtual StepStats* mutable_step_stats() = 0;
  virtual CostGraphDef* mutable_cost_graph() = 0;
  virtual size_t num_partition_graphs() const = 0;
  virtual GraphDef* mutable_partition_graph(size_t i) = 0;
  virtual void AddPartitionGraph(const GraphDef& partition_graph) = 0;

  // Returned status if requested.
  virtual errors::Code status_code() const = 0;
  virtual const string& status_error_message() const = 0;
  virtual void set_status(const Status& status) = 0;

 protected:
  // Returns a mutable protobuf message that represents the contents of
  // this wrapper, for passing to an RPC subsystem that will populate
  // the message.
  //
  // NOTE: Only `WorkerInterface` subclasses may call this method. The
  // `InMemoryRunGraphResponse` subclass does not implement this
  // method, and attempts to call it will fail with a fatal
  // error. However, as long as callers always call
  // `WorkerInterface::RunGraphAsync()` with a wrapper object returned
  // from `WorkerInterface::CreateRunGraphResponse()` called on the
  // *same* WorkerInterface object, this error will never trigger.
  virtual RunGraphResponse* get_proto() = 0;
  friend class WorkerInterface;
};

class InMemoryRunGraphResponse : public MutableRunGraphResponseWrapper {
 public:
  // MutableRunGraphResponseWrapper methods.
  size_t num_recvs() const override;
  const string& recv_key(size_t i) const override;
  Status RecvValue(size_t i, TensorProto* out_tensor) override;
  Status RecvValue(size_t i, Tensor* out_tensor) override;
  void AddRecv(const string& key, const Tensor& value) override;
  StepStats* mutable_step_stats() override;
  CostGraphDef* mutable_cost_graph() override;
  size_t num_partition_graphs() const override;
  GraphDef* mutable_partition_graph(size_t i) override;
  void AddPartitionGraph(const GraphDef& partition_graph) override;
  errors::Code status_code() const override;
  const string& status_error_message() const override;
  void set_status(const Status& status) override;

 protected:
  // NOTE: This method is not implemented. See
  // MutableRunGraphResponseWrapper for an explanation.
  RunGraphResponse* get_proto() override;

 private:
  gtl::InlinedVector<std::pair<string, Tensor>, 4> recvs_;
  StepStats step_stats_;
  CostGraphDef cost_graph_;
  std::vector<GraphDef> partition_graphs_;
  // Store the code and message separately so that they can be updated
  // independently by setters.
  Status status_;
};

// Proto-based message wrapper for use on the client side of the RunGraph RPC.
class OwnedProtoRunGraphResponse : public MutableRunGraphResponseWrapper {
 public:
  // MutableRunGraphResponseWrapper methods.
  size_t num_recvs() const override;
  const string& recv_key(size_t i) const override;
  Status RecvValue(size_t i, TensorProto* out_tensor) override;
  Status RecvValue(size_t i, Tensor* out_tensor) override;
  void AddRecv(const string& key, const Tensor& value) override;
  StepStats* mutable_step_stats() override;
  CostGraphDef* mutable_cost_graph() override;
  size_t num_partition_graphs() const override;
  GraphDef* mutable_partition_graph(size_t i) override;
  void AddPartitionGraph(const GraphDef& partition_graph) override;
  errors::Code status_code() const override;
  const string& status_error_message() const override;
  void set_status(const Status& status) override;

 protected:
  RunGraphResponse* get_proto() override;

 private:
  RunGraphResponse response_;
};

// Proto-based message wrapper for use on the server side of the RunGraph RPC.
class NonOwnedProtoRunGraphResponse : public MutableRunGraphResponseWrapper {
 public:
  NonOwnedProtoRunGraphResponse(RunGraphResponse* response);

  // MutableRunGraphResponseWrapper methods.
  size_t num_recvs() const override;
  const string& recv_key(size_t i) const override;
  Status RecvValue(size_t i, TensorProto* out_tensor) override;
  Status RecvValue(size_t i, Tensor* out_tensor) override;
  void AddRecv(const string& key, const Tensor& value) override;
  StepStats* mutable_step_stats() override;
  CostGraphDef* mutable_cost_graph() override;
  size_t num_partition_graphs() const override;
  GraphDef* mutable_partition_graph(size_t i) override;
  void AddPartitionGraph(const GraphDef& partition_graph) override;
  errors::Code status_code() const override;
  const string& status_error_message() const override;
  void set_status(const Status& status) override;

 protected:
  RunGraphResponse* get_proto() override;

 private:
  RunGraphResponse* const response_;
};

////////////////////////////////////////////////////////////////////////////////
//
// Wrapper classes for the `MasterService.RunStep` response message.
//
// The `RunStepResponse` message can contain potentially large tensor
// data as part of its `tensor` submessages. Here we provide specialized
// wrappers that avoid copying the tensor data wherever possible.
//
// See `RunStepResponse` in tensorflow/core/protobuf/master.proto for the
// protocol buffer definition.
//
////////////////////////////////////////////////////////////////////////////////

// Abstract interface for a mutable RunStepResponse message.
//
// Note that there is no corresponding (immutable)
// RunStepResponseWrapper class, because the RunStepResponse object is
// always used as a mutable pointer.
class MutableRunStepResponseWrapper {
 public:
  virtual ~MutableRunStepResponseWrapper();

  // The values of the tensors whose fetching was requested in the
  // RunStep call.
  //
  // NOTE: The order of the returned tensors may or may not match
  // the fetch order specified in RunStepRequest.
  virtual size_t num_tensors() const = 0;
  virtual const string& tensor_name(size_t i) const = 0;
  virtual Status TensorValue(size_t i, Tensor* out_tensor) const = 0;

  // Stores the i^{th} recv value in `run_graph_response` in this
  // response with the given `name`.
  virtual Status AddTensorFromRunGraphResponse(
      const string& name, MutableRunGraphResponseWrapper* run_graph_response,
      size_t i) = 0;

  // Returned metadata if requested in the options.
  virtual const RunMetadata& metadata() const = 0;
  virtual RunMetadata* mutable_metadata() = 0;

  // Returned status if requested.
  virtual errors::Code status_code() const = 0;
  virtual const string& status_error_message() const = 0;
  virtual void set_status(const Status& status) = 0;

 protected:
  // Returns a mutable protobuf message that represents the contents of
  // this wrapper, for passing to an RPC subsystem that will populate
  // the message.
  //
  // NOTE: Only `MasterInterface` subclasses may call this method. The
  // `InMemoryRunStepResponse` subclass does not implement this
  // method, and attempts to call it will fail with a fatal
  // error. However, as long as callers always call
  // `MasterInterface::RunStep()` with a wrapper object returned
  // from `MasterInterface::CreateRunStepResponse()` called on the
  // *same* MasterInterface object, this error will never trigger.
  virtual RunStepResponse* get_proto() = 0;
  friend class MasterInterface;
};

class InMemoryRunStepResponse : public MutableRunStepResponseWrapper {
 public:
  // MutableRunStepResponseWrapper methods.
  size_t num_tensors() const override;
  const string& tensor_name(size_t i) const override;
  Status TensorValue(size_t i, Tensor* out_tensor) const override;
  Status AddTensorFromRunGraphResponse(
      const string& name, MutableRunGraphResponseWrapper* run_graph_response,
      size_t i) override;
  const RunMetadata& metadata() const override;
  RunMetadata* mutable_metadata() override;
  errors::Code status_code() const override;
  const string& status_error_message() const override;
  void set_status(const Status& status) override;

 protected:
  // NOTE: This method is not implemented. See
  // MutableRunGraphResponseWrapper for an explanation.
  RunStepResponse* get_proto() override;

 private:
  gtl::InlinedVector<std::pair<string, Tensor>, 4> tensors_;
  RunMetadata metadata_;
  // Store the code and message separately so that they can be updated
  // independently by setters.
  Status status_;
};

// Proto-based message wrapper for use on the client side of the RunStep RPC.
class OwnedProtoRunStepResponse : public MutableRunStepResponseWrapper {
 public:
  // MutableRunStepResponseWrapper methods.
  size_t num_tensors() const override;
  const string& tensor_name(size_t i) const override;
  Status TensorValue(size_t i, Tensor* out_tensor) const override;
  Status AddTensorFromRunGraphResponse(
      const string& name, MutableRunGraphResponseWrapper* run_graph_response,
      size_t i) override;
  const RunMetadata& metadata() const override;
  RunMetadata* mutable_metadata() override;
  errors::Code status_code() const override;
  const string& status_error_message() const override;
  void set_status(const Status& status) override;

 protected:
  RunStepResponse* get_proto() override;

 private:
  RunStepResponse response_;
};

// Proto-based message wrapper for use on the server side of the RunStep RPC.
class NonOwnedProtoRunStepResponse : public MutableRunStepResponseWrapper {
 public:
  NonOwnedProtoRunStepResponse(RunStepResponse* response);

  // MutableRunStepResponseWrapper methods.
  size_t num_tensors() const override;
  const string& tensor_name(size_t i) const override;
  Status TensorValue(size_t i, Tensor* out_tensor) const override;
  Status AddTensorFromRunGraphResponse(
      const string& name, MutableRunGraphResponseWrapper* run_graph_response,
      size_t i) override;
  const RunMetadata& metadata() const override;
  RunMetadata* mutable_metadata() override;
  errors::Code status_code() const override;
  const string& status_error_message() const override;
  void set_status(const Status& status) override;

 protected:
  RunStepResponse* get_proto() override;

 private:
  RunStepResponse* response_;  // Not owned.
};

}  // namespace tensorflow

#endif  // TENSORFLOW
