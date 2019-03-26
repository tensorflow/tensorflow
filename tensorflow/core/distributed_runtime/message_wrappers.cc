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

#include "tensorflow/core/distributed_runtime/message_wrappers.h"
#include "tensorflow/core/framework/cost_graph.pb.h"
#include "tensorflow/core/framework/step_stats.pb.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/protobuf/named_tensor.pb.h"

namespace tensorflow {

namespace {

bool ParseTensorProtoToTensor(const TensorProto& tensor_proto,
                              Tensor* out_tensor) {
  if (tensor_proto.dtype() > 0 && tensor_proto.dtype() <= DataType_MAX) {
    Tensor parsed(tensor_proto.dtype());
    if (parsed.FromProto(cpu_allocator(), tensor_proto)) {
      *out_tensor = parsed;
      return true;
    }
  }
  return false;
}

}  // namespace

const string& InMemoryRunStepRequest::session_handle() const {
  return session_handle_;
}

void InMemoryRunStepRequest::set_session_handle(const string& handle) {
  session_handle_ = handle;
}

const string& InMemoryRunStepRequest::partial_run_handle() const {
  return partial_run_handle_;
}

void InMemoryRunStepRequest::set_partial_run_handle(const string& handle) {
  partial_run_handle_ = handle;
}

size_t InMemoryRunStepRequest::num_feeds() const { return feeds_.size(); }
const string& InMemoryRunStepRequest::feed_name(size_t i) const {
  return feeds_[i].first;
}

Status InMemoryRunStepRequest::FeedValue(size_t i, Tensor* out_tensor) const {
  *out_tensor = feeds_[i].second;
  return Status::OK();
}

Status InMemoryRunStepRequest::FeedValue(size_t i,
                                         TensorProto* out_tensor) const {
  feeds_[i].second.AsProtoTensorContent(out_tensor);
  return Status::OK();
}

void InMemoryRunStepRequest::add_feed(const string& name, const Tensor& value) {
  feeds_.emplace_back(name, value);
}

size_t InMemoryRunStepRequest::num_fetches() const { return fetches_.size(); }
const string& InMemoryRunStepRequest::fetch_name(size_t i) const {
  return fetches_[i];
}
void InMemoryRunStepRequest::add_fetch(const string& name) {
  fetches_.push_back(name);
}

size_t InMemoryRunStepRequest::num_targets() const { return targets_.size(); }
const string& InMemoryRunStepRequest::target_name(size_t i) const {
  return targets_[i];
}
void InMemoryRunStepRequest::add_target(const string& name) {
  targets_.push_back(name);
}

const RunOptions& InMemoryRunStepRequest::options() const { return options_; }

RunOptions* InMemoryRunStepRequest::mutable_options() { return &options_; }

bool InMemoryRunStepRequest::store_errors_in_response_body() const {
  return store_errors_in_response_body_;
}

int64 InMemoryRunStepRequest::request_id() const {
  return 0;  // no need to track request id for local version.
}

void InMemoryRunStepRequest::set_store_errors_in_response_body(
    bool store_errors) {
  store_errors_in_response_body_ = store_errors;
}

string InMemoryRunStepRequest::DebugString() const {
  return ToProto().DebugString();
}

const RunStepRequest& InMemoryRunStepRequest::ToProto() const {
  if (!proto_version_) {
    proto_version_.reset(new RunStepRequest);
    proto_version_->set_session_handle(session_handle());
    proto_version_->set_partial_run_handle(partial_run_handle());
    for (size_t i = 0; i < num_feeds(); ++i) {
      auto feed = proto_version_->add_feed();
      feed->set_name(feed_name(i));
      feeds_[i].second.AsProtoTensorContent(feed->mutable_tensor());
    }
    for (size_t i = 0; i < num_fetches(); ++i) {
      proto_version_->add_fetch(fetch_name(i));
    }
    for (size_t i = 0; i < num_targets(); ++i) {
      proto_version_->add_target(target_name(i));
    }
    *proto_version_->mutable_options() = options();
  }
  return *proto_version_;
}

const string& MutableProtoRunStepRequest::session_handle() const {
  return request_.session_handle();
}
void MutableProtoRunStepRequest::set_session_handle(const string& handle) {
  request_.set_session_handle(handle);
}

const string& MutableProtoRunStepRequest::partial_run_handle() const {
  return request_.partial_run_handle();
}
void MutableProtoRunStepRequest::set_partial_run_handle(const string& handle) {
  request_.set_partial_run_handle(handle);
}

size_t MutableProtoRunStepRequest::num_feeds() const {
  return request_.feed_size();
}
const string& MutableProtoRunStepRequest::feed_name(size_t i) const {
  return request_.feed(i).name();
}
Status MutableProtoRunStepRequest::FeedValue(size_t i,
                                             Tensor* out_tensor) const {
  if (!ParseTensorProtoToTensor(request_.feed(i).tensor(), out_tensor)) {
    return errors::InvalidArgument("Invalid TensorProto for feed value ", i);
  } else {
    return Status::OK();
  }
}

Status MutableProtoRunStepRequest::FeedValue(size_t i,
                                             TensorProto* out_tensor) const {
  *out_tensor = request_.feed(i).tensor();
  return Status::OK();
}

void MutableProtoRunStepRequest::add_feed(const string& name,
                                          const Tensor& value) {
  NamedTensorProto* feed = request_.add_feed();
  feed->set_name(name);
  TensorProto* value_proto = feed->mutable_tensor();
  value.AsProtoTensorContent(value_proto);
}

size_t MutableProtoRunStepRequest::num_fetches() const {
  return request_.fetch_size();
}

const string& MutableProtoRunStepRequest::fetch_name(size_t i) const {
  return request_.fetch(i);
}
void MutableProtoRunStepRequest::add_fetch(const string& name) {
  request_.add_fetch(name);
}

size_t MutableProtoRunStepRequest::num_targets() const {
  return request_.target_size();
}

const string& MutableProtoRunStepRequest::target_name(size_t i) const {
  return request_.target(i);
}

void MutableProtoRunStepRequest::add_target(const string& name) {
  request_.add_target(name);
}

const RunOptions& MutableProtoRunStepRequest::options() const {
  return request_.options();
}

RunOptions* MutableProtoRunStepRequest::mutable_options() {
  return request_.mutable_options();
}

bool MutableProtoRunStepRequest::store_errors_in_response_body() const {
  return request_.store_errors_in_response_body();
}

void MutableProtoRunStepRequest::set_store_errors_in_response_body(
    bool store_errors) {
  request_.set_store_errors_in_response_body(store_errors);
}

int64 MutableProtoRunStepRequest::request_id() const {
  return request_.request_id();
}

string MutableProtoRunStepRequest::DebugString() const {
  return request_.DebugString();
}

const RunStepRequest& MutableProtoRunStepRequest::ToProto() const {
  return request_;
}

ProtoRunStepRequest::ProtoRunStepRequest(const RunStepRequest* request)
    : request_(request) {}

const string& ProtoRunStepRequest::session_handle() const {
  return request_->session_handle();
}

const string& ProtoRunStepRequest::partial_run_handle() const {
  return request_->partial_run_handle();
}

size_t ProtoRunStepRequest::num_feeds() const { return request_->feed_size(); }

const string& ProtoRunStepRequest::feed_name(size_t i) const {
  return request_->feed(i).name();
}

Status ProtoRunStepRequest::FeedValue(size_t i, Tensor* out_tensor) const {
  if (!ParseTensorProtoToTensor(request_->feed(i).tensor(), out_tensor)) {
    return errors::InvalidArgument("Invalid TensorProto for feed value ", i);
  } else {
    return Status::OK();
  }
}

Status ProtoRunStepRequest::FeedValue(size_t i, TensorProto* out_tensor) const {
  *out_tensor = request_->feed(i).tensor();
  return Status::OK();
}

size_t ProtoRunStepRequest::num_fetches() const {
  return request_->fetch_size();
}

const string& ProtoRunStepRequest::fetch_name(size_t i) const {
  return request_->fetch(i);
}

size_t ProtoRunStepRequest::num_targets() const {
  return request_->target_size();
}

const string& ProtoRunStepRequest::target_name(size_t i) const {
  return request_->target(i);
}

const RunOptions& ProtoRunStepRequest::options() const {
  return request_->options();
}

bool ProtoRunStepRequest::store_errors_in_response_body() const {
  return request_->store_errors_in_response_body();
}

int64 ProtoRunStepRequest::request_id() const { return request_->request_id(); }

string ProtoRunStepRequest::DebugString() const {
  return request_->DebugString();
}

const RunStepRequest& ProtoRunStepRequest::ToProto() const { return *request_; }

const string& InMemoryRunGraphRequest::session_handle() const {
  return session_handle_;
}

bool InMemoryRunGraphRequest::create_worker_session_called() const {
  return create_worker_session_called_;
}

void InMemoryRunGraphRequest::set_session_handle(const string& handle) {
  session_handle_ = handle;
}

void InMemoryRunGraphRequest::set_create_worker_session_called(bool called) {
  create_worker_session_called_ = called;
}

const string& InMemoryRunGraphRequest::graph_handle() const {
  return graph_handle_;
}

void InMemoryRunGraphRequest::set_graph_handle(const string& handle) {
  graph_handle_ = handle;
}

int64 InMemoryRunGraphRequest::step_id() const { return step_id_; }

void InMemoryRunGraphRequest::set_step_id(int64 step_id) { step_id_ = step_id; }

const ExecutorOpts& InMemoryRunGraphRequest::exec_opts() const {
  return exec_opts_;
}

ExecutorOpts* InMemoryRunGraphRequest::mutable_exec_opts() {
  return &exec_opts_;
}

size_t InMemoryRunGraphRequest::num_sends() const { return sends_.size(); }

const string& InMemoryRunGraphRequest::send_key(size_t i) const {
  return sends_[i].first;
}

Status InMemoryRunGraphRequest::SendValue(size_t i, Tensor* out_tensor) const {
  *out_tensor = sends_[i].second;
  return Status::OK();
}

Status InMemoryRunGraphRequest::AddSendFromRunStepRequest(
    const RunStepRequestWrapper& run_step_request, size_t i,
    const string& send_key) {
  Tensor tensor;
  TF_RETURN_IF_ERROR(run_step_request.FeedValue(i, &tensor));
  sends_.emplace_back(send_key, std::move(tensor));
  return Status::OK();
}

// TODO(b/74355905): Add a specialized implementation that avoids
// copying the tensor when at least two of the {client, master,
// worker} are in the same process.
Status InMemoryRunGraphRequest::AddSendFromRunCallableRequest(
    const RunCallableRequest& run_callable_request, size_t i,
    const string& send_key) {
  Tensor tensor;
  if (!ParseTensorProtoToTensor(run_callable_request.feed(i), &tensor)) {
    return errors::InvalidArgument("Invalid TensorProto for feed value ", i);
  }
  sends_.emplace_back(send_key, std::move(tensor));
  return Status::OK();
}

size_t InMemoryRunGraphRequest::num_recvs() const { return recvs_.size(); }

const string& InMemoryRunGraphRequest::recv_key(size_t i) const {
  return recvs_[i];
}

void InMemoryRunGraphRequest::add_recv_key(const string& recv_key) {
  recvs_.push_back(recv_key);
}

bool InMemoryRunGraphRequest::is_partial() const { return is_partial_; }

void InMemoryRunGraphRequest::set_is_partial(bool is_partial) {
  is_partial_ = is_partial;
}

bool InMemoryRunGraphRequest::is_last_partial_run() const {
  return is_last_partial_run_;
}

void InMemoryRunGraphRequest::set_is_last_partial_run(
    bool is_last_partial_run) {
  is_last_partial_run_ = is_last_partial_run;
}

bool InMemoryRunGraphRequest::store_errors_in_response_body() const {
  return store_errors_in_response_body_;
}

void InMemoryRunGraphRequest::set_store_errors_in_response_body(
    bool store_errors) {
  store_errors_in_response_body_ = store_errors;
}

int64 InMemoryRunGraphRequest::request_id() const { return request_id_; }

void InMemoryRunGraphRequest::set_request_id(int64 request_id) {
  request_id_ = request_id;
}

const RunGraphRequest& InMemoryRunGraphRequest::ToProto() const {
  if (!proto_version_) {
    proto_version_.reset(new RunGraphRequest);
    proto_version_->set_session_handle(session_handle());
    proto_version_->set_create_worker_session_called(
        create_worker_session_called());
    proto_version_->set_graph_handle(graph_handle());
    proto_version_->set_step_id(step_id());
    *proto_version_->mutable_exec_opts() = exec_opts();
    for (size_t i = 0; i < num_sends(); ++i) {
      auto send = proto_version_->add_send();
      send->set_name(send_key(i));
      sends_[i].second.AsProtoTensorContent(send->mutable_tensor());
    }
    for (size_t i = 0; i < num_recvs(); ++i) {
      proto_version_->add_recv_key(recv_key(i));
    }
    proto_version_->set_is_partial(is_partial());
    proto_version_->set_is_last_partial_run(is_last_partial_run());
  }
  proto_version_->set_store_errors_in_response_body(
      store_errors_in_response_body_);
  proto_version_->set_request_id(request_id_);
  return *proto_version_;
}

const string& MutableProtoRunGraphRequest::session_handle() const {
  return request_.session_handle();
}

void MutableProtoRunGraphRequest::set_session_handle(const string& handle) {
  request_.set_session_handle(handle);
}

bool MutableProtoRunGraphRequest::create_worker_session_called() const {
  return request_.create_worker_session_called();
}

void MutableProtoRunGraphRequest::set_create_worker_session_called(
    bool called) {
  request_.set_create_worker_session_called(called);
}

const string& MutableProtoRunGraphRequest::graph_handle() const {
  return request_.graph_handle();
}

void MutableProtoRunGraphRequest::set_graph_handle(const string& handle) {
  request_.set_graph_handle(handle);
}

int64 MutableProtoRunGraphRequest::step_id() const {
  return request_.step_id();
}

void MutableProtoRunGraphRequest::set_step_id(int64 step_id) {
  request_.set_step_id(step_id);
}

const ExecutorOpts& MutableProtoRunGraphRequest::exec_opts() const {
  return request_.exec_opts();
}

ExecutorOpts* MutableProtoRunGraphRequest::mutable_exec_opts() {
  return request_.mutable_exec_opts();
}

size_t MutableProtoRunGraphRequest::num_sends() const {
  return request_.send_size();
}

const string& MutableProtoRunGraphRequest::send_key(size_t i) const {
  return request_.send(i).name();
}

Status MutableProtoRunGraphRequest::SendValue(size_t i,
                                              Tensor* out_tensor) const {
  if (!ParseTensorProtoToTensor(request_.send(i).tensor(), out_tensor)) {
    return errors::InvalidArgument("Invalid TensorProto for feed value ", i);
  } else {
    return Status::OK();
  }
}

Status MutableProtoRunGraphRequest::AddSendFromRunStepRequest(
    const RunStepRequestWrapper& run_step_request, size_t i,
    const string& send_key) {
  NamedTensorProto* send = request_.add_send();
  send->set_name(send_key);
  TF_RETURN_IF_ERROR(run_step_request.FeedValue(i, send->mutable_tensor()));
  return Status::OK();
}

// TODO(b/74355905): Add a specialized implementation that avoids
// copying the tensor when at least two of the {client, master,
// worker} are in the same process.
Status MutableProtoRunGraphRequest::AddSendFromRunCallableRequest(
    const RunCallableRequest& run_callable_request, size_t i,
    const string& send_key) {
  NamedTensorProto* send = request_.add_send();
  send->set_name(send_key);
  *send->mutable_tensor() = run_callable_request.feed(i);
  return Status::OK();
}

size_t MutableProtoRunGraphRequest::num_recvs() const {
  return request_.recv_key_size();
}

const string& MutableProtoRunGraphRequest::recv_key(size_t i) const {
  return request_.recv_key(i);
}

void MutableProtoRunGraphRequest::add_recv_key(const string& recv_key) {
  request_.add_recv_key(recv_key);
}

bool MutableProtoRunGraphRequest::is_partial() const {
  return request_.is_partial();
}

void MutableProtoRunGraphRequest::set_is_partial(bool is_partial) {
  request_.set_is_partial(is_partial);
}

bool MutableProtoRunGraphRequest::is_last_partial_run() const {
  return request_.is_last_partial_run();
}

void MutableProtoRunGraphRequest::set_is_last_partial_run(
    bool is_last_partial_run) {
  request_.set_is_last_partial_run(is_last_partial_run);
}

bool MutableProtoRunGraphRequest::store_errors_in_response_body() const {
  return request_.store_errors_in_response_body();
}

void MutableProtoRunGraphRequest::set_store_errors_in_response_body(
    bool store_errors) {
  request_.set_store_errors_in_response_body(store_errors);
}

int64 MutableProtoRunGraphRequest::request_id() const {
  return request_.request_id();
}

void MutableProtoRunGraphRequest::set_request_id(int64 request_id) {
  request_.set_request_id(request_id);
}

const RunGraphRequest& MutableProtoRunGraphRequest::ToProto() const {
  return request_;
}

ProtoRunGraphRequest::ProtoRunGraphRequest(const RunGraphRequest* request)
    : request_(request) {}

const string& ProtoRunGraphRequest::session_handle() const {
  return request_->session_handle();
}

bool ProtoRunGraphRequest::create_worker_session_called() const {
  return request_->create_worker_session_called();
}

const string& ProtoRunGraphRequest::graph_handle() const {
  return request_->graph_handle();
}

int64 ProtoRunGraphRequest::step_id() const { return request_->step_id(); }

const ExecutorOpts& ProtoRunGraphRequest::exec_opts() const {
  return request_->exec_opts();
}

size_t ProtoRunGraphRequest::num_sends() const { return request_->send_size(); }

const string& ProtoRunGraphRequest::send_key(size_t i) const {
  return request_->send(i).name();
}

Status ProtoRunGraphRequest::SendValue(size_t i, Tensor* out_tensor) const {
  if (!ParseTensorProtoToTensor(request_->send(i).tensor(), out_tensor)) {
    return errors::InvalidArgument("Invalid TensorProto for feed value ", i);
  } else {
    return Status::OK();
  }
}

size_t ProtoRunGraphRequest::num_recvs() const {
  return request_->recv_key_size();
}

const string& ProtoRunGraphRequest::recv_key(size_t i) const {
  return request_->recv_key(i);
}

bool ProtoRunGraphRequest::is_partial() const { return request_->is_partial(); }

bool ProtoRunGraphRequest::is_last_partial_run() const {
  return request_->is_last_partial_run();
}

bool ProtoRunGraphRequest::store_errors_in_response_body() const {
  return request_->store_errors_in_response_body();
}

int64 ProtoRunGraphRequest::request_id() const {
  return request_->request_id();
}

const RunGraphRequest& ProtoRunGraphRequest::ToProto() const {
  return *request_;
}

size_t InMemoryRunGraphResponse::num_recvs() const { return recvs_.size(); }

const string& InMemoryRunGraphResponse::recv_key(size_t i) const {
  return recvs_[i].first;
}

Status InMemoryRunGraphResponse::RecvValue(size_t i, TensorProto* out_tensor) {
  recvs_[i].second.AsProtoTensorContent(out_tensor);
  return Status::OK();
}

Status InMemoryRunGraphResponse::RecvValue(size_t i, Tensor* out_tensor) {
  *out_tensor = recvs_[i].second;
  return Status::OK();
}

void InMemoryRunGraphResponse::AddRecv(const string& key, const Tensor& value) {
  recvs_.emplace_back(key, value);
}

StepStats* InMemoryRunGraphResponse::mutable_step_stats() {
  return &step_stats_;
}

CostGraphDef* InMemoryRunGraphResponse::mutable_cost_graph() {
  return &cost_graph_;
}

errors::Code InMemoryRunGraphResponse::status_code() const {
  return status_.code();
}

const string& InMemoryRunGraphResponse::status_error_message() const {
  return status_.error_message();
}

void InMemoryRunGraphResponse::set_status(const Status& status) {
  status_ = status;
}

RunGraphResponse* InMemoryRunGraphResponse::get_proto() {
  LOG(FATAL) << "Cannot get a mutable protobuf for an InMemoryRunGraphResponse";
  return nullptr;
}

size_t InMemoryRunGraphResponse::num_partition_graphs() const {
  return partition_graphs_.size();
}

GraphDef* InMemoryRunGraphResponse::mutable_partition_graph(size_t i) {
  return &partition_graphs_[i];
}

void InMemoryRunGraphResponse::AddPartitionGraph(
    const GraphDef& partition_graph) {
  partition_graphs_.push_back(partition_graph);
}

size_t OwnedProtoRunGraphResponse::num_recvs() const {
  return response_.recv_size();
}

const string& OwnedProtoRunGraphResponse::recv_key(size_t i) const {
  return response_.recv(i).name();
}

Status OwnedProtoRunGraphResponse::RecvValue(size_t i,
                                             TensorProto* out_tensor) {
  out_tensor->Swap(response_.mutable_recv(i)->mutable_tensor());
  return Status::OK();
}

Status OwnedProtoRunGraphResponse::RecvValue(size_t i, Tensor* out_tensor) {
  if (!ParseTensorProtoToTensor(response_.recv(i).tensor(), out_tensor)) {
    return errors::InvalidArgument("Invalid TensorProto for recv value ", i);
  } else {
    return Status::OK();
  }
}

void OwnedProtoRunGraphResponse::AddRecv(const string& key,
                                         const Tensor& value) {
  NamedTensorProto* recv = response_.add_recv();
  recv->set_name(key);
  TensorProto* value_proto = recv->mutable_tensor();
  value.AsProtoTensorContent(value_proto);
}

StepStats* OwnedProtoRunGraphResponse::mutable_step_stats() {
  return response_.mutable_step_stats();
}

CostGraphDef* OwnedProtoRunGraphResponse::mutable_cost_graph() {
  return response_.mutable_cost_graph();
}

errors::Code OwnedProtoRunGraphResponse::status_code() const {
  return response_.status_code();
}

const string& OwnedProtoRunGraphResponse::status_error_message() const {
  return response_.status_error_message();
}

void OwnedProtoRunGraphResponse::set_status(const Status& status) {
  response_.set_status_code(status.code());
  response_.set_status_error_message(status.error_message());
}

RunGraphResponse* OwnedProtoRunGraphResponse::get_proto() { return &response_; }

size_t OwnedProtoRunGraphResponse::num_partition_graphs() const {
  return response_.partition_graph_size();
}

GraphDef* OwnedProtoRunGraphResponse::mutable_partition_graph(size_t i) {
  return response_.mutable_partition_graph(i);
}

void OwnedProtoRunGraphResponse::AddPartitionGraph(
    const GraphDef& partition_graph) {
  GraphDef* graph_def = response_.mutable_partition_graph()->Add();
  *graph_def = partition_graph;
}

NonOwnedProtoRunGraphResponse::NonOwnedProtoRunGraphResponse(
    RunGraphResponse* response)
    : response_(response) {}

size_t NonOwnedProtoRunGraphResponse::num_recvs() const {
  return response_->recv_size();
}

const string& NonOwnedProtoRunGraphResponse::recv_key(size_t i) const {
  return response_->recv(i).name();
}

Status NonOwnedProtoRunGraphResponse::RecvValue(size_t i,
                                                TensorProto* out_tensor) {
  out_tensor->Swap(response_->mutable_recv(i)->mutable_tensor());
  return Status::OK();
}

Status NonOwnedProtoRunGraphResponse::RecvValue(size_t i, Tensor* out_tensor) {
  if (!ParseTensorProtoToTensor(response_->recv(i).tensor(), out_tensor)) {
    return errors::InvalidArgument("Invalid TensorProto for recv value ", i);
  } else {
    return Status::OK();
  }
}

void NonOwnedProtoRunGraphResponse::AddRecv(const string& key,
                                            const Tensor& value) {
  NamedTensorProto* recv = response_->add_recv();
  recv->set_name(key);
  TensorProto* value_proto = recv->mutable_tensor();
  value.AsProtoTensorContent(value_proto);
}

StepStats* NonOwnedProtoRunGraphResponse::mutable_step_stats() {
  return response_->mutable_step_stats();
}

CostGraphDef* NonOwnedProtoRunGraphResponse::mutable_cost_graph() {
  return response_->mutable_cost_graph();
}

errors::Code NonOwnedProtoRunGraphResponse::status_code() const {
  return response_->status_code();
}

const string& NonOwnedProtoRunGraphResponse::status_error_message() const {
  return response_->status_error_message();
}

void NonOwnedProtoRunGraphResponse::set_status(const Status& status) {
  response_->set_status_code(status.code());
  response_->set_status_error_message(status.error_message());
}

RunGraphResponse* NonOwnedProtoRunGraphResponse::get_proto() {
  return response_;
}

size_t NonOwnedProtoRunGraphResponse::num_partition_graphs() const {
  return response_->partition_graph_size();
}

GraphDef* NonOwnedProtoRunGraphResponse::mutable_partition_graph(size_t i) {
  return response_->mutable_partition_graph(i);
}

void NonOwnedProtoRunGraphResponse::AddPartitionGraph(
    const GraphDef& partition_graph) {
  GraphDef* graph_def = response_->add_partition_graph();
  *graph_def = partition_graph;
}

MutableRunStepResponseWrapper::~MutableRunStepResponseWrapper() {}

size_t InMemoryRunStepResponse::num_tensors() const { return tensors_.size(); }

const string& InMemoryRunStepResponse::tensor_name(size_t i) const {
  return tensors_[i].first;
}

Status InMemoryRunStepResponse::TensorValue(size_t i,
                                            Tensor* out_tensor) const {
  *out_tensor = tensors_[i].second;
  return Status::OK();
}

const RunMetadata& InMemoryRunStepResponse::metadata() const {
  return metadata_;
}

Status InMemoryRunStepResponse::AddTensorFromRunGraphResponse(
    const string& name, MutableRunGraphResponseWrapper* wrapper, size_t i) {
  Tensor tensor;
  TF_RETURN_IF_ERROR(wrapper->RecvValue(i, &tensor));
  tensors_.emplace_back(name, tensor);
  return Status::OK();
}

RunMetadata* InMemoryRunStepResponse::mutable_metadata() { return &metadata_; }

errors::Code InMemoryRunStepResponse::status_code() const {
  return status_.code();
}

const string& InMemoryRunStepResponse::status_error_message() const {
  return status_.error_message();
}

void InMemoryRunStepResponse::set_status(const Status& status) {
  status_ = status;
}

RunStepResponse* InMemoryRunStepResponse::get_proto() {
  LOG(FATAL) << "Cannot get a mutable protobuf for an InMemoryRunStepResponse";
  return nullptr;
}

size_t OwnedProtoRunStepResponse::num_tensors() const {
  return response_.tensor_size();
}

const string& OwnedProtoRunStepResponse::tensor_name(size_t i) const {
  return response_.tensor(i).name();
}

Status OwnedProtoRunStepResponse::TensorValue(size_t i,
                                              Tensor* out_tensor) const {
  if (!ParseTensorProtoToTensor(response_.tensor(i).tensor(), out_tensor)) {
    return errors::InvalidArgument("Invalid TensorProto for fetch value ", i);
  } else {
    return Status::OK();
  }
}

const RunMetadata& OwnedProtoRunStepResponse::metadata() const {
  return response_.metadata();
}

Status OwnedProtoRunStepResponse::AddTensorFromRunGraphResponse(
    const string& name, MutableRunGraphResponseWrapper* run_graph_response,
    size_t i) {
  NamedTensorProto* response_tensor = response_.add_tensor();
  response_tensor->set_name(name);
  return run_graph_response->RecvValue(i, response_tensor->mutable_tensor());
}

RunMetadata* OwnedProtoRunStepResponse::mutable_metadata() {
  return response_.mutable_metadata();
}

errors::Code OwnedProtoRunStepResponse::status_code() const {
  return response_.status_code();
}

const string& OwnedProtoRunStepResponse::status_error_message() const {
  return response_.status_error_message();
}

void OwnedProtoRunStepResponse::set_status(const Status& status) {
  response_.set_status_code(status.code());
  response_.set_status_error_message(status.error_message());
}

RunStepResponse* OwnedProtoRunStepResponse::get_proto() { return &response_; }

NonOwnedProtoRunStepResponse::NonOwnedProtoRunStepResponse(
    RunStepResponse* response)
    : response_(response) {}

size_t NonOwnedProtoRunStepResponse::num_tensors() const {
  return response_->tensor_size();
}

const string& NonOwnedProtoRunStepResponse::tensor_name(size_t i) const {
  return response_->tensor(i).name();
}

Status NonOwnedProtoRunStepResponse::TensorValue(size_t i,
                                                 Tensor* out_tensor) const {
  if (!ParseTensorProtoToTensor(response_->tensor(i).tensor(), out_tensor)) {
    return errors::InvalidArgument("Invalid TensorProto for fetch value ", i);
  } else {
    return Status::OK();
  }
}

const RunMetadata& NonOwnedProtoRunStepResponse::metadata() const {
  return response_->metadata();
}

Status NonOwnedProtoRunStepResponse::AddTensorFromRunGraphResponse(
    const string& name, MutableRunGraphResponseWrapper* run_graph_response,
    size_t i) {
  NamedTensorProto* response_tensor = response_->add_tensor();
  response_tensor->set_name(name);
  return run_graph_response->RecvValue(i, response_tensor->mutable_tensor());
}

RunMetadata* NonOwnedProtoRunStepResponse::mutable_metadata() {
  return response_->mutable_metadata();
}

errors::Code NonOwnedProtoRunStepResponse::status_code() const {
  return response_->status_code();
}

const string& NonOwnedProtoRunStepResponse::status_error_message() const {
  return response_->status_error_message();
}

void NonOwnedProtoRunStepResponse::set_status(const Status& status) {
  response_->set_status_code(status.code());
  response_->set_status_error_message(status.error_message());
}

RunStepResponse* NonOwnedProtoRunStepResponse::get_proto() { return response_; }

}  // namespace tensorflow
