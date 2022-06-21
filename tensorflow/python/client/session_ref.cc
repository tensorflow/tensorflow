/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/python/client/session_ref.h"

#include <stdlib.h>
#include <memory>
#include <utility>

#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/io/record_writer.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/protobuf/master.pb.h"
#include "tensorflow/core/protobuf/named_tensor.pb.h"
#include "tensorflow/core/protobuf/replay_log.pb.h"

namespace tensorflow {

namespace {

// Scope helper to track active calls and manage session lifetime.
// SessionRef blocks closing until all active calls complete or are cancelled.
struct RunCounter {
  std::shared_ptr<Session> session;
  uint64* value;
  mutex* m;
  condition_variable* cv;

  explicit RunCounter(std::shared_ptr<Session> s, uint64* v, mutex* m,
                      condition_variable* cv)
      : session(std::move(s)), value(v), m(m), cv(cv) {
    mutex_lock l(*m);
    ++*value;
  }

  ~RunCounter() {
    mutex_lock l(*m);
    if (--*value == 0) {
      cv->notify_all();
    }
  }
};

std::string SessionToHandle(Session* session) {
  return strings::Printf("%llu", static_cast<unsigned long long>(
                                     reinterpret_cast<uintptr_t>(session)));
}

// The Session interface has many methods of the form:
//
// X(a, b);
// X(RunOptions, a, b);
//
// Not all sessions support the second case (with an empty RunOptions()).
// We use this variable as a sentinel to dispatch to the correct call.
RunOptions* kEmptyRunOptions() {
  static RunOptions* options = new RunOptions();
  return options;
}

}  // namespace

// Run the given session operation, recording start and end timestamps.
// If the operation returns a bad status, return after flushing the current
// log request.  This should be run _after_ all request information has been
// added to the current op.
#define RUN_WITH_TIMESTAMP(OpName, ...)              \
  op.set_start_time_us(Env::Default()->NowMicros()); \
  Status status = session->OpName(__VA_ARGS__);      \
  op.set_end_time_us(Env::Default()->NowMicros());   \
  if (!status.ok()) {                                \
    Flush(op).IgnoreError();                         \
    return status;                                   \
  }

// Records requests (and optionally responses) performed against a session.
// The resulting replay log can be used with the `tf_replay` tool to replicate
// the operations against a simulated environment, without requiring the
// original code or cluster setup.
//
// Session logging by setting the TF_REPLAY_LOG_FILE environment variable.
class SessionLogger {
 public:
  SessionLogger() {
    const char* log_file_env = getenv("TF_REPLAY_LOG_FILE");
    std::string log_name = log_file_env ? std::string(log_file_env) : ".";
    LOG(INFO) << "Constructing new session logger for " << log_name;
    TF_CHECK_OK(
        Env::Default()->RecursivelyCreateDir(string(io::Dirname(log_name))));
    Env::Default()->DeleteFile(log_name).IgnoreError();

    TF_CHECK_OK(Env::Default()->NewWritableFile(log_name, &log_file_));
    log_writer_ = absl::make_unique<io::RecordWriter>(log_file_.get());
  }

  ~SessionLogger() {
    log_writer_->Close().IgnoreError();
    log_writer_.release();
    log_file_->Close().IgnoreError();
  }

  Status RecordNewSession(Session* session) {
    ReplayOp op;
    NewReplaySession* req = op.mutable_new_replay_session();
    req->set_session_handle(SessionToHandle(session));
    return Flush(op);
  }

  Status RecordRun(Session* session,
                   const std::vector<std::pair<string, Tensor> >& inputs,
                   const std::vector<string>& output_tensor_names,
                   const std::vector<string>& target_node_names,
                   std::vector<Tensor>* outputs) {
    return RecordRun(session, *kEmptyRunOptions(), inputs, output_tensor_names,
                     target_node_names, outputs, nullptr);
  }

  Status RecordRun(Session* session, const RunOptions& run_options,
                   const std::vector<std::pair<string, Tensor> >& inputs,
                   const std::vector<string>& output_tensor_names,
                   const std::vector<string>& target_node_names,
                   std::vector<Tensor>* outputs, RunMetadata* run_metadata) {
    ReplayOp op;
    RunStepRequest* req = op.mutable_run_step();
    RunStepResponse* resp = op.mutable_run_step_response();

    req->set_session_handle(SessionToHandle(session));
    *req->mutable_options() = run_options;

    for (const auto& it : inputs) {
      NamedTensorProto* feed = req->add_feed();
      feed->set_name(it.first);
      it.second.AsProtoField(feed->mutable_tensor());
    }

    // Build an index from fetch tensor name to first index in
    // output_tensor_names.
    std::unordered_map<string, int> output_name_to_offset;
    for (int i = 0, end = output_tensor_names.size(); i < end; ++i) {
      const string& name = output_tensor_names[i];
      if (output_name_to_offset.insert(std::make_pair(name, i)).second) {
        req->add_fetch(name);
      }
    }
    for (const string& target : target_node_names) {
      req->add_target(target);
    }

    if (&run_options == kEmptyRunOptions()) {
      RUN_WITH_TIMESTAMP(Run, inputs, output_tensor_names, target_node_names,
                         outputs);
    } else {
      RUN_WITH_TIMESTAMP(Run, run_options, inputs, output_tensor_names,
                         target_node_names, outputs, run_metadata);
    }

    for (size_t i = 0; i < outputs->size(); ++i) {
      const Tensor& tensor = (*outputs)[i];
      NamedTensorProto* tproto = resp->add_tensor();
      tensor.AsProtoField(tproto->mutable_tensor());
      tproto->set_name(output_tensor_names[i]);
    }

    if (run_metadata) {
      *resp->mutable_metadata() = *run_metadata;
    }

    return Flush(op);
  }

  Status RecordCreate(Session* session, const GraphDef& graph) {
    return RecordCreate(session, *kEmptyRunOptions(), graph);
  }

  // N.B. RunOptions is not stored (it has no entry in CreateRequest)
  Status RecordCreate(Session* session, const RunOptions& run_options,
                      const GraphDef& graph) {
    ReplayOp op;
    CreateSessionRequest* req = op.mutable_create_session();
    *req->mutable_graph_def() = graph;

    CreateSessionResponse* resp = op.mutable_create_session_response();
    if (&run_options == kEmptyRunOptions()) {
      RUN_WITH_TIMESTAMP(Create, graph);
    } else {
      RUN_WITH_TIMESTAMP(Create, run_options, graph);
    }
    resp->set_session_handle(SessionToHandle(session));
    return Flush(op);
  }

  Status RecordExtend(Session* session, const GraphDef& graph) {
    return RecordExtend(session, *kEmptyRunOptions(), graph);
  }

  // N.B. RunOptions is not stored (it has no entry in ExtendRequest)
  Status RecordExtend(Session* session, const RunOptions& run_options,
                      const GraphDef& graph) {
    ReplayOp op;
    ExtendSessionRequest* req = op.mutable_extend_session();
    op.mutable_extend_session_response();
    req->set_session_handle(SessionToHandle(session));
    *req->mutable_graph_def() = graph;
    if (&run_options == kEmptyRunOptions()) {
      RUN_WITH_TIMESTAMP(Extend, graph);
    } else {
      RUN_WITH_TIMESTAMP(Extend, run_options, graph);
    }

    return Flush(op);
  }

  Status RecordClose(Session* session) {
    return RecordClose(session, *kEmptyRunOptions());
  }

  // N.B. RunOptions is not stored (it has no entry in CloseRequest)
  Status RecordClose(Session* session, const RunOptions& run_options) {
    ReplayOp op;
    CloseSessionRequest* req = op.mutable_close_session();
    req->set_session_handle(SessionToHandle(session));
    op.mutable_close_session_response();
    if (&run_options == kEmptyRunOptions()) {
      RUN_WITH_TIMESTAMP(Close);
    } else {
      RUN_WITH_TIMESTAMP(Close, run_options);
    }
    return Flush(op);
  }

  Status RecordListDevices(Session* session,
                           std::vector<DeviceAttributes>* response) {
    ReplayOp op;
    ListDevicesRequest* req = op.mutable_list_devices();
    ListDevicesResponse* resp = op.mutable_list_devices_response();
    req->set_session_handle(SessionToHandle(session));
    RUN_WITH_TIMESTAMP(ListDevices, response);

    // TODO(power) -- local vs remote device distinction is lost here!
    *resp->mutable_local_device() = {response->begin(), response->end()};
    return Flush(op);
  }

  Status RecordPRunSetup(Session* session,
                         const std::vector<string>& input_names,
                         const std::vector<string>& output_names,
                         const std::vector<string>& target_nodes,
                         string* handle) {
    ReplayOp op;
    PartialRunSetupRequest* req = op.mutable_partial_run_setup();
    req->set_session_handle(SessionToHandle(session));
    for (auto& input : input_names) {
      req->add_feed(input);
    }
    for (auto& output : output_names) {
      req->add_fetch(output);
    }
    for (auto& target : target_nodes) {
      req->add_target(target);
    }
    RUN_WITH_TIMESTAMP(PRunSetup, input_names, output_names, target_nodes,
                       handle);
    op.mutable_partial_run_setup_response()->set_partial_run_handle(*handle);
    return Flush(op);
  }

  Status RecordPRun(Session* session, const string& handle,
                    const std::vector<std::pair<string, Tensor> >& inputs,
                    const std::vector<string>& output_names,
                    std::vector<Tensor>* outputs) {
    ReplayOp op;
    RunStepRequest* req = op.mutable_run_step();
    RunStepResponse* resp = op.mutable_run_step_response();
    req->set_session_handle(SessionToHandle(session));

    // Mark this step as a partial run for replay.
    req->set_partial_run_handle(handle);
    for (auto& input : inputs) {
      auto* feed = req->add_feed();
      feed->set_name(input.first);
      input.second.AsProtoField(feed->mutable_tensor());
    }

    for (auto& output : output_names) {
      req->add_fetch(output);
    }

    RUN_WITH_TIMESTAMP(PRun, handle, inputs, output_names, outputs);

    for (size_t i = 0; i < outputs->size(); ++i) {
      const Tensor& tensor = (*outputs)[i];
      NamedTensorProto* tproto = resp->add_tensor();
      tensor.AsProtoField(tproto->mutable_tensor());
      tproto->set_name(output_names[i]);
    }

    return Flush(op);
  }

  Status RecordMakeCallable(Session* session,
                            const CallableOptions& callable_options,
                            Session::CallableHandle* handle) {
    ReplayOp op;
    MakeCallableRequest* req = op.mutable_make_callable();
    req->set_session_handle(SessionToHandle(session));
    *req->mutable_options() = callable_options;

    RUN_WITH_TIMESTAMP(MakeCallable, callable_options, handle);

    MakeCallableResponse* resp = op.mutable_make_callable_response();
    resp->set_handle(*handle);

    return Flush(op);
  }

  Status RecordRunCallable(Session* session, Session::CallableHandle handle,
                           const std::vector<Tensor>& feed_tensors,
                           std::vector<Tensor>* fetch_tensors,
                           RunMetadata* run_metadata) {
    ReplayOp op;
    RunCallableRequest* req = op.mutable_run_callable();
    req->set_session_handle(SessionToHandle(session));
    req->set_handle(handle);
    for (auto& tensor : feed_tensors) {
      tensor.AsProtoField(req->add_feed());
    }
    RUN_WITH_TIMESTAMP(RunCallable, handle, feed_tensors, fetch_tensors,
                       run_metadata);

    RunCallableResponse* resp = op.mutable_run_callable_response();
    if (run_metadata) {
      *resp->mutable_metadata() = *run_metadata;
    }
    for (const Tensor& tensor : *fetch_tensors) {
      tensor.AsProtoTensorContent(resp->add_fetch());
    }
    return Flush(op);
  }

  Status RecordReleaseCallable(Session* session,
                               Session::CallableHandle handle) {
    ReplayOp op;
    ReleaseCallableRequest* req = op.mutable_release_callable();
    req->set_session_handle(SessionToHandle(session));
    req->set_handle(handle);
    RUN_WITH_TIMESTAMP(ReleaseCallable, handle);
    return Flush(op);
  }

 private:
  Status Flush(const ReplayOp& op) {
    mutex_lock l(log_mutex_);

    string buf;
    op.SerializeToString(&buf);
    TF_RETURN_IF_ERROR(log_writer_->WriteRecord(buf));

    // TODO(b/116624106): Not all file-systems respect calls to `Sync()`
    return log_file_->Sync();
  }

  std::unique_ptr<WritableFile> log_file_;
  std::unique_ptr<io::RecordWriter> log_writer_;
  mutex log_mutex_;
};

static SessionLogger* global_session_logger() {
  static SessionLogger* logger = new SessionLogger();
  return logger;
}

SessionRef::SessionRef(Session* session) : session_(session) {
  if (getenv("TF_REPLAY_LOG_FILE") != nullptr) {
    logger_ = global_session_logger();
    logger_->RecordNewSession(this->session_.get()).IgnoreError();
  } else {
    logger_ = nullptr;
  }
}

SessionRef::~SessionRef() = default;

Status SessionRef::CheckNotClosed() {
  mutex_lock l(run_lock_);
  if (session_ == nullptr) return errors::Cancelled("Session has been closed.");
  return ::tensorflow::Status::OK();
}

// If logging is active, log the start and end time of the operation along with
// the request and response.
#define LOG_AND_RUN_OPERATION(OpName, ...)                          \
  TF_RETURN_IF_ERROR(CheckNotClosed());                             \
  RunCounter rc(session_, &run_count_, &run_lock_, &run_finished_); \
  if (!logger_) {                                                   \
    return rc.session->OpName(__VA_ARGS__);                         \
  }                                                                 \
  return logger_->Record##OpName(rc.session.get(), __VA_ARGS__);

Status SessionRef::Run(const RunOptions& run_options,
                       const std::vector<std::pair<string, Tensor> >& inputs,
                       const std::vector<string>& output_tensor_names,
                       const std::vector<string>& target_node_names,
                       std::vector<Tensor>* outputs,
                       RunMetadata* run_metadata) {
  LOG_AND_RUN_OPERATION(Run, run_options, inputs, output_tensor_names,
                        target_node_names, outputs, run_metadata);
}

Status SessionRef::Run(const std::vector<std::pair<string, Tensor> >& inputs,
                       const std::vector<string>& output_tensor_names,
                       const std::vector<string>& target_node_names,
                       std::vector<Tensor>* outputs) {
  LOG_AND_RUN_OPERATION(Run, inputs, output_tensor_names, target_node_names,
                        outputs);
}

Status SessionRef::Create(const GraphDef& graph) {
  LOG_AND_RUN_OPERATION(Create, graph);
}

Status SessionRef::Create(const RunOptions& run_options,
                          const GraphDef& graph) {
  LOG_AND_RUN_OPERATION(Create, run_options, graph);
}

Status SessionRef::Extend(const RunOptions& run_options,
                          const GraphDef& graph) {
  LOG_AND_RUN_OPERATION(Extend, run_options, graph);
}

Status SessionRef::Extend(const GraphDef& graph) {
  LOG_AND_RUN_OPERATION(Extend, graph);
}

Status SessionRef::ListDevices(std::vector<DeviceAttributes>* response) {
  LOG_AND_RUN_OPERATION(ListDevices, response);
}

Status SessionRef::PRunSetup(const std::vector<string>& input_names,
                             const std::vector<string>& output_names,
                             const std::vector<string>& target_nodes,
                             string* handle) {
  LOG_AND_RUN_OPERATION(PRunSetup, input_names, output_names, target_nodes,
                        handle);
}

Status SessionRef::PRun(const string& handle,
                        const std::vector<std::pair<string, Tensor> >& inputs,
                        const std::vector<string>& output_names,
                        std::vector<Tensor>* outputs) {
  LOG_AND_RUN_OPERATION(PRun, handle, inputs, output_names, outputs);
}

Status SessionRef::MakeCallable(const CallableOptions& callable_options,
                                CallableHandle* out_handle) {
  LOG_AND_RUN_OPERATION(MakeCallable, callable_options, out_handle);
}

Status SessionRef::RunCallable(CallableHandle handle,
                               const std::vector<Tensor>& feed_tensors,
                               std::vector<Tensor>* fetch_tensors,
                               RunMetadata* run_metadata) {
  LOG_AND_RUN_OPERATION(RunCallable, handle, feed_tensors, fetch_tensors,
                        run_metadata);
}

Status SessionRef::ReleaseCallable(CallableHandle handle) {
  {
    mutex_lock l(run_lock_);
    if (session_ == nullptr) {
      // Session already closed. Do nothing.
      return Status::OK();
    }
  }
  LOG_AND_RUN_OPERATION(ReleaseCallable, handle);
}

Status SessionRef::Close(const RunOptions& run_options) {
  TF_RETURN_IF_ERROR(CheckNotClosed());
  mutex_lock l(run_lock_);
  Status status;
  if (logger_) {
    status = logger_->RecordClose(session_.get(), run_options);
  } else {
    status = session_->Close(run_options);
  }
  session_.reset();
  while (run_count_ > 0) {
    run_finished_.wait(l);
  }
  return status;
}

Status SessionRef::Close() {
  TF_RETURN_IF_ERROR(CheckNotClosed());
  mutex_lock l(run_lock_);
  Status status;
  if (logger_) {
    status = logger_->RecordClose(session_.get());
  } else {
    status = session_->Close();
  }
  session_.reset();
  while (run_count_ > 0) {
    run_finished_.wait(l);
  }
  return status;
}

}  // namespace tensorflow
