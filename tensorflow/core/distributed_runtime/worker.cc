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

#include "tensorflow/core/distributed_runtime/worker.h"

#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/process_util.h"
#include "tensorflow/core/common_runtime/step_stats_collector.h"
#include "tensorflow/core/distributed_runtime/rendezvous_mgr_interface.h"
#include "tensorflow/core/distributed_runtime/tensor_coding.h"
#include "tensorflow/core/distributed_runtime/worker_session.h"
#include "tensorflow/core/platform/tracing.h"

namespace tensorflow {

Worker::Worker(WorkerEnv* env)
    : env_(env), cancellation_manager_(new CancellationManager) {}

void Worker::GetStatusAsync(const GetStatusRequest* request,
                            GetStatusResponse* response, StatusCallback done) {
  DeviceMgr* dm = env_->device_mgr;
  std::vector<DeviceAttributes> devices;
  dm->ListDeviceAttributes(&devices);
  response->mutable_device_attributes()->Reserve(devices.size());
  for (size_t i = 0; i < devices.size(); ++i) {
    response->add_device_attributes()->Swap(&devices[i]);
  }
  done(Status::OK());
}

void Worker::CreateWorkerSessionAsync(const CreateWorkerSessionRequest* request,
                                      CreateWorkerSessionResponse* response,
                                      StatusCallback done) {
  Status s = env_->session_mgr->CreateSession(request->session_handle(),
                                              request->server_def());
  done(s);
}

void Worker::RegisterGraphAsync(const RegisterGraphRequest* request,
                                RegisterGraphResponse* response,
                                StatusCallback done) {
  WorkerSession* session =
      env_->session_mgr->WorkerSessionForSession(request->session_handle());
  Status s = session->graph_mgr->Register(
      request->session_handle(), request->graph_def(), request->graph_options(),
      response->mutable_graph_handle());
  if (s.ok()) {
    env_->session_mgr->AssociateGraphWithSession(request->session_handle(),
                                                 response->graph_handle());
  }
  done(s);
}

void Worker::DeregisterGraphAsync(const DeregisterGraphRequest* request,
                                  DeregisterGraphResponse* response,
                                  StatusCallback done) {
  WorkerSession* session =
      env_->session_mgr->WorkerSessionForGraphHandle(request->graph_handle());
  Status s = session->graph_mgr->Deregister(request->graph_handle());
  env_->session_mgr->DisassociateGraphFromSession(request->graph_handle());

  done(s);
}

Worker::PartialRunState* Worker::FindPartialRun(const string& graph_handle,
                                                int step_id) {
  const std::pair<string, int> k(graph_handle, step_id);
  Worker::PartialRunState* prun_state = nullptr;
  mutex_lock l(mu_);
  auto it = partial_runs_.find(k);
  if (it != partial_runs_.end()) {
    prun_state = it->second.get();
  }
  return prun_state;
}

void Worker::InsertPartialRunLocked(const string& graph_handle, int step_id,
                                    Worker::PartialRunState* partial_run_state)
    EXCLUSIVE_LOCKS_REQUIRED(mu_) {
  const std::pair<string, int> k(graph_handle, step_id);
  partial_runs_.emplace(std::make_pair(
      k, std::unique_ptr<Worker::PartialRunState>(partial_run_state)));
}

void Worker::RemovePartialRun(const string& graph_handle, int step_id) {
  const std::pair<string, int> k(graph_handle, step_id);
  mutex_lock l(mu_);
  partial_runs_.erase(partial_runs_.find(k));
}

void Worker::MaybeCallFinalCallback(const string& graph_handle, int step_id,
                                    const Status& executor_status) {
  const std::pair<string, int> k(graph_handle, step_id);
  StatusCallback done;
  Status s;
  {
    mutex_lock l(mu_);
    auto it = partial_runs_.find(k);
    if (it != partial_runs_.end()) {
      // If we found the partial_run, we call the final callback, if it
      // exists.
      std::swap(done, it->second->final_callback);
      s = it->second->final_status;
      it->second->executor_done = true;
    }
  }
  if (done != nullptr) {
    if (s.ok()) {
      s = executor_status;
    }
    done(s);
  }
}

void Worker::SetOrCallFinalCallback(const string& graph_handle, int step_id,
                                    StatusCallback done, const Status& s) {
  const std::pair<string, int> k(graph_handle, step_id);
  {
    mutex_lock l(mu_);
    auto it = partial_runs_.find(k);
    if (!it->second->executor_done) {
      // If we found the partial_run, we set the final callback to call only
      // when the executor is completely done.
      it->second->final_callback = std::move(done);
      it->second->final_status = s;
      return;
    }
  }
  // Otherwise we call the callback immediately.
  done(s);
}

void Worker::AbortStep(int64 step_id) {
  WorkerSession* session = env_->session_mgr->WorkerSessionForStepId(step_id);
  Rendezvous* rendez = session->rendezvous_mgr->Find(step_id);
  SchedNonBlockingClosureAfter(1000000, [rendez, step_id]() {
    // Delay a bit before aborting the step. This way, the root
    // cause may return first back to the client instead of this
    // cancellation generated abort error.
    rendez->StartAbort(errors::Aborted("Step ", step_id));
    rendez->Unref();
  });
}

Status Worker::PrepareRunGraph(RunGraphRequestWrapper* req,
                               GraphMgr::NamedTensors* in,
                               GraphMgr::NamedTensors* out) {
  static Tensor empty_tensor(DT_FLOAT);
  if (req->num_sends() > 0) {
    Tensor val;
    for (size_t i = 0; i < req->num_sends(); ++i) {
      TF_RETURN_IF_ERROR(req->SendValue(i, &val));
      in->insert({req->send_key(i), val});
    }
  }
  for (size_t i = 0; i < req->num_recvs(); ++i) {
    out->insert({req->recv_key(i), empty_tensor});
  }
  return Status::OK();
}

void Worker::RunGraphAsync(CallOptions* opts, RunGraphRequestWrapper* request,
                           MutableRunGraphResponseWrapper* response,
                           StatusCallback done) {
  if (request->is_partial()) {
    DoPartialRunGraph(opts, request, response, std::move(done));
  } else {
    DoRunGraph(opts, request, response, std::move(done));
  }
}

MutableRunGraphRequestWrapper* Worker::CreateRunGraphRequest() {
  return new InMemoryRunGraphRequest;
}

MutableRunGraphResponseWrapper* Worker::CreateRunGraphResponse() {
  return new InMemoryRunGraphResponse;
}

void Worker::DoRunGraph(CallOptions* opts, RunGraphRequestWrapper* request,
                        MutableRunGraphResponseWrapper* response,
                        StatusCallback done) {
  const int64 step_id = request->step_id();
  TRACEPRINTF("RunGraph: %lld", step_id);
  WorkerSession* session =
      env_->session_mgr->WorkerSessionForGraphHandle(request->graph_handle());
  env_->session_mgr->AssociateStepIdWithGraph(request->graph_handle(), step_id);
  GraphMgr::NamedTensors in;
  GraphMgr::NamedTensors* out = new GraphMgr::NamedTensors;
  Status s = PrepareRunGraph(request, &in, out);
  if (!s.ok()) {
    delete out;
    done(s);
    return;
  }
  StepStatsCollector* collector = nullptr;
  if (request->exec_opts().record_timeline() ||
      request->exec_opts().record_costs()) {
    collector = new StepStatsCollector(response->mutable_step_stats());
    // TODO(mrry,pbar): GPU tracing for distributed steps.
  }
  CancellationManager* cm = new CancellationManager;
  opts->SetCancelCallback([this, cm, step_id]() {
    cm->StartCancel();
    AbortStep(step_id);
  });
  CancellationToken token;
  {
    mutex_lock l(mu_);
    token = cancellation_manager_->get_cancellation_token();
    bool already_cancelled = !cancellation_manager_->RegisterCallback(
        token, [cm]() { cm->StartCancel(); });
    if (already_cancelled) {
      opts->ClearCancelCallback();
      delete cm;
      delete collector;
      delete out;
      done(errors::Aborted("Call was aborted"));
      return;
    }
  }
  CostGraphDef* cost_graph = response->mutable_cost_graph();
  session->graph_mgr->ExecuteAsync(
      request->graph_handle(), step_id, request->exec_opts(), collector,
      cost_graph, cm, in,
      [this, step_id, response, session, cm, out, token, collector, opts,
       done](Status s) {
        if (s.ok()) {
          s = session->graph_mgr->RecvOutputs(step_id, out);
        }
        opts->ClearCancelCallback();
        {
          mutex_lock l(mu_);
          cancellation_manager_->DeregisterCallback(token);
        }
        delete cm;

        if (s.ok()) {
          for (const auto& p : *out) {
            const string& key = p.first;
            const Tensor& val = p.second;
            response->AddRecv(key, val);
          }
        }
        delete collector;
        delete out;
        done(s);
      });
}

// TODO(suharshs): Add stats collection support to partial run.
void Worker::DoPartialRunGraph(CallOptions* opts,
                               RunGraphRequestWrapper* request,
                               MutableRunGraphResponseWrapper* response,
                               StatusCallback done) {
  const int64 step_id = request->step_id();
  const string& graph_handle = request->graph_handle();
  TRACEPRINTF("PartialRunGraph: %lld", step_id);
  WorkerSession* session =
      env_->session_mgr->WorkerSessionForGraphHandle(graph_handle);
  env_->session_mgr->AssociateStepIdWithGraph(graph_handle, step_id);
  GraphMgr::NamedTensors in;
  GraphMgr::NamedTensors* out = new GraphMgr::NamedTensors;
  Status s = PrepareRunGraph(request, &in, out);
  auto finish = [this, done, out, opts](const Status& s) {
    opts->ClearCancelCallback();
    delete out;
    done(s);
  };
  if (!s.ok()) {
    finish(s);
    return;
  }

  PartialRunState* partial_run_state = FindPartialRun(graph_handle, step_id);

  CancellationManager* cm = nullptr;
  // If this is a new partial run call we need to create a new cancellation
  // manager.
  // Otherwise we use the cancellation manager stored in the found partial
  // run state.
  if (partial_run_state == nullptr) {
    cm = new CancellationManager;
  } else {
    cm = partial_run_state->cancellation_manager;
  }

  // Before we start doing anything, we set the RPC cancellation.
  opts->SetCancelCallback([this, cm, step_id]() {
    cm->StartCancel();
    AbortStep(step_id);
  });

  // If this is a new partial run request, the request will need to start the
  // executors.
  if (partial_run_state == nullptr) {
    CancellationToken token;
    {
      mutex_lock l(mu_);
      // Insert the new partial run into the partial_runs_ map.
      partial_run_state = new PartialRunState(cm);
      InsertPartialRunLocked(graph_handle, step_id, partial_run_state);
      token = cancellation_manager_->get_cancellation_token();
      cancellation_manager_->RegisterCallback(token,
                                              [cm]() { cm->StartCancel(); });
    }
    session->graph_mgr->ExecuteAsync(
        graph_handle, step_id, request->exec_opts(), nullptr /* collector */,
        nullptr /* cost_graph */, cm, in,
        [this, token, graph_handle, step_id, cm](Status s) {
          {
            mutex_lock l(mu_);
            cancellation_manager_->DeregisterCallback(token);
          }
          MaybeCallFinalCallback(graph_handle, step_id, s);
          delete cm;
        });
  } else {
    // Send the partial run's new inputs.
    s = session->graph_mgr->SendInputs(step_id, in);
    if (!s.ok()) {
      finish(s);
      return;
    }
  }

  session->graph_mgr->RecvOutputsAsync(
      step_id, out,
      [this, out, request, response, graph_handle, step_id, finish](Status s) {
        if (s.ok()) {
          // Construct and return the resp.
          for (const auto& p : *out) {
            const string& key = p.first;
            const Tensor& val = p.second;
            response->AddRecv(key, val);
          }
        }
        if (request->is_last_partial_run()) {
          SetOrCallFinalCallback(
              graph_handle, step_id,
              [this, graph_handle, step_id, finish](const Status& s) {
                finish(s);
                // We must wait to remove the partial_run_state until both the
                // executor and the RecvAsync are complete.
                RemovePartialRun(graph_handle, step_id);
              },
              s);
        } else {
          finish(s);
        }
      });
}

void Worker::CleanupGraphAsync(const CleanupGraphRequest* request,
                               CleanupGraphResponse* response,
                               StatusCallback done) {
  const int64 step_id = request->step_id();
  WorkerSession* session = env_->session_mgr->WorkerSessionForStepId(step_id);
  session->rendezvous_mgr->Cleanup(step_id);
  done(Status::OK());
}

void Worker::CleanupAllAsync(const CleanupAllRequest* request,
                             CleanupAllResponse* response,
                             StatusCallback done) {
  std::vector<string> containers;
  for (const auto& c : request->container()) containers.push_back(c);
  env_->device_mgr->ClearContainers(containers);
  done(Status::OK());
}

void Worker::LoggingAsync(const LoggingRequest* request,
                          LoggingResponse* response, StatusCallback done) {
  done(errors::Unimplemented("Logging"));
}

void Worker::TracingAsync(const TracingRequest* request,
                          TracingResponse* response, StatusCallback done) {
  done(errors::Unimplemented("Tracing"));
}

// Helper for RecvTensor. Validates "key" and returns the source
// device in "*src_dev".
Status Worker::PrepareRecvTensor(const Rendezvous::ParsedKey& parsed,
                                 Device** src_dev) {
  // Figures out which device the tensor is hosted on.
  TF_RETURN_IF_ERROR(
      env_->device_mgr->LookupDevice(parsed.src_device, src_dev));

  // Does the device have the right incarnation number we expect?
  if ((*src_dev)->attributes().incarnation() != parsed.src_incarnation) {
    return errors::Aborted(
        "RecvTensor expects a different device incarnation: ",
        parsed.src_incarnation, " vs. ", (*src_dev)->attributes().incarnation(),
        ". Your worker job was probably restarted. Check your "
        "worker job for the reason why it was restarted.");
  }

  return Status::OK();
}

void Worker::RecvTensorAsync(CallOptions* opts,
                             const RecvTensorRequest* request,
                             TensorResponse* response, StatusCallback done) {
  // The base Worker class does not implement RecvTensorAsync, because
  // it is not currently used for worker-to-worker communication. Use a
  // transport-specific implementation (such as `GrpcWorker::RecvTensorAsync()`)
  // instead.
  done(errors::Unimplemented("Worker::RecvTensorAsync()"));
}

}  // namespace tensorflow
