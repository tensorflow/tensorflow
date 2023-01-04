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

#include "tensorflow/core/common_runtime/collective_executor_mgr.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/process_util.h"
#include "tensorflow/core/common_runtime/scoped_allocator_mgr.h"
#include "tensorflow/core/common_runtime/step_stats_collector.h"
#include "tensorflow/core/distributed_runtime/error_payloads.h"
#include "tensorflow/core/distributed_runtime/rendezvous_mgr_interface.h"
#include "tensorflow/core/distributed_runtime/tensor_coding.h"
#include "tensorflow/core/distributed_runtime/worker_session.h"
#include "tensorflow/core/framework/collective.h"
#include "tensorflow/core/platform/tracing.h"
#include "tensorflow/core/profiler/lib/device_profiler_session.h"
#include "tensorflow/tsl/protobuf/distributed_runtime_payloads.pb.h"

namespace tensorflow {

Worker::Worker(WorkerEnv* env)
    : env_(env), recent_request_ids_(100000, env_->experimental_num_shards) {
  DCHECK_GT(env_->experimental_num_shards, 0);

  // Enable log history collection in StatusGroup so that recent warning and
  // error log messages will be attached to the root error status to be
  // forwarded to the master.
  StatusGroup::ConfigureLogHistory();
}

void Worker::GetStatusAsync(CallOptions* opts, const GetStatusRequest* request,
                            GetStatusResponse* response, bool fail_fast,
                            StatusCallback done) {
  const DeviceMgr* dm = env_->device_mgr;
  std::vector<DeviceAttributes> devices;
  dm->ListDeviceAttributes(&devices);
  response->mutable_device_attributes()->Reserve(devices.size());
  for (auto& d : devices) {
    response->add_device_attributes()->Swap(&d);
  }
  done(OkStatus());
}

void Worker::CreateWorkerSessionAsync(const CreateWorkerSessionRequest* request,
                                      CreateWorkerSessionResponse* response,
                                      StatusCallback done) {
  Status s = env_->session_mgr->CreateSession(
      request->session_handle(), request->server_def(),
      request->cluster_device_attributes(), request->isolate_session_state(),
      request->master_task(), request->master_incarnation());
  done(s);
}

void Worker::DeleteWorkerSessionAsync(CallOptions* opts,
                                      const DeleteWorkerSessionRequest* request,
                                      DeleteWorkerSessionResponse* response,
                                      StatusCallback done) {
  Status s = env_->session_mgr->DeleteSession(request->session_handle());
  done(s);
}

void Worker::RegisterGraphAsync(const RegisterGraphRequest* request,
                                RegisterGraphResponse* response,
                                StatusCallback done) {
  std::shared_ptr<WorkerSession> session;
  Status s;
  if (request->create_worker_session_called()) {
    s = env_->session_mgr->WorkerSessionForSession(request->session_handle(),
                                                   &session);
  } else {
    session = env_->session_mgr->LegacySession();
  }
  if (s.ok()) {
    s = session->graph_mgr()->Register(
        request->session_handle(), request->graph_def(),
        request->graph_options(), request->debug_options(),
        request->config_proto(), request->collective_graph_key(), session.get(),
        session->cluster_flr(), response->mutable_graph_handle());
  }
  done(s);
}

void Worker::DeregisterGraphAsync(const DeregisterGraphRequest* request,
                                  DeregisterGraphResponse* response,
                                  StatusCallback done) {
  std::shared_ptr<WorkerSession> session;
  Status s;
  if (request->create_worker_session_called()) {
    s = env_->session_mgr->WorkerSessionForSession(request->session_handle(),
                                                   &session);
  } else {
    session = env_->session_mgr->LegacySession();
  }
  if (s.ok()) {
    s = session->graph_mgr()->Deregister(request->graph_handle());
  }

  done(s);
}

void Worker::AbortStep(int64_t step_id) {
  RemoteRendezvous* rendez = env_->rendezvous_mgr->Find(step_id);
  // Do not abort if it's a context global instance for eager op-by-op execution
  if (rendez->IsRemoteEagerContextDefault()) return;
  SchedNonBlockingClosureAfter(1000000, [rendez, step_id]() {
    // Delay a bit before aborting the step. This way, the root
    // cause may return first back to the client instead of this
    // cancellation generated abort error.
    rendez->StartAbort(errors::Aborted("Step ", step_id,
                                       " cancelled.  Cancelling rendezvous."));
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
  return OkStatus();
}

void Worker::RunGraphAsync(CallOptions* opts, RunGraphRequestWrapper* request,
                           MutableRunGraphResponseWrapper* response,
                           StatusCallback done) {
  if (request->store_errors_in_response_body()) {
    done = [response, done](const Status& status) {
      response->set_status(status);
      done(OkStatus());
    };
  }
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
  const int64_t step_id = request->step_id();
  TRACEPRINTF("RunGraph: %lld", step_id);
  Status s = recent_request_ids_.TrackUnique(request->request_id(),
                                             "RunGraph (Worker)", request);
  if (!s.ok()) {
    done(s);
    return;
  }

  std::shared_ptr<WorkerSession> session;
  if (request->create_worker_session_called()) {
    s = env_->session_mgr->WorkerSessionForSession(request->session_handle(),
                                                   &session);
  } else {
    session = env_->session_mgr->LegacySession();
  }
  if (!s.ok()) {
    done(s);
    return;
  }
  GraphMgr::NamedTensors in;
  GraphMgr::NamedTensors* out = new GraphMgr::NamedTensors;
  s = PrepareRunGraph(request, &in, out);
  if (!s.ok()) {
    delete out;
    done(s);
    return;
  }
  StepStatsCollector* collector = nullptr;
  if (request->exec_opts().report_tensor_allocations_upon_oom() ||
      request->exec_opts().record_timeline() ||
      request->exec_opts().record_costs()) {
    collector = new StepStatsCollector(response->mutable_step_stats());
  }
  DeviceProfilerSession* device_profiler_session = nullptr;
  if (collector && request->exec_opts().record_timeline()) {
    // If timeline was requested, assume we want hardware level tracing.
    device_profiler_session = DeviceProfilerSession::Create().release();
  }
  CancellationManager* cm = new CancellationManager;
  opts->SetCancelCallback([this, cm, step_id]() {
    LOG(INFO) << "Cancellation requested for RunGraph.";
    cm->StartCancel();
    AbortStep(step_id);
  });
  CancellationToken token;
  token = cancellation_manager_.get_cancellation_token();
  bool already_cancelled = !cancellation_manager_.RegisterCallback(
      token, [cm]() { cm->StartCancel(); });
  if (already_cancelled) {
    opts->ClearCancelCallback();
    delete cm;
    delete collector;
    delete device_profiler_session;
    delete out;
    done(errors::Aborted("Call was aborted"));
    return;
  }
  session->graph_mgr()->ExecuteAsync(
      request->graph_handle(), step_id, request->exec_opts(), in, session.get(),
      collector, response, cm, env_->session_mgr->GetCoordinationServiceAgent(),
      [this, step_id, response, session, cm, out, token, collector,
       device_profiler_session, opts, done](const Status& status) {
        Status s = status;
        if (s.ok()) {
          s = session->graph_mgr()->RecvOutputs(step_id, out);
        }

        opts->ClearCancelCallback();
        cancellation_manager_.DeregisterCallback(token);
        delete cm;

        if (device_profiler_session) {
          device_profiler_session->CollectData(response->mutable_step_stats())
              .IgnoreError();
        }

        if (s.ok()) {
          for (const auto& p : *out) {
            const string& key = p.first;
            const Tensor& val = p.second;
            response->AddRecv(key, val);
          }
        }

        if (collector) collector->Finalize();
        delete collector;
        delete device_profiler_session;
        delete out;
        done(s);
      });
}

// TODO(suharshs): Add stats collection support to partial run.
void Worker::DoPartialRunGraph(CallOptions* opts,
                               RunGraphRequestWrapper* request,
                               MutableRunGraphResponseWrapper* response,
                               StatusCallback done) {
  const int64_t step_id = request->step_id();
  const string& graph_handle = request->graph_handle();
  TRACEPRINTF("PartialRunGraph: %lld", step_id);
  Status s = recent_request_ids_.TrackUnique(
      request->request_id(), "PartialRunGraph (Worker)", request);
  if (!s.ok()) {
    done(s);
    return;
  }

  std::shared_ptr<WorkerSession> session;
  if (request->create_worker_session_called()) {
    s = env_->session_mgr->WorkerSessionForSession(request->session_handle(),
                                                   &session);
  } else {
    session = env_->session_mgr->LegacySession();
  }
  if (!s.ok()) {
    done(s);
    return;
  }

  GraphMgr::NamedTensors in;
  GraphMgr::NamedTensors* out = new GraphMgr::NamedTensors;
  s = PrepareRunGraph(request, &in, out);
  auto finish = [done, out, opts](const Status& s) {
    opts->ClearCancelCallback();
    delete out;
    done(s);
  };
  if (!s.ok()) {
    finish(s);
    return;
  }

  CancellationManager* cm = nullptr;
  bool is_new_partial_run = partial_run_mgr_.FindOrCreate(step_id, &cm);

  // Before we start doing anything, we set the RPC cancellation.
  opts->SetCancelCallback([this, cm, step_id]() {
    LOG(INFO) << "Cancellation requested for PartialRunGraph.";
    cm->StartCancel();
    AbortStep(step_id);
  });

  // If this is a new partial run request, the request will need to start the
  // executors.
  if (is_new_partial_run) {
    CancellationToken token;
    token = cancellation_manager_.get_cancellation_token();
    cancellation_manager_.RegisterCallback(token,
                                           [cm]() { cm->StartCancel(); });
    session->graph_mgr()->ExecuteAsync(
        graph_handle, step_id, request->exec_opts(), in, session.get(),
        /*collector=*/nullptr, /*response=*/nullptr, cm,
        env_->session_mgr->GetCoordinationServiceAgent(),
        [this, token, step_id, session](Status s) {
          cancellation_manager_.DeregisterCallback(token);
          partial_run_mgr_.ExecutorDone(step_id, s);
        });
  } else {
    // Send the partial run's new inputs.
    s = session->graph_mgr()->SendInputs(step_id, in);
    if (!s.ok()) {
      finish(s);
      return;
    }
  }

  session->graph_mgr()->RecvOutputsAsync(
      step_id, out, [this, out, request, response, step_id, finish](Status s) {
        if (s.ok()) {
          // Construct and return the resp.
          for (const auto& p : *out) {
            const string& key = p.first;
            const Tensor& val = p.second;
            response->AddRecv(key, val);
          }
        }
        if (request->is_last_partial_run()) {
          partial_run_mgr_.PartialRunDone(step_id, finish, s);
        } else {
          finish(s);
        }
      });
}

void Worker::CleanupGraphAsync(const CleanupGraphRequest* request,
                               CleanupGraphResponse* response,
                               StatusCallback done) {
  const int64_t step_id = request->step_id();
  env_->rendezvous_mgr->Cleanup(step_id);
  if (env_->collective_executor_mgr) {
    env_->collective_executor_mgr->Cleanup(step_id);
  }
  for (Device* d : env_->local_devices) {
    ScopedAllocatorMgr* sam = d->GetScopedAllocatorMgr();
    if (sam) {
      sam->Cleanup(step_id);
    }
  }
  done(OkStatus());
}

void Worker::CleanupAllAsync(const CleanupAllRequest* request,
                             CleanupAllResponse* response,
                             StatusCallback done) {
  std::vector<string> containers;
  for (const auto& c : request->container()) containers.push_back(c);
  env_->device_mgr->ClearContainers(containers);
  done(OkStatus());
}

void Worker::LoggingAsync(const LoggingRequest* request,
                          LoggingResponse* response, StatusCallback done) {
  done(errors::Unimplemented("Logging"));
}

void Worker::TracingAsync(const TracingRequest* request,
                          TracingResponse* response, StatusCallback done) {
  done(errors::Unimplemented("Tracing"));
}

void Worker::RecvBufAsync(CallOptions* opts, const RecvBufRequest* request,
                          RecvBufResponse* response, StatusCallback done) {
  // The base Worker class does not implement RecvBufAsync because
  // it is not currently used for worker-to-worker communication. Use a
  // transport-specific implementation (such as `GrpcWorker::RecvBufAsync()`)
  // instead.
  done(errors::Unimplemented("Worker::RecvBufAsync()"));
}

void Worker::CompleteGroupAsync(CallOptions* opts,
                                const CompleteGroupRequest* request,
                                CompleteGroupResponse* response,
                                StatusCallback done) {
  if (!request->has_device_attributes()) {
    done(errors::Internal(
        "CompleteGroupRequest device_attributes is not set. Make sure you're "
        "running the same version of Tensorflow on all workers."));
    return;
  }
  if (env_->collective_executor_mgr) {
    auto group_params = new CollGroupParams();
    group_params->group_key = request->group_key();
    group_params->group_size = request->group_size();
    group_params->device_type = DeviceType(request->device_type());
    env_->collective_executor_mgr->GetParamResolver()->CompleteGroupAsync(
        request->device_attributes(), group_params, &cancellation_manager_,
        [response, group_params, done = std::move(done)](const Status& s) {
          if (s.ok()) {
            response->set_group_key(group_params->group_key);
            response->set_group_size(group_params->group_size);
            response->set_device_type(group_params->device_type.type_string());
            response->set_num_tasks(group_params->num_tasks);
            for (const CollGroupMember& member : group_params->members) {
              *response->add_device_attributes() = member.device;
            }
            response->set_communicator_key(
                group_params->runtime_details.communicator_key);
          } else {
            LOG(ERROR) << "Bad status from CompleteGroupDistributed: " << s;
          }
          delete group_params;
          done(s);
        });
  } else {
    done(
        errors::Internal("Runtime not initialized with CollectiveExecutorMgr"));
  }
}

void Worker::CompleteInstanceAsync(CallOptions* opts,
                                   const CompleteInstanceRequest* request,
                                   CompleteInstanceResponse* response,
                                   StatusCallback done) {
  if (env_->collective_executor_mgr) {
    env_->collective_executor_mgr->GetParamResolver()->CompleteInstanceAsync(
        request, response, &cancellation_manager_, done);
  } else {
    done(
        errors::Internal("Runtime not initialized with CollectiveExecutorMgr"));
  }
}

void Worker::GetStepSequenceAsync(const GetStepSequenceRequest* request,
                                  GetStepSequenceResponse* response,
                                  StatusCallback done) {
  if (env_->collective_executor_mgr) {
    env_->collective_executor_mgr->GetStepSequenceAsync(request, response,
                                                        done);
  } else {
    done(
        errors::Internal("Runtime not initialized with CollectiveExecutorMgr"));
  }
}

// Helper for RecvTensor. Validates "key" and returns the source
// device in "*src_dev".
Status Worker::PrepareRecvTensor(const Rendezvous::ParsedKey& parsed,
                                 Device** src_dev) {
  // Figures out which device the tensor is hosted on.
  string local_name = DeviceNameUtils::LocalName(parsed.src_device);
  TF_RETURN_IF_ERROR(env_->device_mgr->LookupDevice(local_name, src_dev));

  // Does the device have the right incarnation number we expect?
  if ((*src_dev)->attributes().incarnation() != parsed.src_incarnation) {
    return errors::AbortedWithPayloads(
        strings::StrCat("RecvTensor expects a different device incarnation: ",
                        parsed.src_incarnation, " vs. ",
                        (*src_dev)->attributes().incarnation(),
                        ". Your worker job (\"",
                        env_->session_mgr->LegacySession()->worker_name(),
                        "\") was probably restarted. Check your "
                        "worker job for the reason why it was restarted."),
        {{kWorkerPossiblyRestarted,
          distributed_runtime::WorkerPossiblyRestarted().SerializeAsString()}});
  }

  return OkStatus();
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
