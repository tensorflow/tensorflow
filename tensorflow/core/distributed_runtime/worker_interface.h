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

#ifndef TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_WORKER_INTERFACE_H_
#define TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_WORKER_INTERFACE_H_

#include <functional>

#include "tensorflow/core/distributed_runtime/call_options.h"
#include "tensorflow/core/distributed_runtime/message_wrappers.h"
#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/protobuf/worker.pb.h"

namespace tensorflow {

// Status callback.
typedef std::function<void(const Status&)> StatusCallback;

// Custom decoder for a response to RecvTensorAsync.
class TensorResponse;

// Interface for talking with the TensorFlow Worker service.
class WorkerInterface {
 public:
  virtual void GetStatusAsync(const GetStatusRequest* request,
                              GetStatusResponse* response,
                              StatusCallback done) = 0;

  virtual void CreateWorkerSessionAsync(
      const CreateWorkerSessionRequest* request,
      CreateWorkerSessionResponse* response, StatusCallback done) = 0;

  virtual void DeleteWorkerSessionAsync(
      CallOptions* opts, const DeleteWorkerSessionRequest* request,
      DeleteWorkerSessionResponse* response, StatusCallback done) = 0;

  virtual void RegisterGraphAsync(const RegisterGraphRequest* request,
                                  RegisterGraphResponse* response,
                                  StatusCallback done) = 0;

  virtual void DeregisterGraphAsync(const DeregisterGraphRequest* request,
                                    DeregisterGraphResponse* response,
                                    StatusCallback done) = 0;

  virtual void RunGraphAsync(CallOptions* opts, RunGraphRequestWrapper* request,
                             MutableRunGraphResponseWrapper* repsonse,
                             StatusCallback done) = 0;

  virtual void RunGraphAsync(CallOptions* opts, const RunGraphRequest* request,
                             RunGraphResponse* response, StatusCallback done) {
    // TODO(mrry): Convert this to std::bind/std::move if the overhead
    // of std::function copying becomes too much.
    RunGraphRequestWrapper* wrapped_request = new ProtoRunGraphRequest(request);
    MutableRunGraphResponseWrapper* wrapped_response =
        new NonOwnedProtoRunGraphResponse(response);
    RunGraphAsync(opts, wrapped_request, wrapped_response,
                  [wrapped_request, wrapped_response, done](const Status& s) {
                    done(s);
                    delete wrapped_request;
                    delete wrapped_response;
                  });
  }

  // Returns a request object for use in calls to
  // `RunGraphAsync()`. Ownership is transferred to the caller.
  //
  // The message returned from this method must only be used in a
  // `RunGraph()` call on the same `WorkerInterface` instance.
  virtual MutableRunGraphRequestWrapper* CreateRunGraphRequest() {
    return new MutableProtoRunGraphRequest;
  }

  // Returns a response object for use in calls to
  // `RunGraphAsync()`. Ownership is transferred to the caller.
  //
  // The message returned from this method must only be used in a
  // `RunGraph()` call on the same `WorkerInterface` instance.
  virtual MutableRunGraphResponseWrapper* CreateRunGraphResponse() {
    return new OwnedProtoRunGraphResponse;
  }

  virtual void CleanupGraphAsync(const CleanupGraphRequest* request,
                                 CleanupGraphResponse* response,
                                 StatusCallback done) = 0;

  virtual void CleanupAllAsync(const CleanupAllRequest* request,
                               CleanupAllResponse* response,
                               StatusCallback done) = 0;

  virtual void RecvTensorAsync(CallOptions* opts,
                               const RecvTensorRequest* request,
                               TensorResponse* response,
                               StatusCallback done) = 0;

  virtual void LoggingAsync(const LoggingRequest* request,
                            LoggingResponse* response, StatusCallback done) = 0;

  virtual void TracingAsync(const TracingRequest* request,
                            TracingResponse* response, StatusCallback done) = 0;

  virtual void RecvBufAsync(CallOptions* opts, const RecvBufRequest* request,
                            RecvBufResponse* response, StatusCallback done) = 0;

  virtual void CompleteGroupAsync(CallOptions* opts,
                                  const CompleteGroupRequest* request,
                                  CompleteGroupResponse* response,
                                  StatusCallback done) = 0;

  virtual void CompleteInstanceAsync(CallOptions* ops,
                                     const CompleteInstanceRequest* request,
                                     CompleteInstanceResponse* response,
                                     StatusCallback done) = 0;

  virtual void GetStepSequenceAsync(const GetStepSequenceRequest* request,
                                    GetStepSequenceResponse* response,
                                    StatusCallback done) = 0;

  Status GetStatus(const GetStatusRequest* request,
                   GetStatusResponse* response) {
    return CallAndWait(&ME::GetStatusAsync, request, response);
  }

  Status CreateWorkerSession(const CreateWorkerSessionRequest* request,
                             CreateWorkerSessionResponse* response) {
    return CallAndWait(&ME::CreateWorkerSessionAsync, request, response);
  }

  Status DeleteWorkerSession(const DeleteWorkerSessionRequest* request,
                             DeleteWorkerSessionResponse* response) {
    return CallAndWaitWithOptions(&ME::DeleteWorkerSessionAsync, request,
                                  response);
  }

  Status RegisterGraph(const RegisterGraphRequest* request,
                       RegisterGraphResponse* response) {
    return CallAndWait(&ME::RegisterGraphAsync, request, response);
  }

  Status DeregisterGraph(const DeregisterGraphRequest* request,
                         DeregisterGraphResponse* response) {
    return CallAndWait(&ME::DeregisterGraphAsync, request, response);
  }

  Status CleanupGraph(const CleanupGraphRequest* request,
                      CleanupGraphResponse* response) {
    return CallAndWait(&ME::CleanupGraphAsync, request, response);
  }

  Status CleanupAll(const CleanupAllRequest* request,
                    CleanupAllResponse* response) {
    return CallAndWait(&ME::CleanupAllAsync, request, response);
  }

  Status Logging(const LoggingRequest* request, LoggingResponse* response) {
    return CallAndWait(&ME::LoggingAsync, request, response);
  }

  Status Tracing(const TracingRequest* request, TracingResponse* response) {
    return CallAndWait(&ME::TracingAsync, request, response);
  }

  Status GetStepSequence(const GetStepSequenceRequest* request,
                         GetStepSequenceResponse* response) {
    return CallAndWait(&ME::GetStepSequenceAsync, request, response);
  }

 protected:
  // Instances of WorkerInterface must be deleted by a call to
  // WorkerCacheInterface::ReleaseWorker().
  virtual ~WorkerInterface() {}
  friend class WorkerCacheInterface;

  // NOTE: This should only be called by implementations of this
  // interface whose CreateRunGraphResponse() method returns a
  // proto-based wrappers for the RunGraphResponse message.
  RunGraphResponse* get_proto_from_wrapper(
      MutableRunGraphResponseWrapper* wrapper) {
    return wrapper->get_proto();
  }

 private:
  typedef WorkerInterface ME;

  template <typename Method, typename Req, typename Resp>
  Status CallAndWait(Method func, const Req* req, Resp* resp) {
    Status ret;
    Notification n;
    (this->*func)(req, resp, [&ret, &n](const Status& s) {
      ret = s;
      n.Notify();
    });
    n.WaitForNotification();
    return ret;
  }

  template <typename Method, typename Req, typename Resp>
  Status CallAndWaitWithOptions(Method func, const Req* req, Resp* resp) {
    CallOptions call_opts;
    Status ret;
    Notification n;
    (this->*func)(&call_opts, req, resp, [&ret, &n](const Status& s) {
      ret = s;
      n.Notify();
    });
    n.WaitForNotification();
    return ret;
  }
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_WORKER_INTERFACE_H_
