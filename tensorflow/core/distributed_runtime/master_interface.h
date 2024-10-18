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

#ifndef TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_MASTER_INTERFACE_H_
#define TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_MASTER_INTERFACE_H_

#include "tensorflow/core/distributed_runtime/call_options.h"
#include "tensorflow/core/distributed_runtime/message_wrappers.h"
#include "tensorflow/core/distributed_runtime/request_id.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/protobuf/master.pb.h"

namespace tensorflow {

// Abstract interface for communicating with the TensorFlow Master service.
//
// This interface supports both RPC-based master implementations, and
// in-process master implementations that do not require an RPC
// roundtrip.
class MasterInterface {
 public:
  virtual ~MasterInterface() {}
  virtual absl::Status CreateSession(CallOptions* call_options,
                                     const CreateSessionRequest* request,
                                     CreateSessionResponse* response) = 0;

  virtual absl::Status ExtendSession(CallOptions* call_options,
                                     const ExtendSessionRequest* request,
                                     ExtendSessionResponse* response) = 0;

  virtual absl::Status PartialRunSetup(CallOptions* call_options,
                                       const PartialRunSetupRequest* request,
                                       PartialRunSetupResponse* response) {
    return errors::Unimplemented("Partial run not implemented for this master");
  }

  virtual absl::Status RunStep(CallOptions* call_options,
                               RunStepRequestWrapper* request,
                               MutableRunStepResponseWrapper* response) = 0;

  virtual absl::Status RunStep(CallOptions* call_options,
                               const RunStepRequest* request,
                               RunStepResponse* response) {
    std::unique_ptr<RunStepRequestWrapper> wrapped_request(
        new ProtoRunStepRequest(request));
    std::unique_ptr<MutableRunStepResponseWrapper> wrapped_response(
        new NonOwnedProtoRunStepResponse(response));
    return RunStep(call_options, wrapped_request.get(), wrapped_response.get());
  }

  // Returns a request object for use in calls to
  // `RunStep()`. Ownership is transferred to the caller.
  //
  // The message returned from this method must only be used in a
  // `RunStep()` call on the same `MasterInterface` instance.
  virtual MutableRunStepRequestWrapper* CreateRunStepRequest() {
    MutableProtoRunStepRequest* ret = new MutableProtoRunStepRequest;
    ret->request_.set_request_id(GetUniqueRequestId());
    return ret;
  }

  // Returns a response object for use in calls to
  // `RunStep()`. Ownership is transferred to the caller.
  //
  // The message returned from this method must only be used in a
  // `RunStep()` call on the same `MasterInterface` instance.
  virtual MutableRunStepResponseWrapper* CreateRunStepResponse() {
    return new OwnedProtoRunStepResponse;
  }

  virtual absl::Status CloseSession(CallOptions* call_options,
                                    const CloseSessionRequest* request,
                                    CloseSessionResponse* response) = 0;

  virtual absl::Status ListDevices(CallOptions* call_options,
                                   const ListDevicesRequest* request,
                                   ListDevicesResponse* response) = 0;

  virtual absl::Status Reset(CallOptions* call_options,
                             const ResetRequest* request,
                             ResetResponse* response) = 0;

  virtual absl::Status MakeCallable(CallOptions* call_options,
                                    const MakeCallableRequest* request,
                                    MakeCallableResponse* response) = 0;
  virtual absl::Status RunCallable(CallOptions* call_options,
                                   const RunCallableRequest* request,
                                   RunCallableResponse* response) = 0;
  virtual absl::Status ReleaseCallable(CallOptions* call_options,
                                       const ReleaseCallableRequest* request,
                                       ReleaseCallableResponse* response) = 0;

 protected:
  // NOTE: This should only be called by implementations of this
  // interface whose CreateRunStepResponse() method returns a
  // proto-based wrappers for the RunStepResponse message.
  RunStepResponse* get_proto_from_wrapper(
      MutableRunStepResponseWrapper* wrapper) {
    return wrapper->get_proto();
  }
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_MASTER_INTERFACE_H_
