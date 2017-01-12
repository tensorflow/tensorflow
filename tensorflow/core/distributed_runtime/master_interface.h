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
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/protobuf/master.pb.h"

namespace tensorflow {

// Pure virtual interface for communicating with the TensorFlow Master service.
//
// This interface is intended to support in-process master
// implementations that do not require an RPC roundtrip.
class MasterInterface {
 public:
  virtual ~MasterInterface() {}
  virtual Status CreateSession(CallOptions* call_options,
                               const CreateSessionRequest* request,
                               CreateSessionResponse* response) = 0;

  virtual Status ExtendSession(CallOptions* call_options,
                               const ExtendSessionRequest* request,
                               ExtendSessionResponse* response) = 0;

  virtual Status PartialRunSetup(CallOptions* call_options,
                                 const PartialRunSetupRequest* request,
                                 PartialRunSetupResponse* response) {
    return errors::Unimplemented("Partial run not implemented for this master");
  }

  virtual Status RunStep(CallOptions* call_options,
                         RunStepRequestWrapper* request,
                         RunStepResponse* response) = 0;

  virtual Status RunStep(CallOptions* call_options,
                         const RunStepRequest* request,
                         RunStepResponse* response) {
    std::unique_ptr<RunStepRequestWrapper> wrapped_request(
        new ProtoRunStepRequest(request));
    return RunStep(call_options, wrapped_request.get(), response);
  }

  virtual MutableRunStepRequestWrapper* CreateRunStepRequest() {
    return new MutableProtoRunStepRequest;
  }

  virtual Status CloseSession(CallOptions* call_options,
                              const CloseSessionRequest* request,
                              CloseSessionResponse* response) = 0;

  virtual Status ListDevices(CallOptions* call_options,
                             const ListDevicesRequest* request,
                             ListDevicesResponse* response) = 0;

  virtual Status Reset(CallOptions* call_options, const ResetRequest* request,
                       ResetResponse* response) = 0;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_MASTER_INTERFACE_H_
