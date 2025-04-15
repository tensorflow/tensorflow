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

#ifndef TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_LOCAL_MASTER_H_
#define TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_LOCAL_MASTER_H_

#include <memory>

#include "tensorflow/core/distributed_runtime/master_interface.h"

namespace tensorflow {

class Master;

// An implementation of the TensorFlow master interface that enables direct
// intraprocess communication between the client and the master implementation.
//
// This master implementation is intended to provide more efficient access to
// a master service that has been created in the same process as the client.
//
// TODO(mrry): Add methods that avoid protobuf encoding the request/response
// objects where this affects performance.
// TODO(mrry): Avoid closure creation/context switch overhead for synchronous
// invocation of Master methods.
// TODO(mrry): Make all potentially blocking Master methods take CallOptions
// for cancellation.
class LocalMaster : public MasterInterface {
 public:
  ~LocalMaster() override {}

  absl::Status CreateSession(CallOptions* call_options,
                             const CreateSessionRequest* request,
                             CreateSessionResponse* response) override;

  absl::Status ExtendSession(CallOptions* call_options,
                             const ExtendSessionRequest* request,
                             ExtendSessionResponse* response) override;

  absl::Status PartialRunSetup(CallOptions* call_options,
                               const PartialRunSetupRequest* request,
                               PartialRunSetupResponse* response) override;

  absl::Status RunStep(CallOptions* call_options,
                       RunStepRequestWrapper* request,
                       MutableRunStepResponseWrapper* response) override;

  MutableRunStepRequestWrapper* CreateRunStepRequest() override;

  MutableRunStepResponseWrapper* CreateRunStepResponse() override;

  absl::Status CloseSession(CallOptions* call_options,
                            const CloseSessionRequest* request,
                            CloseSessionResponse* response) override;

  absl::Status ListDevices(CallOptions* call_options,
                           const ListDevicesRequest* request,
                           ListDevicesResponse* response) override;

  // See tensorflow::Reset() and the comment on ResetRequest.
  absl::Status Reset(CallOptions* call_options, const ResetRequest* request,
                     ResetResponse* response) override;

  absl::Status MakeCallable(CallOptions* call_options,
                            const MakeCallableRequest* request,
                            MakeCallableResponse* response) override;
  absl::Status RunCallable(CallOptions* call_options,
                           const RunCallableRequest* request,
                           RunCallableResponse* response) override;
  absl::Status ReleaseCallable(CallOptions* call_options,
                               const ReleaseCallableRequest* request,
                               ReleaseCallableResponse* response) override;

  // Registers the mapping from the given `target` to the given `master`.
  //
  // WARNING: The `master` pointer remains owned by the caller. It is
  // the responsibility of the caller to ensure that `master` outlives
  // any LocalMaster objects that may wrap this master. There is no
  // corresponding deregister method, since clean server shutdown is
  // not currently implemented for any server type.
  static void Register(const string& target, Master* master,
                       int64_t default_timeout_in_ms);

  // Returns a pointer to the local master associated with the given
  // `target`, or nullptr if none exists.
  static std::unique_ptr<LocalMaster> Lookup(const string& target);

 private:
  Master* master_impl_;  // Not owned.
  const int64_t default_timeout_in_ms_;

  // See `LocalMaster::Lookup` for the factory function that creates
  // objects of this type.
  LocalMaster(Master* master_impl, const int64_t default_timeout_in_ms);

  LocalMaster(const LocalMaster&) = delete;
  void operator=(const LocalMaster&) = delete;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_LOCAL_MASTER_H_
