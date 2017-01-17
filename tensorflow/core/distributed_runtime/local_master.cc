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

#include "tensorflow/core/distributed_runtime/local_master.h"

#include <unordered_map>

#include "tensorflow/core/distributed_runtime/master.h"
#include "tensorflow/core/platform/mutex.h"

namespace tensorflow {

namespace {

Status WaitForNotification(CallOptions* call_options, Notification* n) {
  int64 timeout_in_ms = call_options->GetTimeout();
  if (timeout_in_ms > 0) {
    bool notified = WaitForNotificationWithTimeout(n, timeout_in_ms);
    if (!notified) {
      call_options->StartCancel();
      // The call has borrowed pointers to the request and response
      // messages, so we must still wait for the call to complete.
      n->WaitForNotification();
      return errors::DeadlineExceeded("Operation timed out.");
    }
  } else {
    n->WaitForNotification();
  }
  return Status::OK();
}
}

LocalMaster::LocalMaster(Master* master_impl) : master_impl_(master_impl) {}

Status LocalMaster::CreateSession(CallOptions* call_options,
                                  const CreateSessionRequest* request,
                                  CreateSessionResponse* response) {
  Notification n;
  Status ret;
  master_impl_->CreateSession(request, response, [&n, &ret](const Status& s) {
    ret.Update(s);
    n.Notify();
  });
  TF_RETURN_IF_ERROR(WaitForNotification(call_options, &n));
  return ret;
}

Status LocalMaster::ExtendSession(CallOptions* call_options,
                                  const ExtendSessionRequest* request,
                                  ExtendSessionResponse* response) {
  Notification n;
  Status ret;
  master_impl_->ExtendSession(request, response, [&n, &ret](const Status& s) {
    ret.Update(s);
    n.Notify();
  });
  TF_RETURN_IF_ERROR(WaitForNotification(call_options, &n));
  return ret;
}

Status LocalMaster::PartialRunSetup(CallOptions* call_options,
                                    const PartialRunSetupRequest* request,
                                    PartialRunSetupResponse* response) {
  Notification n;
  Status ret;
  master_impl_->PartialRunSetup(request, response, [&n, &ret](const Status& s) {
    ret.Update(s);
    n.Notify();
  });
  TF_RETURN_IF_ERROR(WaitForNotification(call_options, &n));
  return ret;
}

Status LocalMaster::RunStep(CallOptions* call_options,
                            RunStepRequestWrapper* request,
                            MutableRunStepResponseWrapper* response) {
  Notification n;
  Status ret;
  master_impl_->RunStep(call_options, request, response,
                        [&n, &ret](const Status& s) {
                          ret.Update(s);
                          n.Notify();
                        });
  TF_RETURN_IF_ERROR(WaitForNotification(call_options, &n));
  return ret;
}

MutableRunStepRequestWrapper* LocalMaster::CreateRunStepRequest() {
  return new InMemoryRunStepRequest;
}

MutableRunStepResponseWrapper* LocalMaster::CreateRunStepResponse() {
  return new InMemoryRunStepResponse;
}

Status LocalMaster::CloseSession(CallOptions* call_options,
                                 const CloseSessionRequest* request,
                                 CloseSessionResponse* response) {
  Notification n;
  Status ret;
  master_impl_->CloseSession(request, response, [&n, &ret](const Status& s) {
    ret.Update(s);
    n.Notify();
  });
  TF_RETURN_IF_ERROR(WaitForNotification(call_options, &n));
  return ret;
}

Status LocalMaster::ListDevices(CallOptions* call_options,
                                const ListDevicesRequest* request,
                                ListDevicesResponse* response) {
  Notification n;
  Status ret;
  master_impl_->ListDevices(request, response, [&n, &ret](const Status& s) {
    ret.Update(s);
    n.Notify();
  });
  TF_RETURN_IF_ERROR(WaitForNotification(call_options, &n));
  return ret;
}

Status LocalMaster::Reset(CallOptions* call_options,
                          const ResetRequest* request,
                          ResetResponse* response) {
  Notification n;
  Status ret;
  master_impl_->Reset(request, response, [&n, &ret](const Status& s) {
    ret.Update(s);
    n.Notify();
  });
  TF_RETURN_IF_ERROR(WaitForNotification(call_options, &n));
  return ret;
}

namespace {
mutex* get_local_master_registry_lock() {
  static mutex local_master_registry_lock;
  return &local_master_registry_lock;
}

typedef std::unordered_map<string, Master*> LocalMasterRegistry;
LocalMasterRegistry* local_master_registry() {
  static LocalMasterRegistry* local_master_registry_ = new LocalMasterRegistry;
  return local_master_registry_;
}
}  // namespace

/* static */
void LocalMaster::Register(const string& target, Master* master) {
  mutex_lock l(*get_local_master_registry_lock());
  local_master_registry()->insert({target, master});
}

/* static */
std::unique_ptr<LocalMaster> LocalMaster::Lookup(const string& target) {
  std::unique_ptr<LocalMaster> ret;
  mutex_lock l(*get_local_master_registry_lock());
  auto iter = local_master_registry()->find(target);
  if (iter != local_master_registry()->end()) {
    ret.reset(new LocalMaster(iter->second));
  }
  return ret;
}

}  // namespace tensorflow
