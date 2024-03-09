/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/common_runtime/next_pluggable_device/c/tf_rendezvous_c_api_internal.h"

#include <functional>
#include <memory>
#include <utility>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow/c/tf_status_helper.h"
#include "tensorflow/c/tf_tensor.h"
#include "tensorflow/c/tf_tensor_helper.h"
#include "tensorflow/core/common_runtime/next_pluggable_device/c/tf_device_context_c_api.h"
#include "tensorflow/core/common_runtime/next_pluggable_device/c/tf_device_context_c_api_internal.h"
#include "tensorflow/core/common_runtime/next_pluggable_device/c/tf_rendezvous_c_api.h"
#include "tensorflow/core/framework/cancellation.h"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/framework/rendezvous.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/statusor.h"
#include "tsl/framework/allocator.h"
#include "tsl/platform/status.h"

#define CONCAT_HELPER(a, b) a##b
#define CONCAT(a, b) CONCAT_HELPER(a, b)
#define CHECK_OK_AND_ASSIGN(lhs, expr)        \
  auto CONCAT(status_or_, __LINE__) = (expr); \
  CHECK(CONCAT(status_or_, __LINE__).ok());   \
  lhs = std::move(CONCAT(status_or_, __LINE__)).value();

namespace tensorflow {

namespace {

tsl::AllocatorAttributes FromC(
    const TFDevice_AllocatorAttributes& c_attributes) {
  tsl::AllocatorAttributes attributes;
  attributes.value = c_attributes.value;
  attributes.scope_id = c_attributes.scope_id;
  return attributes;
}

void Destroy(TFDevice_AllocatorAttributes* c_attributes) {}

static CancellationManager* FromC(
    TFE_CancellationManager* c_cancellation_manager) {
  if (c_cancellation_manager == nullptr) {
    return nullptr;
  }
  return reinterpret_cast<CancellationManager*>(c_cancellation_manager);
}

RendezvousInterface::Args FromC(const TF_RendezvousArgsStruct& c_args) {
  RendezvousInterface::Args args;
  args.device_context = DeviceContext_FromC(c_args.device_context);
  args.alloc_attrs = FromC(c_args.alloc_attrs);
  args.cancellation_manager = FromC(c_args.cancellation_manager);
  return args;
}

RendezvousInterface::ParsedKey FromC(const TF_RendezvousParsedKey& c_key) {
  RendezvousInterface::ParsedKey key;
  absl::string_view full_key(c_key.full_key, c_key.full_key_size);
  TF_CHECK_OK(Rendezvous::ParseKey(full_key, &key));
  return key;
}

using DoneCallbackParamDeleter =
    std::function<void(TF_RendezvousDoneCallback_Params*)>;

using DoneCallbackParamPtr =
    std::unique_ptr<TF_RendezvousDoneCallback_Params, DoneCallbackParamDeleter>;

DoneCallbackParamDeleter MakeDoneCallbackParamDeleter() {
  return [](TF_RendezvousDoneCallback_Params* params) {
    if (params == nullptr) {
      return;
    }
    // TODO: Pass args through.
    // TF_RendezvousArgsStruct* sender_args =
    //     const_cast<TF_RendezvousArgsStruct*>(params->sender_args);
    // TF_RendezvousArgsStruct* recver_args =
    //     const_cast<TF_RendezvousArgsStruct*>(params->recver_args);
    // Destroy(sender_args);
    // Destroy(recver_args);
    // delete params->sender_args;
    // delete params->recver_args;
    TF_Status* status = const_cast<TF_Status*>(params->status);
    TF_DeleteStatus(status);
    params->status = nullptr;

    TF_DeleteTensor(const_cast<TF_Tensor*>(params->tensor));
    params->tensor = nullptr;

    delete params;
  };
}

absl::StatusOr<DoneCallbackParamPtr> DoneCallbackParamsToC(
    const Status& status, const RendezvousInterface::Args& sender_args,
    const RendezvousInterface::Args& recver_args, const Tensor& tensor,
    const bool is_dead) {
  TF_RendezvousDoneCallback_Params* params =
      new TF_RendezvousDoneCallback_Params;
  TF_Status* c_status = TF_NewStatus();
  tsl::Set_TF_Status_from_Status(c_status, status);
  params->status = c_status;
  // TODO: Pass args through.
  // params->sender_args = new TF_RendezvousArgsStruct(ToC(sender_args));
  // params->recver_args = new TF_RendezvousArgsStruct(ToC(recver_args));
  Status tensor_status;
  params->tensor = TF_TensorFromTensor(tensor, &tensor_status);
  if (!tensor_status.ok()) {
    MakeDoneCallbackParamDeleter()(params);
    return tensor_status;
  }
  params->is_dead = is_dead;
  return DoneCallbackParamPtr(params, MakeDoneCallbackParamDeleter());
}

RendezvousInterface::DoneCallback FromC(
    const TF_RendezvousDoneCallbackImpl& c_on_done) {
  if (c_on_done.context == nullptr) {
    return nullptr;
  }
  TF_RendezvousDoneCallback_Function callback = c_on_done.callback;
  void* context = c_on_done.context;
  auto cpp_callback = [callback, context](const Status& status,
                                          RendezvousInterface::Args sender_args,
                                          RendezvousInterface::Args recver_args,
                                          const Tensor& tensor,
                                          const bool is_dead) -> void {
    CHECK_OK_AND_ASSIGN(DoneCallbackParamPtr params,
                        DoneCallbackParamsToC(status, sender_args, recver_args,
                                              tensor, is_dead));
    callback(context, params.get());
  };
  return cpp_callback;
}

void SendFunctionThunk(void* opa_rendezvous, TF_RendezvousSend_Params* params) {
  RendezvousInterface* rendezvous =
      static_cast<RendezvousInterface*>(opa_rendezvous);
  RendezvousInterface::ParsedKey key = FromC(*params->key);
  RendezvousInterface::Args args = FromC(*params->args);
  Tensor tensor;
  Status tensor_status = TF_TensorToTensor(params->tensor, &tensor);
  bool is_dead = params->is_dead;
  if (tensor_status.ok()) {
    tsl::Set_TF_Status_from_Status(
        params->status, rendezvous->Send(key, args, tensor, is_dead));
  } else {
    tsl::Set_TF_Status_from_Status(params->status, tensor_status);
  }
  // Releases TfCThunkDeviceContext allocated in FromC().
  args.device_context->Unref();
}

void RecvFunctionThunk(void* opa_rendezvous,
                       TF_RendezvousAsyncRecv_Params* params) {
  RendezvousInterface* rendezvous =
      static_cast<RendezvousInterface*>(opa_rendezvous);
  RendezvousInterface::ParsedKey key = FromC(*params->key);
  RendezvousInterface::Args args = FromC(*params->args);
  RendezvousInterface::DoneCallback on_done =
      [device_context = args.device_context, on_done = params->on_done](
          const Status& status, const RendezvousInterface::Args& send_args,
          const RendezvousInterface::Args& recv_args, const Tensor& tensor,
          const bool is_dead) {
        FromC(on_done)(status, send_args, recv_args, tensor, is_dead);
        // Releases TfCThunkDeviceContext allocated in FromC().
        device_context->Unref();
      };
  rendezvous->RecvAsync(key, args, on_done);
}

void StartAbortFunctionThunk(void* opa_rendezvous, const TF_Status* tf_status) {
  auto* rendezvous = static_cast<RendezvousInterface*>(opa_rendezvous);
  absl::Status status = tsl::StatusFromTF_Status(tf_status);
  rendezvous->StartAbort(status);
}

}  // namespace

TF_RendezvousThunk* ToC(RendezvousInterface* rendezvous) {
  TF_RendezvousThunk* thunk = new TF_RendezvousThunk();
  thunk->rendezvous = static_cast<void*>(rendezvous);
  thunk->send_func = SendFunctionThunk;
  thunk->async_recv_func = RecvFunctionThunk;
  thunk->start_abort_func = StartAbortFunctionThunk;
  return thunk;
}

void Destroy(TF_RendezvousThunk* thunk) {}

}  // namespace tensorflow
