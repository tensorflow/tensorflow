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

#include "tensorflow/core/common_runtime/next_pluggable_device/c/tf_rendezvous_c_api_conversions.h"

#include <cstring>
#include <functional>
#include <memory>
#include <utility>

#include "absl/log/check.h"
#include "absl/strings/string_view.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow/c/tf_status_helper.h"
#include "tensorflow/c/tf_tensor_internal.h"
#include "tensorflow/compiler/xla/stream_executor/tpu/proto_helper.h"
#include "tensorflow/core/common_runtime/next_pluggable_device/c/outside_compilation_params.h"
#include "tensorflow/core/common_runtime/next_pluggable_device/c/tf_device_context_c_api_conversions.h"
#include "tensorflow/core/common_runtime/next_pluggable_device/c/tf_rendezvous_c_api.h"
#include "tensorflow/core/common_runtime/next_pluggable_device/c/tf_rendezvous_c_api_defn.h"
#include "tensorflow/core/framework/cancellation.h"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/framework/rendezvous.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/tsl/platform/status.h"

#define CONCAT_HELPER(a, b) a##b
#define CONCAT(a, b) CONCAT_HELPER(a, b)
#define CHECK_OK_AND_ASSIGN(lhs, expr)        \
  auto CONCAT(status_or_, __LINE__) = (expr); \
  CHECK(CONCAT(status_or_, __LINE__).ok());   \
  lhs = std::move(CONCAT(status_or_, __LINE__)).value();

namespace tensorflow {

TFDevice_AllocatorAttributes ToC(const tsl::AllocatorAttributes& attributes) {
  TFDevice_AllocatorAttributes c_attributes;
  c_attributes.value = attributes.value;
  c_attributes.scope_id = attributes.scope_id;
  return c_attributes;
}

tsl::AllocatorAttributes FromC(
    const TFDevice_AllocatorAttributes& c_attributes) {
  tsl::AllocatorAttributes attributes;
  attributes.value = c_attributes.value;
  attributes.scope_id = c_attributes.scope_id;
  return attributes;
}

void Destroy(TFDevice_AllocatorAttributes* c_attributes) {}

static TFE_CancellationManager* ToC(CancellationManager* cancellation_manager) {
  return reinterpret_cast<TFE_CancellationManager*>(cancellation_manager);
}

static CancellationManager* FromC(
    TFE_CancellationManager* c_cancellation_manager) {
  if (c_cancellation_manager == nullptr) {
    return nullptr;
  }
  return reinterpret_cast<CancellationManager*>(c_cancellation_manager);
}

TF_RendezvousArgsStruct ToC(const RendezvousInterface::Args& args) {
  TF_RendezvousArgsStruct c_args;
  c_args.device_context = ToC(args.device_context);
  c_args.alloc_attrs = ToC(args.alloc_attrs);
  c_args.cancellation_manager = ToC(args.cancellation_manager);
  return c_args;
}

RendezvousInterface::Args FromC(const TF_RendezvousArgsStruct& c_args) {
  RendezvousInterface::Args args;
  args.device_context = FromC(c_args.device_context);
  args.alloc_attrs = FromC(c_args.alloc_attrs);
  args.cancellation_manager = FromC(c_args.cancellation_manager);
  return args;
}

void Destroy(TF_RendezvousArgsStruct* c_args) {
  Destroy(c_args->device_context);
  delete c_args->device_context;
}

TF_RendezvousParsedKey ToC(const RendezvousInterface::ParsedKey& key) {
  TF_RendezvousParsedKey c_key;
  absl::string_view full_key = key.FullKey();
  c_key.full_key_size = full_key.size();
  c_key.full_key = new char[c_key.full_key_size + 1];
  std::strncpy(c_key.full_key, full_key.data(), c_key.full_key_size);
  c_key.full_key[c_key.full_key_size] = 0;
  return c_key;
}

RendezvousInterface::ParsedKey FromC(const TF_RendezvousParsedKey& c_key) {
  RendezvousInterface::ParsedKey key;
  absl::string_view full_key(c_key.full_key, c_key.full_key_size);
  TF_CHECK_OK(Rendezvous::ParseKey(full_key, &key));
  return key;
}

void Destroy(TF_RendezvousParsedKey* c_key) { delete[] c_key->full_key; }

namespace {

using SendParamDeleter = std::function<void(TF_RendezvousSend_Params*)>;
using DoneCallbackParamDeleter =
    std::function<void(TF_RendezvousDoneCallback_Params*)>;

using SendParamPtr =
    std::unique_ptr<TF_RendezvousSend_Params, SendParamDeleter>;
using DoneCallbackParamPtr =
    std::unique_ptr<TF_RendezvousDoneCallback_Params, DoneCallbackParamDeleter>;

SendParamDeleter MakeSendParamDeleter();
StatusOr<SendParamPtr> SendParamsToC(const RendezvousInterface::ParsedKey& key,
                                     const RendezvousInterface::Args& args,
                                     const Tensor& tensor, bool is_dead);

DoneCallbackParamDeleter MakeDoneCallbackParamDeleter();
StatusOr<DoneCallbackParamPtr> DoneCallbackParamsToC(
    const Status& status, const RendezvousInterface::Args& sender_args,
    const RendezvousInterface::Args& recver_args, const Tensor& tensor,
    bool is_dead);

// Use in `TF_RendezvousThunk ToC(tensorflow::RendezvousInterface* rendezvous)`
TF_RendezvousSenderImpl BindSendFunction(RendezvousInterface* rendezvous);

// Use in `TF_RendezvousThunk ToC(tensorflow::RendezvousInterface* rendezvous)`
TF_RendezvousAsyncRecverImpl BindAsyncRecvFunction(
    RendezvousInterface* rendezvous);

TF_RendezvousStartAbortImpl BindStartAborter(RendezvousInterface* rendezvous);

void RendezvousCallbackThunk(void* context,
                             TF_RendezvousDoneCallback_Params* params) {
  using CallbackType = std::function<void(TF_RendezvousDoneCallback_Params*)>;
  auto* callback = static_cast<CallbackType*>(context);
  (*callback)(params);
}

}  // namespace

TF_RendezvousDoneCallbackImpl ToC(
    const RendezvousInterface::DoneCallback& on_done) {
  TF_RendezvousDoneCallbackImpl done_func;
  using CallbackType = std::function<void(TF_RendezvousDoneCallback_Params*)>;
  auto c_callback = new CallbackType(
      [on_done](TF_RendezvousDoneCallback_Params* params) -> void {
        Status status = tsl::StatusFromTF_Status(params->status);
        // TODO: Pass args through.
        // auto sender_args = FromC(*params->sender_args);
        // auto recver_args = FromC(*params->recver_args);
        Tensor tensor;
        CopyTF_TensorToTensor(params->tensor, &tensor);
        on_done(status, RendezvousInterface::Args(),
                RendezvousInterface::Args(), tensor, params->is_dead);
      });
  done_func.context = static_cast<void*>(c_callback);
  done_func.callback = RendezvousCallbackThunk;
  return done_func;
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

void Destroy(TF_RendezvousDoneCallbackImpl* c_on_done) {
  if (c_on_done == nullptr) {
    return;
  }
  if (c_on_done->context != nullptr) {
    auto runner =
        static_cast<std::function<void(TF_RendezvousDoneCallback_Params*)>*>(
            c_on_done->context);
    delete runner;
  }
}

TF_RendezvousThunk* ToC(RendezvousInterface* rendezvous) {
  TF_RendezvousThunk* thunk = new TF_RendezvousThunk();
  thunk->send = BindSendFunction(rendezvous);
  thunk->async_recv = BindAsyncRecvFunction(rendezvous);
  thunk->start_abort = BindStartAborter(rendezvous);
  return thunk;
}

std::unique_ptr<c_api::TfCThunkRendezvous> FromC(
    const TF_RendezvousThunk* thunk) {
  return std::make_unique<c_api::TfCThunkRendezvous>(thunk);
}

void Destroy(TF_RendezvousThunk* thunk) {
  if (thunk == nullptr) {
    return;
  }
  Destroy(&thunk->send);
  Destroy(&thunk->async_recv);
  Destroy(&thunk->start_abort);
}

namespace {

SendParamDeleter MakeSendParamDeleter() {
  return [](TF_RendezvousSend_Params* params) {
    if (params == nullptr) {
      return;
    }
    TF_RendezvousParsedKey* key =
        const_cast<TF_RendezvousParsedKey*>(params->key);
    TF_RendezvousArgsStruct* args =
        const_cast<TF_RendezvousArgsStruct*>(params->args);
    Destroy(key);
    Destroy(args);
    delete params->key;
    delete params->args;
    TF_DeleteTensor(params->tensor);
    TF_DeleteStatus(params->status);
    delete params;
  };
}

StatusOr<SendParamPtr> SendParamsToC(const RendezvousInterface::ParsedKey& key,
                                     const RendezvousInterface::Args& args,
                                     const Tensor& tensor, const bool is_dead) {
  TF_RendezvousSend_Params* params = new TF_RendezvousSend_Params();
  params->key = new TF_RendezvousParsedKey(ToC(key));
  params->args = new TF_RendezvousArgsStruct(ToC(args));
  params->is_dead = is_dead;
  params->status = TF_NewStatus();
  params->tensor = CopyTensorToTF_Tensor(tensor);
  return SendParamPtr(params, MakeSendParamDeleter());
}

struct RecvParamDeleter {
  void operator()(TF_RendezvousAsyncRecv_Params* params) {
    if (params == nullptr) {
      return;
    }
    TF_RendezvousParsedKey* key =
        const_cast<TF_RendezvousParsedKey*>(params->key);
    TF_RendezvousArgsStruct* args =
        const_cast<TF_RendezvousArgsStruct*>(params->args);
    Destroy(key);
    Destroy(args);
    Destroy(&params->on_done);
    delete params->key;
    delete params->args;
    delete params;
  }
};

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
    delete params->tensor;
    delete params;
  };
}

StatusOr<DoneCallbackParamPtr> DoneCallbackParamsToC(
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

void SendFunctionThunk(void* context, TF_RendezvousSend_Params* params) {
  using SendFunction = std::function<void(TF_RendezvousSend_Params*)>;
  auto* send_func = static_cast<SendFunction*>(context);
  (*send_func)(params);
}

// Use in `TF_RendezvousThunk ToC(tensorflow::RendezvousInterface* rendezvous)`
TF_RendezvousSenderImpl BindSendFunction(RendezvousInterface* rendezvous) {
  TF_RendezvousSenderImpl send_func;
  using SendFunction = std::function<void(TF_RendezvousSend_Params*)>;
  auto sender =
      new SendFunction([rendezvous](TF_RendezvousSend_Params* params) -> void {
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
      });
  send_func.context = static_cast<void*>(sender);
  send_func.send_func = SendFunctionThunk;
  return send_func;
}

void RecvFunctionThunk(void* context, TF_RendezvousAsyncRecv_Params* params) {
  using RecvFunction = std::function<void(TF_RendezvousAsyncRecv_Params*)>;
  auto* recv_func = static_cast<RecvFunction*>(context);
  (*recv_func)(params);
}

// Use in `TF_RendezvousThunk ToC(tensorflow::RendezvousInterface* rendezvous)`
TF_RendezvousAsyncRecverImpl BindAsyncRecvFunction(
    RendezvousInterface* rendezvous) {
  TF_RendezvousAsyncRecverImpl recv_func;
  using RecvFunction = std::function<void(TF_RendezvousAsyncRecv_Params*)>;
  auto recver = new RecvFunction(
      [rendezvous](TF_RendezvousAsyncRecv_Params* params) -> void {
        RendezvousInterface::ParsedKey key = FromC(*params->key);
        RendezvousInterface::Args args = FromC(*params->args);
        RendezvousInterface::DoneCallback on_done =
            [device_context = args.device_context, on_done = params->on_done](
                const Status& status,
                const RendezvousInterface::Args& send_args,
                const RendezvousInterface::Args& recv_args,
                const Tensor& tensor, const bool is_dead) {
              FromC(on_done)(status, send_args, recv_args, tensor, is_dead);
              // Releases TfCThunkDeviceContext allocated in FromC().
              device_context->Unref();
            };
        rendezvous->RecvAsync(key, args, on_done);
      });
  recv_func.context = static_cast<void*>(recver);
  recv_func.async_recv_func = RecvFunctionThunk;
  return recv_func;
}

void StartAbortFunctionThunk(void* context, const TF_Status* tf_status) {
  auto* rendezvous = static_cast<RendezvousInterface*>(context);
  absl::Status status = tsl::StatusFromTF_Status(tf_status);
  rendezvous->StartAbort(status);
}

// Use in `TF_RendezvousThunk ToC(tensorflow::RendezvousInterface* rendezvous)`
TF_RendezvousStartAbortImpl BindStartAborter(RendezvousInterface* rendezvous) {
  TF_RendezvousStartAbortImpl start_abort;
  start_abort.context = static_cast<void*>(rendezvous);
  start_abort.start_abort_func = StartAbortFunctionThunk;
  return start_abort;
}

}  // namespace

void Destroy(TF_RendezvousSenderImpl* send_func) {
  if (send_func == nullptr) {
    return;
  }
  if (send_func->context != nullptr) {
    auto runner = static_cast<std::function<void(TF_RendezvousSend_Params*)>*>(
        send_func->context);
    delete runner;
  }
}

void Destroy(TF_RendezvousAsyncRecverImpl* recv_func) {
  if (recv_func == nullptr) {
    return;
  }
  if (recv_func->context != nullptr) {
    auto runner =
        static_cast<std::function<void(TF_RendezvousAsyncRecv_Params*)>*>(
            recv_func->context);
    delete runner;
  }
}

void Destroy(TF_RendezvousStartAbortImpl* start_abort_func) {
  if (start_abort_func == nullptr) {
    return;
  }
}

namespace c_api {

Status TfCThunkRendezvous::Send(const ParsedKey& key, const Args& args,
                                const Tensor& val, const bool is_dead) {
  CHECK_OK_AND_ASSIGN(SendParamPtr params,
                      SendParamsToC(key, args, val, is_dead));
  const TF_RendezvousSenderImpl& sender = thunk_->send;
  sender.send_func(sender.context, params.get());
  return tsl::StatusFromTF_Status(params->status);
}

void TfCThunkRendezvous::RecvAsync(const ParsedKey& key, const Args& args,
                                   DoneCallback done) {
  TF_RendezvousAsyncRecv_Params* params = new TF_RendezvousAsyncRecv_Params();
  params->key = new TF_RendezvousParsedKey(ToC(key));
  params->args = new TF_RendezvousArgsStruct(ToC(args));
  params->on_done =
      ToC([done = std::move(done), params](
              const absl::Status& status,
              const tensorflow::RendezvousInterface::Args& send_args,
              const tensorflow::RendezvousInterface::Args& recv_args,
              const tensorflow::Tensor& tensor, bool is_dead) {
        done(status, send_args, recv_args, tensor, is_dead);
        RecvParamDeleter()(params);
      });

  const TF_RendezvousAsyncRecverImpl& async_recv = thunk_->async_recv;
  async_recv.async_recv_func(async_recv.context, params);
}

void TfCThunkRendezvous::StartAbort(const Status& status) {
  std::unique_ptr<TF_Status, std::function<void(TF_Status*)>> c_status(
      TF_NewStatus(), &TF_DeleteStatus);
  tsl::Set_TF_Status_from_Status(c_status.get(), status);
  const TF_RendezvousStartAbortImpl& start_abort = thunk_->start_abort;
  start_abort.start_abort_func(start_abort.context, c_status.get());
}

}  // namespace c_api

void DestroyOCParams::operator()(SE_OutsideCompilationParams* params) {
  if (params == nullptr) {
    return;
  }
  delete[] params->device_name;
  delete[] params->rendezvous_key;
  Destroy(params->rendezvous);
  delete params->rendezvous;
  if (params->host_transfers.size > 0) {
    StreamExecutor_Tpu_FreeSerializedProto(&params->host_transfers);
  }
  delete params;
}

}  // namespace tensorflow
