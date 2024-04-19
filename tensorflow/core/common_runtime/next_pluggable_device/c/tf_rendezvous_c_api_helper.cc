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

#include "tensorflow/core/common_runtime/next_pluggable_device/c/tf_rendezvous_c_api_helper.h"

#include <cstring>
#include <functional>
#include <memory>
#include <utility>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow/c/tf_status_helper.h"
#include "tensorflow/c/tf_tensor.h"
#include "tensorflow/core/common_runtime/next_pluggable_device/c/tf_device_context_c_api.h"
#include "tensorflow/core/common_runtime/next_pluggable_device/c/tf_device_context_c_api_helper.h"
#include "tensorflow/core/common_runtime/next_pluggable_device/c/tf_rendezvous_c_api.h"
#include "tensorflow/core/common_runtime/next_pluggable_device/c/tf_tensor_utils.h"
#include "tensorflow/core/framework/cancellation.h"
#include "tensorflow/core/framework/rendezvous.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/statusor.h"
#include "tsl/framework/allocator.h"

#define CONCAT_HELPER(a, b) a##b
#define CONCAT(a, b) CONCAT_HELPER(a, b)
#define CHECK_OK_AND_ASSIGN(lhs, expr)        \
  auto CONCAT(status_or_, __LINE__) = (expr); \
  CHECK(CONCAT(status_or_, __LINE__).ok());   \
  lhs = std::move(CONCAT(status_or_, __LINE__)).value();

namespace tensorflow {

namespace {

TFDevice_AllocatorAttributes ToC(const tsl::AllocatorAttributes& attributes) {
  TFDevice_AllocatorAttributes c_attributes;
  c_attributes.value = attributes.value;
  c_attributes.scope_id = attributes.scope_id;
  return c_attributes;
}

void Destroy(TFDevice_AllocatorAttributes* c_attributes) {}

static TFE_CancellationManager* ToC(CancellationManager* cancellation_manager) {
  return reinterpret_cast<TFE_CancellationManager*>(cancellation_manager);
}

TF_RendezvousArgsStruct ToC(const RendezvousInterface::Args& args) {
  TF_RendezvousArgsStruct c_args;
  c_args.device_context = DeviceContext_ToC(args.device_context);
  c_args.alloc_attrs = ToC(args.alloc_attrs);
  c_args.cancellation_manager = ToC(args.cancellation_manager);
  return c_args;
}

void Destroy(TF_RendezvousArgsStruct* c_args) {
  DeviceContext_Destroy(c_args->device_context);
  delete c_args->device_context;
  c_args->device_context = nullptr;
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

void Destroy(TF_RendezvousParsedKey* c_key) {
  delete[] c_key->full_key;
  c_key->full_key = nullptr;
}

using SendParamDeleter = std::function<void(TF_RendezvousSend_Params*)>;
using DoneCallbackParamDeleter =
    std::function<void(TF_RendezvousDoneCallback_Params*)>;

using SendParamPtr =
    std::unique_ptr<TF_RendezvousSend_Params, SendParamDeleter>;
using DoneCallbackParamPtr =
    std::unique_ptr<TF_RendezvousDoneCallback_Params, DoneCallbackParamDeleter>;

SendParamDeleter MakeSendParamDeleter();
absl::StatusOr<SendParamPtr> SendParamsToC(
    const RendezvousInterface::ParsedKey& key,
    const RendezvousInterface::Args& args, const Tensor& tensor, bool is_dead);

void RendezvousCallbackThunk(void* context,
                             TF_RendezvousDoneCallback_Params* params) {
  using CallbackType = std::function<void(TF_RendezvousDoneCallback_Params*)>;
  auto* callback = static_cast<CallbackType*>(context);
  (*callback)(params);
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
    c_on_done->context = nullptr;
  }
}

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
    params->key = nullptr;
    delete params->args;
    params->args = nullptr;
    TF_DeleteTensor(params->tensor);
    params->tensor = nullptr;
    TF_DeleteStatus(params->status);
    params->status = nullptr;
    delete params;
  };
}

absl::StatusOr<SendParamPtr> SendParamsToC(
    const RendezvousInterface::ParsedKey& key,
    const RendezvousInterface::Args& args, const Tensor& tensor,
    const bool is_dead) {
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
    delete params->key;
    params->key = nullptr;

    Destroy(args);
    delete params->args;
    params->args = nullptr;

    Destroy(&params->on_done);

    delete params;
  }
};

class TfCThunkRendezvous final : public ::tensorflow::RendezvousInterface {
 public:
  explicit TfCThunkRendezvous(const TF_RendezvousThunk& thunk)
      : thunk_(thunk) {}

  ~TfCThunkRendezvous() override = default;

  Status Send(const ParsedKey& key, const Args& args, const Tensor& val,
              bool is_dead) override;

  void RecvAsync(const ParsedKey& key, const Args& args,
                 DoneCallback done) override;

  void StartAbort(const Status& status) override;

 private:
  const TF_RendezvousThunk thunk_;
};

Status TfCThunkRendezvous::Send(const ParsedKey& key, const Args& args,
                                const Tensor& val, const bool is_dead) {
  CHECK_OK_AND_ASSIGN(SendParamPtr params,
                      SendParamsToC(key, args, val, is_dead));
  thunk_.send_func(thunk_.rendezvous, params.get());
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

  thunk_.async_recv_func(thunk_.rendezvous, params);
}

void TfCThunkRendezvous::StartAbort(const Status& status) {
  std::unique_ptr<TF_Status, std::function<void(TF_Status*)>> c_status(
      TF_NewStatus(), &TF_DeleteStatus);
  tsl::Set_TF_Status_from_Status(c_status.get(), status);
  thunk_.start_abort_func(thunk_.rendezvous, c_status.get());
}

}  // namespace

std::unique_ptr<tensorflow::RendezvousInterface> FromC(
    const TF_RendezvousThunk* thunk) {
  return std::make_unique<TfCThunkRendezvous>(*thunk);
}

}  // namespace tensorflow
