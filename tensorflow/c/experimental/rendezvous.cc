/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/c/experimental/rendezvous.h"

#include <functional>

#include "tensorflow/c/c_api.h"
#include "tensorflow/c/c_api_internal.h"
#include "tensorflow/c/experimental/rendezvous_internal.h"
#include "tensorflow/core/distributed_runtime/base_rendezvous_mgr.h"
#include "tensorflow/core/distributed_runtime/worker_env.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/rendezvous.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/stringpiece.h"

namespace tensorflow {

CRemoteRendezvous::CRemoteRendezvous(const WorkerEnv* env, int64 step_id,
                                     void (*receive_from_remote_async_function)(
                                         TF_ParsedKey*, TF_RendezvousArgs*,
                                         TF_RendezvousDoneCallback*,
                                         void* context),
                                     void (*delete_function)(void* context),
                                     void* server_context)
    : BaseRemoteRendezvous(env, step_id),
      receive_from_remote_async_function_(receive_from_remote_async_function),
      delete_function_(delete_function),
      context_(nullptr) {}

void CRemoteRendezvous::RecvFromRemoteAsync(const Rendezvous::ParsedKey& parsed,
                                            const Rendezvous::Args& args,
                                            DoneCallback done) {
  if (args.cancellation_manager != nullptr) {
    VLOG(1) << "WARNING: CRemoteRendezvous does not support cancellation.";
  }
  TF_ParsedKey key;
  key.src_device = parsed.src_device.data();
  key.src_device_len = parsed.src_device.size();
  key.dst_device = parsed.dst_device.data();
  key.dst_device_len = parsed.dst_device.size();
  key.full_key = parsed.FullKey().data();
  key.full_key_len = parsed.FullKey().size();

  TF_DeviceContext* device_context = new TF_DeviceContext();
  device_context->context = args.device_context;

  TF_AllocatorAttributes* alloc_attrs = new TF_AllocatorAttributes();
  alloc_attrs->value = args.alloc_attrs.value;
  alloc_attrs->scope_id = args.alloc_attrs.scope_id;
  alloc_attrs->on_host = args.alloc_attrs.on_host();
  alloc_attrs->nic_compatible = args.alloc_attrs.nic_compatible();

  TF_RendezvousArgs* cargs = new TF_RendezvousArgs();
  cargs->device_context = device_context;
  cargs->alloc_attrs = alloc_attrs;

  TF_RendezvousDoneCallback* done_callback = new TF_RendezvousDoneCallback();
  done_callback->done_callback = done;
  done_callback->recv_args = cargs;

  receive_from_remote_async_function_(&key, cargs, done_callback, context_);
}

CRemoteRendezvous::~CRemoteRendezvous() { delete_function_(context_); }
}  // namespace tensorflow

TF_RemoteRendezvousBuilder* TF_NewRemoteRendezvousBuilder(
    void* (*init_function)(void* server_context),
    void (*receive_from_remote_async_function)(TF_ParsedKey*,
                                               TF_RendezvousArgs*,
                                               TF_RendezvousDoneCallback*,
                                               void* context),
    void (*delete_function)(void* context)) {
  TF_RemoteRendezvousBuilder* builder = new TF_RemoteRendezvousBuilder();
  builder->init_function = init_function;
  builder->delete_function = delete_function;
  builder->receive_from_remote_async_function =
      receive_from_remote_async_function;
  return builder;
}

void TF_DeleteRemoteRendezvousBuilder(
    TF_RemoteRendezvousBuilder* rendezvous_builder) {
  DCHECK_NE(rendezvous_builder, nullptr);
  delete rendezvous_builder;
}

TF_CAPI_EXPORT extern void TF_RendezvousDone(
    TF_RendezvousDoneCallback* callback) {
  DCHECK_NE(callback, nullptr);
  ::tensorflow::Tensor tensor;
  TF_CHECK_OK(TF_TensorToTensor(callback->tensor, &tensor));
  ::tensorflow::Rendezvous::Args recv_args;
  recv_args.alloc_attrs.value = callback->recv_args->alloc_attrs->value;
  recv_args.alloc_attrs.scope_id = callback->recv_args->alloc_attrs->scope_id;
  recv_args.device_context = callback->recv_args->device_context->context;
  ::tensorflow::Rendezvous::Args sent_args;

  callback->done_callback(callback->status->status, sent_args, recv_args,
                          tensor, callback->dead);

  if (callback->recv_args) {
    DCHECK_NE(callback->recv_args, nullptr);
    DCHECK_NE(callback->recv_args->alloc_attrs, nullptr);
    DCHECK_NE(callback->recv_args->device_context, nullptr);
    delete callback->recv_args->alloc_attrs;
    delete callback->recv_args->device_context;
    delete callback->recv_args;
  }
  delete callback;
  callback = nullptr;
}
