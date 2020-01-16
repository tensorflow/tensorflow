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
#ifndef TENSORFLOW_C_EXPERIMENTAL_RENDEZVOUS_INTERNAL_H_
#define TENSORFLOW_C_EXPERIMENTAL_RENDEZVOUS_INTERNAL_H_

#include <stddef.h>

#include "tensorflow/c/c_api.h"
#include "tensorflow/c/experimental/rendezvous.h"
#include "tensorflow/core/distributed_runtime/base_rendezvous_mgr.h"
#include "tensorflow/core/distributed_runtime/worker_env.h"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/framework/rendezvous.h"
#include "tensorflow/core/platform/macros.h"

struct TF_ParsedKey {
  // char* members might not be null-terminated.
  const char* src_device;
  size_t src_device_len;
  const char* dst_device;
  size_t dst_device_len;
  const char* full_key;
  size_t full_key_len;
};

struct TF_AllocatorAttributes {
  bool on_host;
  bool nic_compatible;
  // NOTE: The upper 8 bits of the value are reserved for
  // device-specific uses.  Implementors of a device can interpret these
  // upper 8 bits in device-specific ways, and ops implemented for those
  // devices are responsible for setting those 8 bits appropriately.
  tensorflow::uint32 value = 0;
  // EXPERIMENTAL: If this is greater than zero, then allocation is delegated to
  // a named special-purpose allocator on the same device.
  tensorflow::int32 scope_id = 0;
};

struct TF_DeviceContext {
  ::tensorflow::DeviceContext* context;
};

struct TF_RendezvousArgs {
  const TF_DeviceContext* device_context;
  const TF_AllocatorAttributes* alloc_attrs;
};

struct TF_RendezvousDoneCallback {
  ::tensorflow::Rendezvous::DoneCallback done_callback;

  // TODO(annarev): figure out if we should also support sent_args.
  const TF_RendezvousArgs* recv_args;
  TF_Tensor* tensor = nullptr;
  TF_Status* status;
  bool dead;
};

struct TF_RemoteRendezvousBuilder {
  void* (*init_function)(void* server_context);
  void (*receive_from_remote_async_function)(TF_ParsedKey*, TF_RendezvousArgs*,
                                             TF_RendezvousDoneCallback*,
                                             void* context);
  void (*delete_function)(void* context);
  void* server_context;
};

namespace tensorflow {

class CRemoteRendezvous : public BaseRemoteRendezvous {
 public:
  CRemoteRendezvous(const WorkerEnv* env, int64 step_id,
                    void (*receive_from_remote_async_function)(
                        TF_ParsedKey*, TF_RendezvousArgs*,
                        TF_RendezvousDoneCallback*, void* context),
                    void (*delete_function)(void* context),
                    void* server_context);

  void SetContext(void* context) { context_ = context; }

 protected:
  void RecvFromRemoteAsync(const Rendezvous::ParsedKey& parsed,
                           const Rendezvous::Args& args,
                           DoneCallback done) override;

 private:
  ~CRemoteRendezvous() override;

  void (*receive_from_remote_async_function_)(TF_ParsedKey*, TF_RendezvousArgs*,
                                              TF_RendezvousDoneCallback*,
                                              void* context);
  void (*delete_function_)(void* context);
  void* context_;
  TF_DISALLOW_COPY_AND_ASSIGN(CRemoteRendezvous);
};

class CRendezvousMgr : public BaseRendezvousMgr {
 public:
  CRendezvousMgr(const WorkerEnv* env,
                 const TF_RemoteRendezvousBuilder* rendezvous_builder)
      : BaseRendezvousMgr(env), rendezvous_builder_(rendezvous_builder) {}

 protected:
  BaseRemoteRendezvous* Create(int64 step_id,
                               const WorkerEnv* worker_env) override {
    auto* rendezvous = new CRemoteRendezvous(
        worker_env, step_id,
        rendezvous_builder_->receive_from_remote_async_function,
        rendezvous_builder_->delete_function,
        rendezvous_builder_->server_context);

    rendezvous->SetContext(rendezvous_builder_->init_function(
        rendezvous_builder_->server_context));
    return rendezvous;
  }

 private:
  const TF_RemoteRendezvousBuilder* rendezvous_builder_;
  TF_DISALLOW_COPY_AND_ASSIGN(CRendezvousMgr);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_C_EXPERIMENTAL_RENDEZVOUS_INTERNAL_H_
