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

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_NEXT_PLUGGABLE_DEVICE_C_TF_RENDEZVOUS_C_API_CONVERSIONS_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_NEXT_PLUGGABLE_DEVICE_C_TF_RENDEZVOUS_C_API_CONVERSIONS_H_

#include <memory>

#include "tensorflow/core/common_runtime/next_pluggable_device/c/outside_compilation_params.h"  // IWYU pragma: keep
#include "tensorflow/core/common_runtime/next_pluggable_device/c/tf_rendezvous_c_api.h"
#include "tensorflow/core/framework/rendezvous.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/tsl/framework/allocator.h"

namespace tensorflow {

namespace c_api {

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

}  // namespace c_api

TFDevice_AllocatorAttributes ToC(const tsl::AllocatorAttributes& attributes);
tsl::AllocatorAttributes FromC(
    const TFDevice_AllocatorAttributes& c_attributes);
void Destroy(TFDevice_AllocatorAttributes* c_attributes);

TF_RendezvousArgsStruct ToC(const tensorflow::RendezvousInterface::Args& args);
tensorflow::RendezvousInterface::Args FromC(
    const TF_RendezvousArgsStruct& c_args);
void Destroy(TF_RendezvousArgsStruct* c_args);

TF_RendezvousParsedKey ToC(
    const tensorflow::RendezvousInterface::ParsedKey& key);
tensorflow::RendezvousInterface::ParsedKey FromC(
    const TF_RendezvousParsedKey& c_key);
void Destroy(TF_RendezvousParsedKey* c_key);

TF_RendezvousDoneCallbackImpl ToC(
    const tensorflow::RendezvousInterface::DoneCallback& on_done);
tensorflow::RendezvousInterface::DoneCallback FromC(
    const TF_RendezvousDoneCallbackImpl& c_on_done);
void Destroy(TF_RendezvousDoneCallbackImpl* c_on_done);

TF_RendezvousThunk* ToC(tensorflow::RendezvousInterface* rendezvous);
// `tensorflow::RendezvousInterface` has a protected destructor, so this
// function can't return std::unique_ptr<tensorflow::RendezvousInterface>.
std::unique_ptr<tensorflow::c_api::TfCThunkRendezvous> FromC(
    const TF_RendezvousThunk* thunk);
void Destroy(TF_RendezvousThunk* thunk);

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_NEXT_PLUGGABLE_DEVICE_C_TF_RENDEZVOUS_C_API_CONVERSIONS_H_
