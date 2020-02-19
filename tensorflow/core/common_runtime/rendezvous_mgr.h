/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_RENDEZVOUS_MGR_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_RENDEZVOUS_MGR_H_

#include <string>
#include <unordered_map>

#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/framework/local_rendezvous.h"
#include "tensorflow/core/framework/rendezvous.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

// The IntraProcessRendezvous classes are implementations of a Rendezvous that
// expects all producers and consumers to be devices immediately accessible
// within the process. That is, it will never be necessary to perform an RPC to
// communicate with either.
//
// Buffering of Tensor values is delegated to a `LocalRendezvous`. An
// IntraProcessRendezvous. just adds functionality to coordinate multiple
// process-local devices.

// Reference-counted implementation that may be shared between multiple threads.
class RefCountedIntraProcessRendezvous : public Rendezvous {
 public:
  explicit RefCountedIntraProcessRendezvous(const DeviceMgr* device_mgr);

  // Implementation of RendezvousInterface methods.
  Status Send(const ParsedKey& key, const Rendezvous::Args& args,
              const Tensor& val, const bool is_dead) override;
  void RecvAsync(const ParsedKey& key, const Rendezvous::Args& args,
                 DoneCallback done) override;
  void StartAbort(const Status& status) override;

 private:
  const DeviceMgr* device_mgr_;
  LocalRendezvous local_;

  ~RefCountedIntraProcessRendezvous() override;

  TF_DISALLOW_COPY_AND_ASSIGN(RefCountedIntraProcessRendezvous);
};

// RefCountedIntraProcessRendezvous is aliased to IntraProcessRendezvous for
// backwards compatibility with existing users.
using IntraProcessRendezvous = RefCountedIntraProcessRendezvous;

// Non-reference-counted implementation that may be stack-allocated for
// performance.
//
// Prefer to use PrivateIntraProcessRendezvous in new code.
class PrivateIntraProcessRendezvous : public RendezvousInterface {
 public:
  explicit PrivateIntraProcessRendezvous(const DeviceMgr* device_mgr);
  ~PrivateIntraProcessRendezvous() override;

  // Implementation of RendezvousInterface methods.
  Status Send(const ParsedKey& key, const Rendezvous::Args& args,
              const Tensor& val, const bool is_dead) override;
  void RecvAsync(const ParsedKey& key, const Rendezvous::Args& args,
                 DoneCallback done) override;
  void StartAbort(const Status& status) override;

 private:
  const DeviceMgr* device_mgr_;
  LocalRendezvous local_;

  TF_DISALLOW_COPY_AND_ASSIGN(PrivateIntraProcessRendezvous);
};

}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_RENDEZVOUS_MGR_H_
