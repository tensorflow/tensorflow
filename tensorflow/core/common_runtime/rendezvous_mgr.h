/* Copyright 2015 Google Inc. All Rights Reserved.

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

#ifndef TENSORFLOW_COMMON_RUNTIME_RENDEZVOUS_MGR_H_
#define TENSORFLOW_COMMON_RUNTIME_RENDEZVOUS_MGR_H_

#include <string>
#include <unordered_map>

#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/framework/rendezvous.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

// IntraProcessRendezvous is a Rendezvous which expects all producers
// and consumers to be devices immediately accessible within the
// process.  That is, it will never be necessary to perform an RPC to
// communicate with either.
//
// Buffering of Tensor values is delegated to a "local" Rendezvous
// obtained from NewLocalRendezvous().  This class just adds
// functionality to coordinate multiple process-local devices.
class IntraProcessRendezvous : public Rendezvous {
 public:
  explicit IntraProcessRendezvous(const DeviceMgr* device_mgr);

  // Forwards to local_, where the Tensor "val" will be buffered and
  // any waiting callback stored.
  Status Send(const string& key, const Rendezvous::Args& args,
              const Tensor& val, const bool is_dead) override;

  // This method is called only by the RecvOp.  It tests to see
  // whether the value will be produced by a local or remote device
  // and handles accordingly.  In the local case it forwards to
  // local_, in the remote case it initiates an RPC request.
  void RecvAsync(const string& key, const Rendezvous::Args& args,
                 DoneCallback done) override;

  void StartAbort(const Status& status) override;

 private:
  const DeviceMgr* device_mgr_;
  Rendezvous* local_;  // Owns a Ref on this object.

  mutable mutex mu_;

  // Status given by StartAbort() if any.
  Status status_ GUARDED_BY(mu_);

  ~IntraProcessRendezvous() override;

  // Parses "key" into "parsed". If "is_src" is true, checks that the
  // rendezvous key's source is in this process. If "is_src" is false,
  // checks that the rendezvous key's destination is in this process.
  Status ParseKey(const string& key, bool is_src,
                  Rendezvous::ParsedKey* parsed);

  // Callback handling the case when a rendezvous has been
  // accomplished in local_ and the consumer is local to this process.
  // Tensor "in" will be copied into "out". The key "parsed" encodes
  // the src and dst devices.
  typedef std::function<void(const Status&)> StatusCallback;
  void SameWorkerRecvDone(const Rendezvous::ParsedKey& parsed,
                          const Rendezvous::Args& send_args,
                          const Rendezvous::Args& recv_args, const Tensor& in,
                          Tensor* out, StatusCallback done);

  TF_DISALLOW_COPY_AND_ASSIGN(IntraProcessRendezvous);
};

}  // end namespace tensorflow

#endif  // TENSORFLOW_COMMON_RUNTIME_RENDEZVOUS_MGR_H_
