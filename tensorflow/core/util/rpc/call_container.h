/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_UTIL_RPC_CALL_CONTAINER_H_
#define TENSORFLOW_CORE_UTIL_RPC_CALL_CONTAINER_H_

#include <list>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/refcount.h"
#include "tensorflow/core/util/reffed_status_callback.h"

namespace tensorflow {

template <typename Call>
class CallContainer {
 public:
  explicit CallContainer(OpKernelContext* ctx, int num_calls, bool fail_fast,
                         bool try_rpc, AsyncOpKernel::DoneCallback done,
                         CancellationToken token)
      : ctx_(ctx),
        done_(std::move(done)),
        token_(token),
        fail_fast_(fail_fast),
        try_rpc_(try_rpc) {
    CHECK_GT(num_calls, 0);

    // This will run when all RPCs are finished.
    reffed_status_callback_ = new ReffedStatusCallback([this](const Status& s) {
      ctx_->cancellation_manager()->DeregisterCallback(token_);
      ctx_->SetStatus(s);
      done_();
      delete this;
    });

    // Subtract reference count from the initial creation.
    core::ScopedUnref unref(reffed_status_callback_);

    for (int i = 0; i < num_calls; ++i) {
      // Increase the reference on the callback for each new RPC.
      reffed_status_callback_->Ref();
    }
  }

  std::list<Call>* calls() { return &calls_; }

  void StartCancel() {
    // Once this loop is done, can no longer assume anything is valid
    // because "delete this" may have been immediately called.
    // Nothing should run after this loop.
    for (auto& call : calls_) {
      call.StartCancel();
    }
  }

  void Done(const Status& s, int index) {
    if (!try_rpc_) {
      reffed_status_callback_->UpdateStatus(s);
    }
    reffed_status_callback_->Unref();
  }

 private:
  OpKernelContext* ctx_;
  std::list<Call> calls_;
  const AsyncOpKernel::DoneCallback done_;
  const CancellationToken token_;
  const bool fail_fast_;
  const bool try_rpc_;

  // Performs its own reference counting.
  ReffedStatusCallback* reffed_status_callback_;
};

}  // namespace tensorflow
#endif  // TENSORFLOW_CORE_UTIL_RPC_CALL_CONTAINER_H_
