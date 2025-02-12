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

#ifndef XLA_TSL_DISTRIBUTED_RUNTIME_CALL_OPTIONS_H_
#define XLA_TSL_DISTRIBUTED_RUNTIME_CALL_OPTIONS_H_

#include <functional>

#include "xla/tsl/platform/macros.h"
#include "xla/tsl/platform/types.h"
#include "tsl/platform/mutex.h"
#include "tsl/platform/thread_annotations.h"

namespace tsl {

// Options passed to interface calls. This class provides portable
// functionality across different RPC systems on top of
// platform-specific mechanisms (for client and server contexts,
// cancellation, etc.).
//
// TODO(zhifengc): Maybe change all RPC methods to take CallOptions.
class CallOptions {
 public:
  CallOptions();

  // Cancellation.
  //
  // The caller may call StartCancel() anytime as long as this
  // CallOptions object is alive. The callee may or may not receive
  // the cancellation notification depending on the rpc layer
  // implementation.
  void StartCancel();

  // The callee (the rpc layer implementation) must set a cancellation
  // notifier before its blocking operation and clear the notifier
  // before the call returns.
  //
  // "cancel_func" may be called zero, once or more time. Therefore, it
  // should _not_ be responsible for memory management of any objects.
  //
  // "cancel_func" must be very light-weight. It should not block on
  // IO or locking. Typically, it just calls the rpc implementation
  // layer's specific cancellation mechanism and does nothing else.
  //
  // NOTE: "cancel_func" itself is pass-by-value. Therefore, we do not
  // worry about its ownership here.
  typedef std::function<void()> CancelFunction;
  void SetCancelCallback(CancelFunction cancel_func);
  void ClearCancelCallback();

  // Get and set operation timeout. Timeout value is in milliseconds.
  //
  // Default: 0. indicating there is no timeout for this call.
  int64_t GetTimeout();
  void SetTimeout(int64_t ms);

 private:
  mutex mu_;
  CancelFunction cancel_func_ TF_GUARDED_BY(mu_);

  // RPC operation timeout in milliseconds.
  int64_t timeout_in_ms_ TF_GUARDED_BY(mu_) = 0;

  CallOptions(const CallOptions&) = delete;
  void operator=(const CallOptions&) = delete;
};

}  // namespace tsl

#endif  // XLA_TSL_DISTRIBUTED_RUNTIME_CALL_OPTIONS_H_
