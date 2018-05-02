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

#ifndef TENSORFLOW_CORE_UTIL_RPC_RPC_FACTORY_H_
#define TENSORFLOW_CORE_UTIL_RPC_RPC_FACTORY_H_

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_types.h"

namespace tensorflow {

// Return the environment variable `key`.  If the variable is not set,
// use the default value.  If it is set but could not be parsed,
// return `false`.  Otherwise set `value` and return `true`.
template <typename T>
bool GetEnvVar(const char* key, const T& default_value, T* value);

class RPCFactory {
 public:
  RPCFactory() {}
  virtual ~RPCFactory() {}

  // Asynchronously invokes methods `method_t` at addresses `address_t` with
  // request strings from `request_t`.  Any of these may be scalar
  // Tensors, in which case the operands are broadcasted.
  // Upon completion of all requests, `response_t` will be populated and the
  // `done` callback will be invoked.
  //
  // If `try_rpc` is `true`, then `status_message_t` and
  // `status_code_t` will be populated as well.
  //
  // If `try_rpc` is `false`, then `status_message_t` and
  // `status_code_t` are ignored (and may be nullptr).  Instead, the
  // status of any failed call will be propagated to the op.
  //
  // REQUIRES:
  //   - `response_t` is not null, and is a string Tensor with the same shape as
  //     `request_t`.
  //
  //   If `try_rpc` is `true`:
  //      - `status_code_t` and `status_message_t` are not null.
  //      - `status_code_t` is an int32 Tensor with the same shape as
  //        `request_t`.
  //      - `status_message_t` is a string Tensor with the same shape as
  //        `request_t`.
  virtual void Call(OpKernelContext* ctx, int64 num_elements,
                    const Tensor& address_t, const Tensor& method_t,
                    const Tensor& request_t, const bool try_rpc,
                    Tensor* response_t, Tensor* status_code_t,
                    Tensor* status_message_t,
                    AsyncOpKernel::DoneCallback done) = 0;

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(RPCFactory);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_UTIL_RPC_RPC_FACTORY_H_
