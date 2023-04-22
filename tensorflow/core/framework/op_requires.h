/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_FRAMEWORK_OP_REQUIRES_H_
#define TENSORFLOW_CORE_FRAMEWORK_OP_REQUIRES_H_

#include "tensorflow/core/platform/macros.h"

namespace tensorflow {

// Convenience macros for asserting and handling exceptional conditions.
// Analogous to the CHECK* macros provided by logging.h.
//
// Example use:
// void Compute(OperationContext* context) {
//   OP_REQUIRES(context, context->num_inputs() == 2,
//               errors::InvalidArgument("FooOp requires 2 arguments"));
//   ...
//   Status status = SomeUncertainMethod();
//   OP_REQUIRES_OK(context, status);
//   ...
// }
//
// These macros depend on CheckNotInComputeAsync, which must be defined before
// invoking the macro. We specifically don't include op_kernel.h from this
// header to reduce this header's dependencies. These macros may be used with
// alternative implementations of OpKernelContext with fewer dependencies.

#define OP_REQUIRES(CTX, EXP, STATUS)                     \
  do {                                                    \
    if (!TF_PREDICT_TRUE(EXP)) {                          \
      CheckNotInComputeAsync((CTX), "OP_REQUIRES_ASYNC"); \
      (CTX)->CtxFailure(__FILE__, __LINE__, (STATUS));    \
      return;                                             \
    }                                                     \
  } while (0)

#define OP_REQUIRES_OK(CTX, ...)                             \
  do {                                                       \
    ::tensorflow::Status _s(__VA_ARGS__);                    \
    if (!TF_PREDICT_TRUE(_s.ok())) {                         \
      CheckNotInComputeAsync((CTX), "OP_REQUIRES_OK_ASYNC"); \
      (CTX)->CtxFailureWithWarning(__FILE__, __LINE__, _s);  \
      return;                                                \
    }                                                        \
  } while (0)

#define OP_REQUIRES_OK_OR_SET_PAYLOAD(CTX, PAYLOAD_KEY, PAYLOAD_VALUE, STATUS) \
  do {                                                                         \
    if (!TF_PREDICT_TRUE(STATUS.ok())) {                                       \
      CheckNotInComputeAsync((CTX), "OP_REQUIRES_OK_ASYNC");                   \
      if (!PAYLOAD_VALUE.empty()) {                                            \
        STATUS.SetPayload(PAYLOAD_KEY, PAYLOAD_VALUE);                         \
      }                                                                        \
      (CTX)->CtxFailureWithWarning(__FILE__, __LINE__, STATUS);                \
      return;                                                                  \
    }                                                                          \
  } while (0)

#define OP_REQUIRES_ASYNC(CTX, EXP, STATUS, CALLBACK)  \
  do {                                                 \
    if (!TF_PREDICT_TRUE(EXP)) {                       \
      (CTX)->CtxFailure(__FILE__, __LINE__, (STATUS)); \
      (CALLBACK)();                                    \
      return;                                          \
    }                                                  \
  } while (0)

#define OP_REQUIRES_OK_ASYNC(CTX, STATUS, CALLBACK)         \
  do {                                                      \
    const ::tensorflow::Status& _s(STATUS);                 \
    if (!TF_PREDICT_TRUE(_s.ok())) {                        \
      (CTX)->CtxFailureWithWarning(__FILE__, __LINE__, _s); \
      (CALLBACK)();                                         \
      return;                                               \
    }                                                       \
  } while (0)

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_FRAMEWORK_OP_REQUIRES_H_
