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

#include <utility>

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
//   absl::Status status = SomeUncertainMethod();
//   OP_REQUIRES_OK(context, status);
//
//   // Or in one go:
//   OP_REQUIRES_OK(context, SomeUncertainMethod());
//   ...
// }
//
// The *_ASYNC versions take a CALLBACK macro argument which is called just
// before the return in the failure case; the expression in the macro itself
// is evaluated only in the failure case, and can therefore be expensive or
// have side effects that must not occur in the successful case. For example:
//
//   auto done = MakeCleanup([&]() { /* necessary continuation */ });
//   OP_REQUIRES_OK_ASYNC(context, SomeUncertainMethod(), done.release());
//   // `done` is still engaged if and only if control reaches here.
//
// These macros depend on CheckNotInComputeAsync and on absl::Status, both
// of which must be defined before invoking the macros. We specifically don't
// include op_kernel.h or the Abseil headers from this header to reduce this
// header's dependencies. These macros may be used with alternative
// implementations of OpKernelContext with fewer dependencies.

#define OP_REQUIRES(CTX, EXP, STATUS)                     \
  do {                                                    \
    if (!TF_PREDICT_TRUE(EXP)) {                          \
      CheckNotInComputeAsync((CTX), "OP_REQUIRES_ASYNC"); \
      (CTX)->CtxFailure(__FILE__, __LINE__, (STATUS));    \
      return;                                             \
    }                                                     \
  } while (0)

// The macro arguements passed to the ellipsis must combine to a single
// expression that is convertible to absl::Status. We accept a variable
// number of macro arguments only so as to support interior commas.
#define OP_REQUIRES_OK(CTX, ...)                                        \
  do {                                                                  \
    if (!TF_PREDICT_TRUE(                                               \
            ::tensorflow::op_requires_internal::OkImpl<::absl::Status>( \
                (CTX), __FILE__, __LINE__,                              \
                static_cast<const ::absl::Status&>(__VA_ARGS__)))) {    \
      return;                                                           \
    }                                                                   \
  } while (0)

#define OP_REQUIRES_OK_OR_SET_PAYLOAD(CTX, PAYLOAD_KEY, PAYLOAD_VALUE, STATUS) \
  do {                                                                         \
    if (!TF_PREDICT_TRUE(STATUS.ok())) {                                       \
      CheckNotInComputeAsync((CTX), "OP_REQUIRES_OK_ASYNC");                   \
      if (!PAYLOAD_VALUE.empty()) {                                            \
        STATUS.SetPayload(PAYLOAD_KEY, absl::Cord(PAYLOAD_VALUE));             \
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

#define OP_REQUIRES_OK_ASYNC(CTX, STATUS, CALLBACK)                          \
  do {                                                                       \
    if (!TF_PREDICT_TRUE(                                                    \
            ::tensorflow::op_requires_internal::OkAsyncImpl<::absl::Status>( \
                (CTX), __FILE__, __LINE__, (STATUS)))) {                     \
      (CALLBACK)();                                                          \
      return;                                                                \
    }                                                                        \
  } while (0)

#define OP_REQUIRES_VALUE(lhs, ctx, rexpr)                                   \
  OP_REQUIRES_VALUE_IMPL(                                                    \
      TF_STATUS_MACROS_CONCAT_NAME(_status_or_value, __COUNTER__), lhs, ctx, \
      rexpr)

#define OP_REQUIRES_VALUE_IMPL(statusor, lhs, ctx, rexpr) \
  auto statusor = (rexpr);                                \
  OP_REQUIRES_OK(ctx, statusor.status());                 \
  lhs = std::move(statusor.value())

// The "Impl" functions are implementation details for the above macros. They
// accept values constructed by the macros, and the values are guaranteed to
// be alive for the duration of the function call. Passing the macro arguments
// through a function call is important to support macro arguments that expand
// to short-lived values (which could not be bound to a reference directly).
//
// We use a template parameter S instead of the concrete type absl::Status
// so as to not require the inclusion of the Abseil header in this file.
// The header must be included before the macros are used.

namespace op_requires_internal {

// ctx is usually a plain pointer, but could be a smart pointer, so we accept it
// by const ref.
template <typename S, typename Ctx>
bool OkImpl(const Ctx& ctx, const char* file, int line, const S& s) {
  if (!TF_PREDICT_TRUE(s.ok())) {
    CheckNotInComputeAsync(ctx, "OP_REQUIRES_OK_ASYNC");
    ctx->CtxFailureWithWarning(file, line, s);
    return false;
  } else {
    return true;
  }
}

// ctx is usually a plain pointer, but could be a smart pointer, so we accept it
// by const ref.
template <typename S, typename Ctx>
bool OkAsyncImpl(const Ctx& ctx, const char* file, int line, const S& s) {
  if (!TF_PREDICT_TRUE(s.ok())) {
    ctx->CtxFailureWithWarning(file, line, s);
    return false;
  } else {
    return true;
  }
}

}  // namespace op_requires_internal
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_FRAMEWORK_OP_REQUIRES_H_
