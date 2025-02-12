/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

// StatusOr<T> is the union of a Status object and a T object. StatusOr models
// the concept of an object that is either a value, or an error Status
// explaining why such a value is not present. To this end, StatusOr<T> does not
// allow its Status value to be Status::OK.
//
// The primary use-case for StatusOr<T> is as the return value of a
// function which may fail.
//
// Example client usage for a StatusOr<T>, where T is not a pointer:
//
//  StatusOr<float> result = DoBigCalculationThatCouldFail();
//  if (result.ok()) {
//    float answer = result.value();
//    printf("Big calculation yielded: %f", answer);
//  } else {
//    LOG(ERROR) << result.status();
//  }
//
// Example client usage for a StatusOr<T*>:
//
//  StatusOr<Foo*> result = FooFactory::MakeNewFoo(arg);
//  if (result.ok()) {
//    std::unique_ptr<Foo> foo(result.value());
//    foo->DoSomethingCool();
//  } else {
//    LOG(ERROR) << result.status();
//  }
//
// Example client usage for a StatusOr<std::unique_ptr<T>>:
//
//  StatusOr<std::unique_ptr<Foo>> result = FooFactory::MakeNewFoo(arg);
//  if (result.ok()) {
//    std::unique_ptr<Foo> foo = std::move(result.value());
//    foo->DoSomethingCool();
//  } else {
//    LOG(ERROR) << result.status();
//  }
//
// Example factory implementation returning StatusOr<T*>:
//
//  StatusOr<Foo*> FooFactory::MakeNewFoo(int arg) {
//    if (arg <= 0) {
//      return tsl::InvalidArgument("Arg must be positive");
//    } else {
//      return new Foo(arg);
//    }
//  }
//
// Note that the assignment operators require that destroying the currently
// stored value cannot invalidate the argument; in other words, the argument
// cannot be an alias for the current value, or anything owned by the current
// value.
#ifndef XLA_TSL_PLATFORM_STATUSOR_H_
#define XLA_TSL_PLATFORM_STATUSOR_H_

#include "absl/base/attributes.h"
#include "absl/base/macros.h"
#include "absl/status/statusor.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/macros.h"
#include "xla/tsl/platform/status.h"
#include "tsl/platform/platform.h"

// Include appropriate platform-dependent `TF_ASSIGN_OR_RETURN`.
#if defined(PLATFORM_GOOGLE)
#include "xla/tsl/platform/google/statusor.h"  // IWYU pragma: export
#else
#include "xla/tsl/platform/default/statusor.h"  // IWYU pragma: export
#endif

// TODO: b/323943471 - This macro should eventually be provided by Abseil.
#ifndef ABSL_DEPRECATE_AND_INLINE
#define ABSL_DEPRECATE_AND_INLINE()
#endif

namespace tsl {

template <typename T>
using StatusOr ABSL_DEPRECATE_AND_INLINE() = absl::StatusOr<T>;

}  // namespace tsl

#define TF_ASSERT_OK_AND_ASSIGN(lhs, rexpr)                             \
  TF_ASSERT_OK_AND_ASSIGN_IMPL(                                         \
      TF_STATUS_MACROS_CONCAT_NAME(_status_or_value, __COUNTER__), lhs, \
      rexpr);

#define TF_ASSERT_OK_AND_ASSIGN_IMPL(statusor, lhs, rexpr)  \
  auto statusor = (rexpr);                                  \
  ASSERT_TRUE(statusor.status().ok()) << statusor.status(); \
  lhs = std::move(statusor).value()

#define TF_STATUS_MACROS_CONCAT_NAME(x, y) TF_STATUS_MACROS_CONCAT_IMPL(x, y)
#define TF_STATUS_MACROS_CONCAT_IMPL(x, y) x##y

#endif  // XLA_TSL_PLATFORM_STATUSOR_H_
