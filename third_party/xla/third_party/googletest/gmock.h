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

#ifndef THIRD_PARTY_GOOGLETEST_GMOCK_H_
#define THIRD_PARTY_GOOGLETEST_GMOCK_H_

// gmock/gmock.h wrapper that also provides assert macros.
//
// These already exist in internal version of gmock, but upstream version
// doesn't have them. We use this wrapper to make dependency translation when
// exporting to OSS easier.

// Do not reorder the includes. status_matchers.h requires symbols from upstream
// gmock.h internally, and it will attempt to include this wrapper itself. This
// will only work when upstream gmock.h is included before absl
// status_matchers.h.
// clang-format off
#include "gmock/gmock.upstream.h"
// clang-format on
#include "absl/status/status_matchers.h"

// We build without ABSL_DEFINE_UNQUALIFIED_STATUS_TESTING_MACROS, so we need to
// define these ourselves.
//
// While we could #define it before including status_matchers.h in this file,
// this wouldn't work when the users include absl status_matchers.h before
// gmock.h - which will be the case when the includes are sorted alphabetically.
#define ASSERT_OK(expr) ASSERT_THAT((expr), ::testing::status::IsOk())
#define EXPECT_OK(expr) EXPECT_THAT((expr), ::testing::status::IsOk())

#define ASSERT_OK_AND_ASSIGN(lhs, rexpr) \
  __ASSERT_OK_AND_ASSIGN_IMPL(           \
      __STATUS_MACROS_CONCAT_NAME(_status_or_value, __COUNTER__), lhs, rexpr);

#define __ASSERT_OK_AND_ASSIGN_IMPL(statusor, lhs, rexpr) \
  auto statusor = (rexpr);                                \
  ASSERT_TRUE(statusor.status().ok())                     \
      << ADD_SOURCE_LOCATION(statusor.status());          \
  lhs = std::move(statusor).value()

#define __STATUS_MACROS_CONCAT_NAME(x, y) __STATUS_MACROS_CONCAT_IMPL(x, y)
#define __STATUS_MACROS_CONCAT_IMPL(x, y) x##y

#endif  // THIRD_PARTY_GOOGLETEST_GMOCK_H_
