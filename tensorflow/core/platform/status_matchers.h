/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_PLATFORM_STATUS_MATCHERS_H_
#define TENSORFLOW_CORE_PLATFORM_STATUS_MATCHERS_H_


#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/protobuf/error_codes.pb.h"
#include "tensorflow/tsl/platform/status_matchers.h"

namespace tensorflow {
// NOLINTBEGIN(misc-unused-using-decls)

using tsl::PrintTo;

namespace error {
using tsl::error::PrintTo;
}  // namespace error

namespace testing {
namespace internal_status {
using tsl::testing::internal_status::GetStatus;
using tsl::testing::internal_status::IsOkAndHoldsMatcher;
using tsl::testing::internal_status::IsOkAndHoldsMatcherImpl;
using tsl::testing::internal_status::IsOkMatcher;
using tsl::testing::internal_status::MonoIsOkMatcherImpl;
using tsl::testing::internal_status::MonoStatusIsMatcherImpl;
using tsl::testing::internal_status::StatusIsMatcher;
using tsl::testing::internal_status::StatusIsMatcherCommonImpl;
}  // namespace internal_status
using tsl::testing::IsOk;
using tsl::testing::IsOkAndHolds;
using tsl::testing::StatusIs;
// NOLINTEND(misc-unused-using-decls)
}  // namespace testing
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PLATFORM_STATUS_MATCHERS_H_
