/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/stream_executor/sycl/onemkl_util.h"

#include "oneapi/mkl.hpp"
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "xla/tsl/platform/test.h"

namespace stream_executor::sycl {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::testing::HasSubstr;

TEST(ExecMklFuncTest, VoidCallReturnsOkStatus) {
  bool called = false;
  EXPECT_THAT(ExecMklFunc([&] { called = true; }), IsOk());
  EXPECT_TRUE(called);
}

TEST(ExecMklFuncTest, ReturnValueIsForwarded) {
  EXPECT_THAT(ExecMklFunc([] { return 42; }), IsOkAndHolds(42));
}

TEST(ExecMklFuncTest, MklExceptionBecomesInternalError) {
  absl::Status status = ExecMklFunc(
      [] { throw oneapi::mkl::exception("blas", "trsm", "lda too small"); });
  EXPECT_THAT(status, StatusIs(absl::StatusCode::kInternal,
                               HasSubstr("Mkl exception: oneapi::mkl::blas::"
                                         "trsm: lda too small")));
}

}  // namespace
}  // namespace stream_executor::sycl
