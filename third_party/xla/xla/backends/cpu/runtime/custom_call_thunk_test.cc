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

#include "xla/backends/cpu/runtime/custom_call_thunk.h"

#include <cstdint>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/status_matchers.h"
#include "xla/backends/cpu/runtime/thunk.h"
#include "xla/custom_options.h"
#include "xla/executable_run_options.h"
#include "xla/ffi/ffi.h"
#include "xla/service/hlo.pb.h"

namespace xla::cpu {

absl::Status ReadCustomOptions(int64_t attr, ffi::Dictionary options) {
  EXPECT_EQ(attr, 7);
  if (!options.contains("value")) {
    return absl::OkStatus();
  }
  ABSL_ASSIGN_OR_RETURN(int64_t value, options.get<int64_t>("value"));
  return absl::UnknownError(std::to_string(value));
}

XLA_FFI_DEFINE_HANDLER(
    kReadCustomOptions, ReadCustomOptions,
    ffi::Ffi::Bind().Attr<int64_t>("value").Ctx<ffi::CustomOptions>());
XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "cpu_test_custom_options", "Host",
                         kReadCustomOptions);

TEST(CustomCallThunkTest, ForwardsCustomOptionsOnEveryExecution) {
  ASSERT_OK_AND_ASSIGN(
      auto thunk,
      CustomCallThunk::Create({}, "cpu_test_custom_options", {},
                              "{value = 7 : i64}", API_VERSION_TYPED_FFI));
  Thunk::CustomCallExecuteParams custom_call_params(RunId{0}, 0, nullptr,
                                                    nullptr);
  Thunk::ExecuteParams params;
  params.custom_call_params = &custom_call_params;

  auto no_options = thunk->Execute(params);
  ASSERT_TRUE(no_options.IsAvailable());
  EXPECT_FALSE(no_options.IsError());

  xla::CustomOptions empty;
  params.custom_options = &empty;
  auto empty_options = thunk->Execute(params);
  ASSERT_TRUE(empty_options.IsAvailable());
  EXPECT_FALSE(empty_options.IsError());

  for (int64_t value : {42, 43}) {
    xla::CustomOptions options({{"value", value}});
    params.custom_options = &options;
    auto done = thunk->Execute(params);
    ASSERT_TRUE(done.IsAvailable());
    ASSERT_TRUE(done.IsError());
    EXPECT_THAT(done.GetError(),
                absl_testing::StatusIs(absl::StatusCode::kUnknown,
                                       std::to_string(value)));
  }
}

}  // namespace xla::cpu
