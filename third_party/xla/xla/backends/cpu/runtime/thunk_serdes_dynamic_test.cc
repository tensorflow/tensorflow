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

#include <string>

#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_replace.h"
#include "xla/backends/cpu/runtime/thunk.h"
#include "xla/backends/cpu/runtime/thunk.pb.h"
#include "xla/backends/cpu/runtime/thunk_proto_serdes.h"
#include "xla/tsl/platform/test.h"

namespace xla::cpu {
namespace {

class ThunkSerDesDynamicTest : public ::testing::TestWithParam<Thunk::Kind> {};

TEST_P(ThunkSerDesDynamicTest, ThunkFailsWhenNotLinked) {
  Thunk::Kind kind = GetParam();
  auto from_proto_fn_or = ThunkSerDesRegistry::Get().GetFromProtoFn(kind);

  EXPECT_FALSE(from_proto_fn_or.ok());
  EXPECT_EQ(from_proto_fn_or.status().code(), absl::StatusCode::kNotFound);
  EXPECT_PRED_FORMAT2(
      testing::IsSubstring,
      absl::StrCat("No FromProto function registered for thunk kind: ",
                   Thunk::KindToString(kind)),
      std::string(from_proto_fn_or.status().message()));
}

INSTANTIATE_TEST_SUITE_P(
    ThunkSerDesDynamicTestSuite, ThunkSerDesDynamicTest,
    ::testing::Values(Thunk::Kind::kConvolution, Thunk::Kind::kDot,
                      Thunk::Kind::kYnnFusion, Thunk::Kind::kCollective,
                      Thunk::Kind::kFft, Thunk::Kind::kCustomCall,
                      Thunk::Kind::kCopy),
    [](const ::testing::TestParamInfo<ThunkSerDesDynamicTest::ParamType>&
           info) {
      return absl::StrReplaceAll(Thunk::KindToString(info.param), {{"-", "_"}});
    });

}  // namespace
}  // namespace xla::cpu
