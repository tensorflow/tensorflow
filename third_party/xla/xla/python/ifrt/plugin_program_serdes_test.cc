// Copyright 2024 The OpenXLA Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <memory>

#include <gtest/gtest.h>
#include "xla/python/ifrt/plugin_program.h"
#include "xla/python/ifrt/serdes.h"
#include "xla/python/ifrt/serdes.pb.h"
#include "tsl/lib/core/status_test_util.h"
#include "tsl/platform/statusor.h"
#include "tsl/protobuf/error_codes.pb.h"
#include "tsl/protobuf/status.pb.h"

namespace xla {
namespace ifrt {
namespace {

TEST(PluginProgramSerDesTest, RoundTrip) {
  PluginProgram orig;
  orig.data = "foo";
  TF_ASSERT_OK_AND_ASSIGN(Serialized serialized, Serialize(orig));
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<PluginProgram> deserialized_program,
      Deserialize<PluginProgram>(serialized, /*options=*/nullptr));

  EXPECT_EQ(deserialized_program->data, "foo");
}

TEST(PluginCompileOptionsSerDesTest, RoundTrip) {
  PluginCompileOptions orig;
  TF_ASSERT_OK_AND_ASSIGN(Serialized serialized, Serialize(orig));
  TF_EXPECT_OK(
      Deserialize<PluginCompileOptions>(serialized, /*options=*/nullptr)
          .status());
}

}  // namespace
}  // namespace ifrt
}  // namespace xla
