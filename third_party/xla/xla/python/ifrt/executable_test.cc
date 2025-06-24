/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/python/ifrt/executable.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xla/python/ifrt/attribute_map.h"
#include "xla/python/ifrt/execute_options.pb.h"
#include "xla/python/ifrt/serdes_test_util.h"
#include "xla/python/ifrt/serdes_version.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"

namespace xla {
namespace ifrt {

using ::testing::Pair;
using ::testing::UnorderedElementsAre;

class ExecuteOptionsSerDesTest : public testing::TestWithParam<SerDesVersion> {
 public:
  ExecuteOptionsSerDesTest() : version_(GetParam()) {}

  SerDesVersion version() const { return version_; }

 private:
  SerDesVersion version_;
};

TEST_P(ExecuteOptionsSerDesTest, RoundTrip) {
  LoadedExecutable::ExecuteOptions options;
  options.launch_id = 1234;
  options.non_donatable_input_indices.insert(0);
  options.non_donatable_input_indices.insert(3);
  options.fill_status = true;
  options.custom_options = AttributeMap(
      AttributeMap::Map({{"foo", AttributeMap::StringValue("bar")}}));
  TF_ASSERT_OK_AND_ASSIGN(ExecuteOptionsProto serialized,
                          options.ToProto(version()));
  TF_ASSERT_OK_AND_ASSIGN(
      auto deserialized,
      LoadedExecutable::ExecuteOptions::FromProto(serialized));
  EXPECT_EQ(deserialized.launch_id, 1234);
  EXPECT_THAT(deserialized.non_donatable_input_indices,
              UnorderedElementsAre(0, 3));
  EXPECT_TRUE(deserialized.fill_status);
  ASSERT_TRUE(deserialized.custom_options.has_value());
  EXPECT_THAT(
      deserialized.custom_options->map(),
      UnorderedElementsAre(Pair("foo", AttributeMap::StringValue("bar"))));
}

INSTANTIATE_TEST_SUITE_P(
    SerDesVersion, ExecuteOptionsSerDesTest,
    testing::ValuesIn(test_util::AllSupportedSerDesVersions()));

}  // namespace ifrt
}  // namespace xla
