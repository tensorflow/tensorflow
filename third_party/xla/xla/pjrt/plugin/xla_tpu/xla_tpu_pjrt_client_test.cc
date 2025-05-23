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

#include "xla/pjrt/plugin/xla_tpu/xla_tpu_pjrt_client.h"

#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"
#include "xla/pjrt/pjrt_common.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {

TEST(XlaCpuPjrtClientTest, GetXlaPjrtTpuClientWithDefaultOptions) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, GetXlaPjrtTpuClient());
  EXPECT_EQ(client->platform_name(), "tpu");
}

TEST(XlaCpuPjrtClientTest, GetXlaPjrtTpuClientWithInvalidOptions) {
  absl::flat_hash_map<std::string, PjRtValueType> create_options;
  create_options.insert({"invalid_option", true});

  TF_ASSERT_OK_AND_ASSIGN(auto client, GetXlaPjrtTpuClient());
  EXPECT_EQ(client->platform_name(), "tpu");
}

}  // namespace xla
