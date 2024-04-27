/*
 * Copyright 2023 The OpenXLA Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "xla/python/ifrt_proxy/server/version.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "tsl/platform/status_matchers.h"

namespace xla {
namespace ifrt {
namespace proxy {
namespace {

using ::tsl::testing::IsOk;
using ::tsl::testing::StatusIs;

struct Param {
  int client_min_version;
  int client_max_version;
  int server_min_version;
  int server_max_version;
};

class CompatibleVersionTest : public ::testing::TestWithParam<Param> {};

TEST_P(CompatibleVersionTest, Verify) {
  const Param& param = GetParam();
  EXPECT_THAT(ChooseVersion(param.client_min_version, param.client_max_version,
                            param.server_min_version, param.server_max_version),
              IsOk());
}

INSTANTIATE_TEST_SUITE_P(CompatibleVersionTest, CompatibleVersionTest,
                         ::testing::Values(Param{1, 1, 1, 1}, Param{1, 2, 2, 2},
                                           Param{2, 2, 1, 2},
                                           Param{1, 3, 3, 4}));

class IncompatibleVersionTest : public ::testing::TestWithParam<Param> {};

TEST_P(IncompatibleVersionTest, Verify) {
  const Param& param = GetParam();
  EXPECT_THAT(ChooseVersion(param.client_min_version, param.client_max_version,
                            param.server_min_version, param.server_max_version),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

INSTANTIATE_TEST_SUITE_P(IncompatibleVersionTest, IncompatibleVersionTest,
                         ::testing::Values(Param{1, 2, 3, 3}, Param{1, 3, 4, 6},
                                           Param{1, 1, 2, 2}));

}  // namespace
}  // namespace proxy
}  // namespace ifrt
}  // namespace xla
