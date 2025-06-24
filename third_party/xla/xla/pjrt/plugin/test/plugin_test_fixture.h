/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_PJRT_PLUGIN_TEST_PLUGIN_TEST_FIXTURE_H_
#define XLA_PJRT_PLUGIN_TEST_PLUGIN_TEST_FIXTURE_H_

#include <memory>
#include <string>

#include <gtest/gtest.h>
#include "absl/status/statusor.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/pjrt_api.h"
#include "xla/pjrt/pjrt_c_api_client.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {

absl::StatusOr<std::string> GetRegisteredPluginName();

class PluginTestFixture : public ::testing::Test {
 public:
  void SetUp() override {
    TF_ASSERT_OK_AND_ASSIGN(plugin_name_, GetRegisteredPluginName());
    TF_ASSERT_OK_AND_ASSIGN(api_, pjrt::PjrtApi(plugin_name_));

    TF_ASSERT_OK_AND_ASSIGN(client_, GetCApiClient(plugin_name_));
  }

 protected:
  std::string plugin_name_;
  std::unique_ptr<PjRtClient> client_;
  const PJRT_Api* api_;
};

}  // namespace xla

#endif  // XLA_PJRT_PLUGIN_TEST_PLUGIN_TEST_FIXTURE_H_
