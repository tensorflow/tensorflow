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

#include "xla/pjrt/plugin/xla_gpu/xla_gpu_pjrt_client.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xla/pjrt/gpu/se_gpu_pjrt_client.h"
#include "tsl/platform/test.h"

namespace xla {

TEST(XlaCpuPjrtClientTest, GetXlaPjrtGpuClient) {
  GpuClientOptions options;
  ASSERT_OK_AND_ASSIGN(auto client, GetXlaPjrtGpuClient(options));
  EXPECT_EQ(client->platform_name(), "cuda");
}

}  // namespace xla
