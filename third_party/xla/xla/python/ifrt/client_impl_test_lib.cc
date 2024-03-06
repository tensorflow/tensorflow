/* Copyright 2022 The OpenXLA Authors.

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

#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt/test_util.h"
#include "tsl/platform/test.h"

namespace xla {
namespace ifrt {
namespace {

using ::testing::IsEmpty;
using ::testing::Not;
using ::testing::NotNull;
using ::testing::SizeIs;

TEST(ClientImplTest, RuntimeTypeAndPlatform) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, test_util::GetClient());

  EXPECT_THAT(client->runtime_type(), Not(IsEmpty()));
  EXPECT_THAT(client->platform_name(), Not(IsEmpty()));
  EXPECT_THAT(client->platform_version(), Not(IsEmpty()));
  client->platform_id();
}

TEST(ClientImplTest, Devices) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, test_util::GetClient());

  ASSERT_GE(client->device_count(), 0);
  EXPECT_THAT(client->devices(), SizeIs(client->device_count()));

  ASSERT_GE(client->addressable_device_count(), 0);
  EXPECT_THAT(client->addressable_devices(),
              SizeIs(client->addressable_device_count()));

  for (Device* device : client->devices()) {
    TF_ASSERT_OK_AND_ASSIGN(auto* looked_up_device,
                            client->LookupDevice(device->id()));
    EXPECT_EQ(device, looked_up_device);
  }

  EXPECT_GE(client->process_index(), 0);
}

TEST(ClientImplTest, DefaultCompiler) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, test_util::GetClient());
  EXPECT_THAT(client->GetDefaultCompiler(), NotNull());
}

}  // namespace
}  // namespace ifrt
}  // namespace xla
