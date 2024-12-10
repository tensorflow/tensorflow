/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/tfrt/common/pjrt_util.h"

#include <memory>
#include <utility>

#include "xla/pjrt/plugin/xla_cpu/cpu_client_options.h"
#include "xla/pjrt/plugin/xla_cpu/xla_cpu_pjrt_client.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/protobuf/error_codes.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/tfrt/common/pjrt_state.h"
#include "tsl/platform/status_matchers.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/test.h"

namespace tensorflow {
namespace {

using ::testing::ElementsAre;
using ::testing::HasSubstr;
using ::tsl::testing::StatusIs;

TEST(PjRtUtilTest, SetGetAndDeletePjRtClient) {
  xla::CpuClientOptions options;
  options.asynchronous = true;
  options.cpu_device_count = 1;
  TF_ASSERT_OK(SetPjRtClientInTFGlobalResourceManager(
      DEVICE_CPU, xla::GetXlaPjrtCpuClient(options).value()));
  TF_ASSERT_OK_AND_ASSIGN(auto pjrt_client, GetPjRtClient(DEVICE_CPU));
  EXPECT_THAT(pjrt_client, ::testing::NotNull());
}

TEST(PjRtStateResourceManagerTest, SetNullPjRtClient) {
  EXPECT_THAT(
      SetPjRtClientInTFGlobalResourceManager(DEVICE_CPU, nullptr),
      StatusIs(error::INVALID_ARGUMENT, HasSubstr("PJRT client is nullptr")));
}

TEST(PjRtGpuClientCreationInfoTest, SetAndGet) {
  auto info = std::make_unique<PjRtGpuClientCreationInfo>();
  info->allowed_devices.insert(123);
  TF_ASSERT_OK(
      SetPjRtGpuClientCreationInfoInTFGlobalResourceManager(std::move(info)));

  TF_ASSERT_OK_AND_ASSIGN(PjRtGpuClientCreationInfo * retrieved_info,
                          GetPjRtGpuClientCreationInfo());

  EXPECT_THAT(retrieved_info->allowed_devices, ElementsAre(123));
}

}  // namespace
}  // namespace tensorflow
