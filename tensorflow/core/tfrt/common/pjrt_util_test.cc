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

#include "tensorflow/compiler/xla/pjrt/tfrt_cpu_pjrt_client.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/tfrt/common/global_state.h"
#include "tensorflow/core/tfrt/common/pjrt_state.h"
#include "tensorflow/tsl/lib/core/status_test_util.h"
#include "tensorflow/tsl/platform/status_matchers.h"
#include "tensorflow/tsl/protobuf/error_codes.pb.h"

namespace tensorflow {
namespace {

using ::testing::HasSubstr;
using ::tsl::testing::StatusIs;

TEST(PjRtUtilTest, SetGetAndDeletePjRtClient) {
  TF_ASSERT_OK(SetPjRtClientInTFGlobalResourceManager(
      DEVICE_CPU,
      xla::GetTfrtCpuClient(/*asynchronous=*/true, /*cpu_device_count=*/1)
          .value()));
  TF_ASSERT_OK_AND_ASSIGN(auto pjrt_client, GetPjRtClient(DEVICE_CPU));
  EXPECT_THAT(pjrt_client, ::testing::NotNull());
  TF_ASSERT_OK(
      DeletePjRtClientFromTFGlobalResourceManagerIfResourceExists(DEVICE_CPU));
}

TEST(PjRtStateResourceManagerTest, SetNullPjRtClient) {
  EXPECT_THAT(
      SetPjRtClientInTFGlobalResourceManager(DEVICE_CPU, nullptr),
      StatusIs(error::INVALID_ARGUMENT, HasSubstr("PJRT client is nullptr")));
}

TEST(PjRtUtilTest, DeleteNotExistPjRtClientOk) {
  TF_ASSERT_OK(SetPjRtClientInTFGlobalResourceManager(
      DEVICE_CPU,
      xla::GetTfrtCpuClient(/*asynchronous=*/true, /*cpu_device_count=*/1)
          .value()));
  TF_ASSERT_OK(
      DeletePjRtClientFromTFGlobalResourceManagerIfResourceExists(DEVICE_TPU));
}

TEST(PjRtUtilTest, DeleteNoPjRtStateOk) {
  ResourceMgr* rmgr = tfrt_global::GetTFGlobalResourceMgr();
  auto status = rmgr->Delete<PjRtState>(rmgr->default_container(),
                                        kPjRtStateResourceName);
  TF_ASSERT_OK(
      DeletePjRtClientFromTFGlobalResourceManagerIfResourceExists(DEVICE_TPU));
}

}  // namespace
}  // namespace tensorflow
