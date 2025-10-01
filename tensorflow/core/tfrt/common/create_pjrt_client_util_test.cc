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
#include "tensorflow/core/tfrt/common/create_pjrt_client_util.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/str_cat.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"  // IWYU pragma: keep
#include "xla/tsl/protobuf/error_codes.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tsl/platform/status_matchers.h"

namespace tensorflow {
namespace {

using ::testing::HasSubstr;

TEST(CreatePjRtClientTest, GetNotExistPjRtClientNotImplemented) {
  EXPECT_THAT(GetOrCreatePjRtClient(DEVICE_CPU),
              absl_testing::StatusIs(
                  error::NOT_FOUND,
                  HasSubstr(absl::StrCat("The PJRT client factory of `",
                                         DEVICE_CPU, "` is not registered"))));
}

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
TEST(CreatePjRtClientTest, GetNotExistGpuPjRtClient) {
  TF_ASSERT_OK_AND_ASSIGN(auto pjrt_client,
                          GetOrCreatePjRtClient(DEVICE_XLA_GPU));
  EXPECT_THAT(pjrt_client, ::testing::NotNull());
}
#endif

}  // namespace
}  // namespace tensorflow
