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
#include "tensorflow/compiler/xla/pjrt/pjrt_api.h"

#include <gtest/gtest.h>
#include "tensorflow/tsl/lib/core/status_test_util.h"
#include "tensorflow/tsl/platform/status_matchers.h"
#include "tensorflow/tsl/protobuf/error_codes.pb.h"
namespace {

using ::testing::HasSubstr;
using ::tsl::testing::StatusIs;

TEST(PjRtApiTest, SetAndGetGlobalPjRtApi) {
  PJRT_Api api;

  TF_ASSERT_OK(pjrt::SetPjrtApi("CPU", &api));
  TF_ASSERT_OK_AND_ASSIGN(const PJRT_Api* output, pjrt::PjrtApi("CPU"));
  TF_ASSERT_OK_AND_ASSIGN(const PJRT_Api* output_lowercase,
                          pjrt::PjrtApi("cpu"));

  EXPECT_EQ(output, &api);
  EXPECT_EQ(output_lowercase, &api);
  EXPECT_THAT(pjrt::SetPjrtApi("CPU", &api),
              StatusIs(tensorflow::error::ALREADY_EXISTS,
                       HasSubstr("PJRT_Api already exists for device type")));
  EXPECT_THAT(pjrt::PjrtApi("TPU"),
              StatusIs(tensorflow::error::NOT_FOUND,
                       HasSubstr("PJRT_Api not found for device type tpu")));
}

}  // namespace
