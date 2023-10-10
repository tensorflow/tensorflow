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

#include "tensorflow/python/framework/offset_counter_helper.h"

#include <string>

#include "absl/strings/str_format.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/python/framework/op_reg_offset.pb.h"
#include "tsl/lib/core/status_test_util.h"

namespace tensorflow {
namespace {

TEST(OffsetCounterHelper, FindOpRegistationFromFile) {
  std::string content = R"code(
REGISTER_OP("Test>Op1");
REGISTER_OP("Test>Op2")
    .Input("input: int32")
    .Output("output: int32");
)code";
  Env* env = Env::Default();
  string fname;
  ASSERT_TRUE(env->LocalTempFilename(&fname));
  TF_ASSERT_OK(WriteStringToFile(env, fname, content));

  OpRegOffsets actual;
  TF_CHECK_OK(FindOpRegistationFromFile(fname, actual));

  // EqualsProto is not available in OSS. b/135192747
  EXPECT_EQ(actual.offsets(0).name(), "Test>Op1");
  EXPECT_EQ(actual.offsets(0).filepath(), fname);
  EXPECT_EQ(actual.offsets(0).start(), 13);
  EXPECT_EQ(actual.offsets(0).end(), 23);

  EXPECT_EQ(actual.offsets(1).name(), "Test>Op2");
  EXPECT_EQ(actual.offsets(1).filepath(), fname);
  EXPECT_EQ(actual.offsets(1).start(), 38);
  EXPECT_EQ(actual.offsets(1).end(), 48);
}

}  // namespace
}  // namespace tensorflow
