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

#include "tensorflow/core/tpu/tpu_embedding_errors.h"

#include <string>

#include "testing/base/public/gunit.h"
#include "absl/strings/string_view.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/errors.h"

namespace tensorflow::tpu {
namespace {

StatusOr<std::string> GenerateTFStatusOr(errors::Code code,
                                         absl::string_view value = "") {
  if (code == errors::Code::OK) {
    return std::string(value);
  } else {
    return errors::Create(code, /*message=*/"", /*payloads=*/{});
  }
}

TEST(TpuEmbeddingErrors, StatusOk) {
  constexpr absl::string_view kValue = "success";

  {
    const Status status = AppendTpuEmbeddingErrorPayload(OkStatus());
    TF_EXPECT_OK(status);
    EXPECT_FALSE(HasTpuEmbeddingErrorPayload(status));
  }

  {
    TF_ASSERT_OK_AND_ASSIGN(const std::string value,
                            AppendTpuEmbeddingErrorPayload(
                                GenerateTFStatusOr(errors::Code::OK, kValue)));
    EXPECT_EQ(value, kValue);
  }
}

TEST(TpuEmbeddingErrors, StatusFailed) {
  {
    const Status status =
        AppendTpuEmbeddingErrorPayload(errors::InvalidArgument(""));
    EXPECT_EQ(status.code(), error::Code::INVALID_ARGUMENT);
    EXPECT_TRUE(HasTpuEmbeddingErrorPayload(status));
  }

  {
    StatusOr<std::string> status_or = AppendTpuEmbeddingErrorPayload(
        GenerateTFStatusOr(errors::Code::RESOURCE_EXHAUSTED));
    EXPECT_FALSE(status_or.ok());
    EXPECT_EQ(status_or.status().code(), error::Code::RESOURCE_EXHAUSTED);
    EXPECT_TRUE(HasTpuEmbeddingErrorPayload(status_or.status()));
  }
}

}  // namespace
}  // namespace tensorflow::tpu
