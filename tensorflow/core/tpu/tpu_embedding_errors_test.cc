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

#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/protobuf/error_codes.pb.h"
#include "tensorflow/core/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace tensorflow::tpu {
namespace {

using absl::Status;
using absl::StatusOr;

StatusOr<std::string> GenerateTFStatusOr(absl::StatusCode code,
                                         absl::string_view value = "") {
  if (code == absl::StatusCode::kOk) {
    return std::string(value);
  } else {
    return absl::Status(code, value);
  }
}

TEST(TpuEmbeddingErrors, StatusOk) {
  constexpr absl::string_view kValue = "success";

  {
    const Status status = AppendTpuEmbeddingErrorPayload(absl::OkStatus());
    TF_EXPECT_OK(status);
    EXPECT_FALSE(HasTpuEmbeddingErrorPayload(status));
    EXPECT_FALSE(HasTpuEmbeddingErrorMessage(status));
  }

  {
    TF_ASSERT_OK_AND_ASSIGN(const std::string value,
                            AppendTpuEmbeddingErrorPayload(GenerateTFStatusOr(
                                absl::StatusCode::kOk, kValue)));
    EXPECT_EQ(value, kValue);
  }
}

TEST(TpuEmbeddingErrors, StatusFailed) {
  {
    const Status status =
        AppendTpuEmbeddingErrorPayload(errors::InvalidArgument(""));
    EXPECT_EQ(status.code(), error::Code::INVALID_ARGUMENT);
    EXPECT_TRUE(HasTpuEmbeddingErrorPayload(status));
    EXPECT_TRUE(HasTpuEmbeddingErrorMessage(status));
  }

  {
    StatusOr<std::string> status_or = AppendTpuEmbeddingErrorPayload(
        GenerateTFStatusOr(absl::StatusCode::kResourceExhausted));
    EXPECT_FALSE(status_or.ok());
    const Status& status = status_or.status();
    EXPECT_EQ(status.code(), error::Code::RESOURCE_EXHAUSTED);
    EXPECT_TRUE(HasTpuEmbeddingErrorPayload(status));
    EXPECT_TRUE(HasTpuEmbeddingErrorMessage(status));
  }
}

}  // namespace
}  // namespace tensorflow::tpu
