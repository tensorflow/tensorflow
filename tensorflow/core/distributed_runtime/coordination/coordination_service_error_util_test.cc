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
#include "tensorflow/core/distributed_runtime/coordination/coordination_service_error_util.h"

#include <string>

#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/protobuf/coordination_service.pb.h"
namespace tensorflow {
namespace {

TEST(CoordinationServiceErrorUtil, MakeCoordinationErrorWithEmptyPayload) {
  Status error = errors::Internal("Test Error");

  Status coordination_error = MakeCoordinationError(error);

  EXPECT_EQ(coordination_error.code(), error.code());
  EXPECT_EQ(coordination_error.error_message(), error.error_message());
  // Payload exists but has no value.
  EXPECT_EQ(
      coordination_error.GetPayload(CoordinationErrorPayloadKey()).value(), "");
}

TEST(CoordinationServiceErrorUtil, MakeCoordinationErrorWithErrorOrigin) {
  Status error = errors::Internal("Test Error");
  constexpr absl::string_view job_name = "test_worker";
  int32_t task_id = 7;

  Status coordination_error = MakeCoordinationError(error, job_name, task_id);

  EXPECT_EQ(coordination_error.code(), error.code());
  EXPECT_EQ(coordination_error.error_message(), error.error_message());
  CoordinationServiceError payload;
  // Explicit string conversion for open source builds.
  payload.ParseFromString(std::string(
      coordination_error.GetPayload(CoordinationErrorPayloadKey()).value()));
  EXPECT_EQ(payload.job(), job_name);
  EXPECT_EQ(payload.task(), task_id);
  EXPECT_EQ(payload.is_reported_error(), false);
}

TEST(CoordinationServiceErrorUtil, MakeCoordinationErrorWithUserReportedError) {
  Status error = errors::Internal("Test Error");
  constexpr absl::string_view job_name = "test_worker";
  int32_t task_id = 7;

  Status coordination_error = MakeCoordinationError(error, job_name, task_id,
                                                    /*is_reported_error=*/true);

  EXPECT_EQ(coordination_error.code(), error.code());
  EXPECT_EQ(coordination_error.error_message(), error.error_message());
  CoordinationServiceError payload;
  // Explicit string conversion for open source builds.
  payload.ParseFromString(std::string(
      coordination_error.GetPayload(CoordinationErrorPayloadKey()).value()));
  EXPECT_EQ(payload.job(), job_name);
  EXPECT_EQ(payload.task(), task_id);
  EXPECT_EQ(payload.is_reported_error(), true);
}

}  // namespace
}  // namespace tensorflow
