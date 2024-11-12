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
#include "xla/tsl/distributed_runtime/coordination/coordination_service_error_util.h"

#include <string>

#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "xla/tsl/protobuf/coordination_service.pb.h"
#include "tsl/platform/test.h"
namespace tsl {
namespace {
using ::tensorflow::BarrierError;
using ::tensorflow::CoordinatedTask;
using ::tensorflow::CoordinationServiceError;

TEST(CoordinationServiceErrorUtil, MakeCoordinationErrorWithEmptyPayload) {
  absl::Status error = absl::InternalError("Test Error");

  absl::Status coordination_error = MakeCoordinationError(error);

  EXPECT_EQ(coordination_error.code(), error.code());
  EXPECT_EQ(coordination_error.message(), error.message());
  // Payload exists but has no value.
  EXPECT_EQ(
      coordination_error.GetPayload(CoordinationErrorPayloadKey()).value(), "");
}

TEST(CoordinationServiceErrorUtil, MakeCoordinationErrorWithErrorOrigin) {
  absl::Status error = absl::InternalError("Test Error");
  CoordinatedTask source_task;
  source_task.set_job_name("test_worker");
  source_task.set_task_id(7);

  absl::Status coordination_error = MakeCoordinationError(error, source_task);

  EXPECT_EQ(coordination_error.code(), error.code());
  EXPECT_EQ(coordination_error.message(), error.message());
  CoordinationServiceError payload;
  // Explicit string conversion for open source builds.
  payload.ParseFromString(std::string(
      coordination_error.GetPayload(CoordinationErrorPayloadKey()).value()));
  EXPECT_EQ(payload.source_task().job_name(), source_task.job_name());
  EXPECT_EQ(payload.source_task().task_id(), source_task.task_id());
  EXPECT_EQ(payload.is_reported_error(), false);
}

TEST(CoordinationServiceErrorUtil, MakeCoordinationErrorWithUserReportedError) {
  absl::Status error = absl::InternalError("Test Error");
  CoordinatedTask source_task;
  source_task.set_job_name("test_worker");
  source_task.set_task_id(7);

  absl::Status coordination_error =
      MakeCoordinationError(error, source_task,
                            /*is_reported_error=*/true);

  EXPECT_EQ(coordination_error.code(), error.code());
  EXPECT_EQ(coordination_error.message(), error.message());
  CoordinationServiceError payload;
  // Explicit string conversion for open source builds.
  payload.ParseFromString(std::string(
      coordination_error.GetPayload(CoordinationErrorPayloadKey()).value()));
  EXPECT_EQ(payload.source_task().job_name(), source_task.job_name());
  EXPECT_EQ(payload.source_task().task_id(), source_task.task_id());
  EXPECT_EQ(payload.is_reported_error(), true);
}

TEST(CoordinationServiceErrorUtil, MakeCoordinationErrorWithPayload) {
  absl::Status error = absl::InternalError("Test Error");
  CoordinationServiceError payload;
  CoordinatedTask* source_task = payload.mutable_source_task();
  source_task->set_job_name("test_worker");
  source_task->set_task_id(7);
  payload.set_is_reported_error(true);

  absl::Status coordination_error = MakeCoordinationError(error, payload);

  EXPECT_EQ(coordination_error.code(), error.code());
  EXPECT_EQ(coordination_error.message(), error.message());
  CoordinationServiceError actual_payload;
  // Explicit string conversion for open source builds.
  actual_payload.ParseFromString(std::string(
      coordination_error.GetPayload(CoordinationErrorPayloadKey()).value()));
  EXPECT_EQ(actual_payload.source_task().job_name(),
            payload.source_task().job_name());
  EXPECT_EQ(actual_payload.source_task().task_id(),
            payload.source_task().task_id());
  EXPECT_EQ(actual_payload.is_reported_error(), payload.is_reported_error());
}

TEST(CoordinationServiceErrorUtil, MakeBarrierErrorWithPayload) {
  absl::Status barrier_error =
      MakeBarrierError(absl::InternalError("Test Error"), "barrier_id", 8);

  BarrierError payload;
  payload.ParseFromString(
      std::string(barrier_error.GetPayload(BarrierErrorPayloadKey()).value()));
  EXPECT_EQ(payload.barrier_id(), "barrier_id");
  EXPECT_EQ(payload.counter(), 8);
  EXPECT_EQ(GetBarrierCounterFromError(barrier_error), 8);
  // Payload exists but has no value.
  EXPECT_EQ(barrier_error.GetPayload(CoordinationErrorPayloadKey()).value(),
            "");
}

TEST(CoordinationServiceErrorUtil,
     TrimCoordinationErrorMessage_CoordinationError) {
  absl::Status error = MakeCoordinationError(absl::InternalError(
      "Coordination service has stopped. RecordHeartbeat() from task: "
      "/job:jax_worker/replica:0/task:2 failed. Additional GRPC error "
      "information from remote target coordination_service while calling "
      "/tensorflow.CoordinationService/Heartbeat::UNKNOWN:Error received from "
      "peer  "
      "{file:'third_party/grpc/src/core/lib/surface/filter_stack_call.cc', "
      "file_line:464, created_time:'2024-08-05T13:57:51.331198242-07:00', "
      "grpc_status:13, grpc_message:'Coordination service has stopped. "
      "RecordHeartbeat() from task: /job:jax_worker/replica:0/task:2 failed. "
      "'} "));

  absl::Status trimmed_error = TrimCoordinationErrorMessage(error);
  EXPECT_EQ(trimmed_error.code(), error.code());
  EXPECT_EQ(trimmed_error.message(),
            "Coordination service has stopped. RecordHeartbeat() from task: "
            "/job:jax_worker/replica:0/task:2 failed. \nRPC: "
            "/tensorflow.CoordinationService/Heartbeat");
  // Payload exists but has no value.
  EXPECT_EQ(trimmed_error.GetPayload(CoordinationErrorPayloadKey()).value(),
            "");
}

TEST(CoordinationServiceErrorUtil, TrimCoordinationErrorMessage_NetworkError) {
  absl::Status error = absl::UnavailableError(
      "failed to connect to all addresses; last error: UNKNOWN: "
      "ipv4:127.0.0.1:10001: Failed to connect to remote host: Connection "
      "refused. Additional GRPC error information from remote target "
      "coordination_service while calling "
      "/tensorflow.CoordinationService/Heartbeat::UNKNOWN:Error received from "
      "peer "
      "{file:'third_party/grpc/src/core/lib/surface/filter_stack_call.cc', "
      "file_line:464, created_time:'2024-08-05T13:57:53.123562608-07:00', "
      "grpc_status:14, grpc_message:'failed to connect to all addresses; last "
      "error: UNKNOWN: ipv4:127.0.0.1:10001: Failed to connect to remote host: "
      "Connection refused'} ");

  absl::Status trimmed_error = TrimCoordinationErrorMessage(error);
  auto message = trimmed_error.message();
  EXPECT_EQ(trimmed_error.code(), error.code());
  EXPECT_TRUE(absl::StrContains(message, "Check earlier logs"));
  EXPECT_TRUE(absl::StrContains(message, "preempted"));
  // Message is not duplicated.
  EXPECT_EQ(message.find("failed to connect"),
            message.rfind("failed to connect"))
      << trimmed_error;
}

}  // namespace
}  // namespace tsl
