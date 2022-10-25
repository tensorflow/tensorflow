/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/tsl/distributed_runtime/rpc/grpc_util.h"

#include "grpcpp/grpcpp.h"
#include "tensorflow/tsl/platform/errors.h"
#include "tensorflow/tsl/platform/test.h"

namespace tsl {

TEST(PayloadSerialization, PayloadsAreTransmitted) {
  Status status = errors::InvalidArgument("invalid arg message");
  status.SetPayload("a", "\\xFF\\x02\\x03");
  Status status_recovered = FromGrpcStatus(ToGrpcStatus(status));

  ASSERT_TRUE(status_recovered.GetPayload("a").has_value());
  EXPECT_EQ(status_recovered.GetPayload("a").value(), "\\xFF\\x02\\x03");
}

TEST(PayloadSerialization, PayloadsCorrupted) {
  ::grpc::Status status(
      ::grpc::StatusCode::INVALID_ARGUMENT, "invalid arg message",
      "string that can not be serialized to the GrpcPayloadContainer proto");

  Status converted = FromGrpcStatus(status);
  EXPECT_TRUE(converted.GetPayload(kGrpcPayloadsLost).has_value());
}

}  // namespace tsl
