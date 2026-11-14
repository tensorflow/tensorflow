/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/distributed_runtime/rpc/grpc_server_lib.h"

#include <grpc/grpc.h>

#include <cstddef>
#include <memory>
#include <string>

#include "grpcpp/support/channel_arguments.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

class GrpcServerLibTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Clear env vars before each test to ensure clean state.
    unsetenv("TF_GRPC_MAX_PENDING_REQUESTS");
    unsetenv("TF_GRPC_MAX_PENDING_REQUESTS_HARD_LIMIT");
  }
};

TEST_F(GrpcServerLibTest, MaybeCreateMaxPendingRequestsOption_NotSet) {
  auto option = GrpcServer::MaybeCreateMaxPendingRequestsOption();
  EXPECT_EQ(option, nullptr);
}

#if defined(GRPC_ARG_SERVER_MAX_PENDING_REQUESTS) || \
    defined(GRPC_ARG_SERVER_MAX_PENDING_REQUESTS_HARD_LIMIT)
TEST_F(GrpcServerLibTest, MaybeCreateMaxPendingRequestsOption_SetSoftOnly) {
  setenv("TF_GRPC_MAX_PENDING_REQUESTS", "100", 1);
  auto option = GrpcServer::MaybeCreateMaxPendingRequestsOption();
  ASSERT_NE(option, nullptr);

  ::grpc::ChannelArguments args;
  option->UpdateArguments(&args);

  grpc_channel_args c_args = args.c_channel_args();
  bool found_soft = false;
  for (size_t i = 0; i < c_args.num_args; ++i) {
    if (std::string(c_args.args[i].key) == "grpc.server.max_pending_requests") {
      EXPECT_EQ(c_args.args[i].value.integer, 100);
      found_soft = true;
    }
  }
  EXPECT_TRUE(found_soft);
}

TEST_F(GrpcServerLibTest, MaybeCreateMaxPendingRequestsOption_SetHardOnly) {
  setenv("TF_GRPC_MAX_PENDING_REQUESTS_HARD_LIMIT", "200", 1);
  auto option = GrpcServer::MaybeCreateMaxPendingRequestsOption();
  ASSERT_NE(option, nullptr);

  ::grpc::ChannelArguments args;
  option->UpdateArguments(&args);

  grpc_channel_args c_args = args.c_channel_args();
  bool found_hard = false;
  for (size_t i = 0; i < c_args.num_args; ++i) {
    if (std::string(c_args.args[i].key) ==
        "grpc.server.max_pending_requests_hard_limit") {
      EXPECT_EQ(c_args.args[i].value.integer, 200);
      found_hard = true;
    }
  }
  EXPECT_TRUE(found_hard);
}

TEST_F(GrpcServerLibTest, MaybeCreateMaxPendingRequestsOption_SetBoth) {
  setenv("TF_GRPC_MAX_PENDING_REQUESTS", "100", 1);
  setenv("TF_GRPC_MAX_PENDING_REQUESTS_HARD_LIMIT", "200", 1);
  auto option = GrpcServer::MaybeCreateMaxPendingRequestsOption();
  ASSERT_NE(option, nullptr);

  ::grpc::ChannelArguments args;
  option->UpdateArguments(&args);

  grpc_channel_args c_args = args.c_channel_args();
  bool found_soft = false;
  bool found_hard = false;
  for (size_t i = 0; i < c_args.num_args; ++i) {
    if (std::string(c_args.args[i].key) == "grpc.server.max_pending_requests") {
      EXPECT_EQ(c_args.args[i].value.integer, 100);
      found_soft = true;
    }
    if (std::string(c_args.args[i].key) ==
        "grpc.server.max_pending_requests_hard_limit") {
      EXPECT_EQ(c_args.args[i].value.integer, 200);
      found_hard = true;
    }
  }
  EXPECT_TRUE(found_soft);
  EXPECT_TRUE(found_hard);
}
#endif

TEST_F(GrpcServerLibTest, MaybeCreateMaxPendingRequestsOption_InvalidValues) {
  setenv("TF_GRPC_MAX_PENDING_REQUESTS", "invalid", 1);
  setenv("TF_GRPC_MAX_PENDING_REQUESTS_HARD_LIMIT", "-5", 1);
  auto option = GrpcServer::MaybeCreateMaxPendingRequestsOption();
  EXPECT_EQ(option, nullptr);
}

}  // namespace
}  // namespace tensorflow
