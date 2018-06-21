/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

// Simple C++ test to exercise the GRPC capabilities of XLA.
//
// Launches an RPC service in a subprocess and connects to it over a socket
// using an RPCStub.
#include <memory>
#include <vector>

#include "grpcpp/create_channel.h"
#include "grpcpp/security/credentials.h"

#include "tensorflow/compiler/xla/client/client.h"
#include "tensorflow/compiler/xla/client/xla_client/xla_builder.h"
#include "tensorflow/compiler/xla/rpc/grpc_stub.h"
#include "tensorflow/compiler/xla/tests/literal_test_util.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/net.h"
#include "tensorflow/core/platform/subprocess.h"
#include "tensorflow/core/platform/test.h"

namespace xla {
namespace {

class GRPCClientTestBase : public ::testing::Test {
 protected:
  GRPCClientTestBase() {
    string test_srcdir = tensorflow::testing::TensorFlowSrcRoot();
    string service_main_path = tensorflow::io::JoinPath(
        test_srcdir, "compiler/xla/rpc/grpc_service_main_cpu");
    int port = tensorflow::internal::PickUnusedPortOrDie();
    subprocess_.SetProgram(
        service_main_path,
        {service_main_path, tensorflow::strings::Printf("--port=%d", port)});
    subprocess_.SetChannelAction(tensorflow::CHAN_STDOUT,
                                 tensorflow::ACTION_DUPPARENT);
    subprocess_.SetChannelAction(tensorflow::CHAN_STDERR,
                                 tensorflow::ACTION_DUPPARENT);
    CHECK(subprocess_.Start());
    LOG(INFO) << "Launched subprocess";

    auto channel =
        ::grpc::CreateChannel(tensorflow::strings::Printf("localhost:%d", port),
                              ::grpc::InsecureChannelCredentials());
    channel->WaitForConnected(gpr_time_add(
        gpr_now(GPR_CLOCK_REALTIME), gpr_time_from_seconds(10, GPR_TIMESPAN)));
    LOG(INFO) << "Channel to server is connected on port " << port;

    xla_service_ = grpc::XlaService::NewStub(channel);
    stub_.reset(new GRPCStub(xla_service_.get()));
    client_.reset(new Client(stub_.get()));
  }

  ~GRPCClientTestBase() override {
    LOG(INFO) << "Killing subprocess";
    subprocess_.Kill(SIGKILL);
  }

  tensorflow::SubProcess subprocess_;
  std::unique_ptr<grpc::XlaService::Stub> xla_service_;
  std::unique_ptr<GRPCStub> stub_;
  std::unique_ptr<Client> client_;
};

TEST_F(GRPCClientTestBase, ItsAlive) {
  ASSERT_NE(xla_service_, nullptr);
  ASSERT_NE(stub_, nullptr);
  ASSERT_NE(client_, nullptr);
}

TEST_F(GRPCClientTestBase, AxpyTenValues) {
  XlaBuilder builder("axpy_10");
  auto alpha = builder.ConstantR0<float>(3.1415926535);
  auto x = builder.ConstantR1<float>(
      {-1.0, 1.0, 2.0, -2.0, -3.0, 3.0, 4.0, -4.0, -5.0, 5.0});
  auto y = builder.ConstantR1<float>(
      {5.0, -5.0, -4.0, 4.0, 3.0, -3.0, -2.0, 2.0, 1.0, -1.0});
  auto ax = builder.Mul(alpha, x);
  auto axpy = builder.Add(ax, y);

  std::vector<float> expected = {
      1.85840735, -1.85840735, 2.28318531,   -2.28318531,  -6.42477796,
      6.42477796, 10.56637061, -10.56637061, -14.70796327, 14.70796327};
  std::unique_ptr<Literal> expected_literal =
      Literal::CreateR1<float>(expected);
  TF_ASSERT_OK_AND_ASSIGN(auto computation, builder.Build());
  TF_ASSERT_OK_AND_ASSIGN(auto result_literal, client_->ExecuteAndTransfer(
                                                   computation, {}, nullptr));
  EXPECT_TRUE(LiteralTestUtil::Near(*expected_literal, *result_literal,
                                    ErrorSpec(0.0001)));
}

}  // namespace
}  // namespace xla
